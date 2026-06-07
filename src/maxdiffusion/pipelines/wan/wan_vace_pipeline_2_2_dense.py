# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VACE pipeline for WAN 2.2 Dense (TI2V-5B) single-transformer model.

Unlike WAN 2.1 VACE, there is no official pretrained VACE variant of the
WAN 2.2 TI2V-5B model.  This pipeline therefore bootstraps the VACE model by:
  1. Loading the base WanModel config from the pretrained HF repo.
  2. Injecting vace_layers / vace_in_channels from the training config.
  3. Loading base transformer weights from HF; VACE branch weights are
     randomly initialised (they will be trained from scratch).
"""

from functools import partial
import math
from typing import Optional

import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated
from ... import max_logging
from ...models.wan.wan_utils import load_wan_transformer
from ...models.wan.transformers.transformer_wan import WanModel
from ...models.wan.transformers.transformer_wan_vace import WanVACEModel
from ...models.modeling_flax_pytorch_utils import torch2jax
from ...pyconfig import HyperParameters
from .wan_pipeline import cast_with_exclusion
from .wan_pipeline_2_2_dense import WanPipeline2_2_Dense


def _parse_vace_layers(value):
  """Parse vace_layers from config: accepts list/tuple or comma-sep string."""
  if isinstance(value, (list, tuple)):
    return list(int(x) for x in value)
  return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _create_sharded_logical_transformer_2_2_dense(
    devices_array: np.ndarray,
    mesh,
    rngs: nnx.Rngs,
    config: HyperParameters,
    restored_checkpoint=None,
    subfolder: str = "transformer",
):
  """Create a sharded WanVACEModel for WAN 2.2 dense.

  On fresh training (no checkpoint) the base WanModel config is loaded from HF
  and VACE fields are injected from the training config.  VACE branch weights
  start randomly initialised; base transformer weights are loaded from HF.
  """

  def create_model(rngs: nnx.Rngs, wan_config: dict):
    return WanVACEModel(**wan_config, rngs=rngs)

  if restored_checkpoint:
    wan_config = restored_checkpoint["wan_config"]
  else:
    # WAN 2.2 TI2V-5B has no VACE pretrained variant: load the base model
    # config and inject VACE fields from the training config.
    wan_config = WanModel.load_config(config.pretrained_model_name_or_path, subfolder=subfolder)
    # Remove config-metadata keys that WanVACEModel.__init__ doesn't accept.
    wan_config = {k: v for k, v in wan_config.items() if not k.startswith("_")}

    vace_layers = _parse_vace_layers(getattr(config, "vace_layers", "0,5,10,15,20,25"))
    vace_in_channels = int(getattr(config, "vace_in_channels", 96))
    wan_config["vace_layers"] = vace_layers
    wan_config["vace_in_channels"] = vace_in_channels
    max_logging.log(f"VACE 2.2 dense: vace_layers={vace_layers}, vace_in_channels={vace_in_channels}")

  wan_config["mesh"] = mesh
  wan_config["dtype"] = config.activations_dtype
  wan_config["weights_dtype"] = config.weights_dtype
  wan_config["attention"] = config.attention
  wan_config["precision"] = get_precision(config)
  wan_config["flash_block_sizes"] = get_flash_block_sizes(config)
  wan_config["remat_policy"] = config.remat_policy
  wan_config["names_which_can_be_saved"] = config.names_which_can_be_saved
  wan_config["names_which_can_be_offloaded"] = config.names_which_can_be_offloaded
  wan_config["flash_min_seq_length"] = config.flash_min_seq_length
  wan_config["dropout"] = config.dropout
  wan_config["mask_padding_tokens"] = config.mask_padding_tokens
  wan_config["scan_layers"] = config.scan_layers
  wan_config["enable_jax_named_scopes"] = config.enable_jax_named_scopes
  wan_config["use_base2_exp"] = config.use_base2_exp
  wan_config["use_experimental_scheduler"] = config.use_experimental_scheduler
  wan_config["debug_vace_numerics"] = str(getattr(config, "vace_debug_print", False)).lower() == "true"
  wan_config["lora_rank"] = int(getattr(config, "lora_rank", 0))
  wan_config["lora_alpha"] = float(getattr(config, "lora_alpha", 0.0))

  p_model_factory = partial(create_model, wan_config=wan_config)
  wan_vace_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(wan_vace_transformer, nnx.Param, ...)

  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = flax.linen.logical_to_mesh_sharding(
      logical_state_spec, mesh, config.logical_axis_rules
  )
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  if restored_checkpoint:
    if "params" in restored_checkpoint["wan_state"]:
      params = restored_checkpoint["wan_state"]["params"]
    else:
      params = restored_checkpoint["wan_state"]
  else:
    # Load base transformer weights. VACE branch keys are absent in the HF
    # checkpoint; load_wan_transformer leaves them at their random-init values.
    params = load_wan_transformer(
        config.wan_transformer_pretrained_model_name_or_path,
        eval_shapes=params,
        device="cpu",
        num_layers=wan_config["num_layers"],
        scan_layers=config.scan_layers,
        subfolder=subfolder,
    )

  params = jax.tree_util.tree_map_with_path(
      lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=config.weights_dtype),
      params,
  )
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint and path[-1] == "value":
      path = path[:-1]
    try:
      path = path[:1] + (int(path[1]),) + path[2:]
    except Exception:
      pass
    sharding = logical_state_sharding[path].value
    try:
      state[path].value = device_put_replicated(val, sharding)
    except Exception as e:
      max_logging.log(f"Failed device_put_replicated for {path}: {e}; trying process_allgather")
      val_on_host = jax.experimental.multihost_utils.process_allgather(val, tiled=True)
      state[path].value = device_put_replicated(val_on_host, sharding)
      del val_on_host

  # Materialize VACE params absent from the HF checkpoint.
  # nnx.eval_shape leaves them as ShapeDtypeStruct; initialize them to match
  # DiffSynth's official VACE init (PyTorch defaults = LeCun / Kaiming uniform).
  #   kernel → LeCun uniform: U(-1/√fan_in, 1/√fan_in), fan_in = ∏ shape[:-1]
  #   bias   → zeros
  #   scale  → ones  (LayerNorm / RMSNorm identity)
  #   other (adaln_scale_shift_table) → N(0, 1/√dim)
  key = jax.random.PRNGKey(0)
  for path, var in state.items():
    if not isinstance(var.value, jax.Array):
      key, subkey = jax.random.split(key)
      sds = var.value  # jax.ShapeDtypeStruct
      sharding = logical_state_sharding[path].value
      name = path[-1]
      if name == "scale":
        init_val = jnp.ones(sds.shape, dtype=sds.dtype)
      elif name == "bias":
        init_val = jnp.zeros(sds.shape, dtype=sds.dtype)
      elif name == "kernel":
        fan_in = max(1, int(math.prod(sds.shape[:-1])))
        bound = 1.0 / math.sqrt(fan_in)
        init_val = jax.random.uniform(subkey, sds.shape, dtype=jnp.float32, minval=-bound, maxval=bound).astype(sds.dtype)
      else:
        # adaln_scale_shift_table: scaled normal matching WanTransformerBlock init
        std = 1.0 / math.sqrt(max(1, sds.shape[-1]))
        init_val = (jax.random.normal(subkey, sds.shape, dtype=jnp.float32) * std).astype(sds.dtype)
      state[path].value = device_put_replicated(init_val, sharding)
      max_logging.log(f"VACE fresh-init ({name}): {path} shape={sds.shape}")

  state = nnx.from_flat_state(state)
  return nnx.merge(graphdef, state, rest_of_state)


class VaceWanPipeline2_2_Dense(WanPipeline2_2_Dense):
  """WAN 2.2 Dense pipeline extended with VACE conditioning.

  Loads the base WAN 2.2 TI2V-5B weights and adds a randomly-initialised VACE
  branch (vace_patch_embedding + vace_blocks) configured via vace_layers and
  vace_in_channels in the training config.

  Inference via __call__ runs without VACE conditioning (standard generation).
  VACE conditioning during inference can be added by overriding __call__.
  """

  @classmethod
  def load_transformer(
      cls,
      devices_array: np.ndarray,
      mesh,
      rngs: nnx.Rngs,
      config: HyperParameters,
      restored_checkpoint=None,
      subfolder: str = "transformer",
  ):
    with mesh:
      return _create_sharded_logical_transformer_2_2_dense(
          devices_array, mesh, rngs, config, restored_checkpoint, subfolder
      )

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only and load_transformer:
      transformer = cls.load_transformer(
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          rngs=common_components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder="transformer",
      )
    pipeline = cls(
        tokenizer=common_components["tokenizer"],
        text_encoder=common_components["text_encoder"],
        transformer=transformer,
        vae=common_components["vae"],
        vae_cache=common_components["vae_cache"],
        scheduler=common_components["scheduler"],
        scheduler_state=common_components["scheduler_state"],
        devices_array=common_components["devices_array"],
        mesh=common_components["mesh"],
        vae_mesh=common_components["vae_mesh"],
        vae_logical_axis_rules=common_components["vae_logical_axis_rules"],
        config=config,
    )
    return pipeline, transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline, transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    if transformer is not None:
      pipeline.transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline
