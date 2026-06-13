"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import glob
import json
import torch
import jax
import jax.numpy as jnp
from maxdiffusion import max_logging
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)

CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH = "lightx2v/Wan2.1-T2V-14B-CausVid"
WAN_21_FUSION_X_MODEL_NAME_OR_PATH = "vrgamedevgirl84/Wan14BT2VFusioniX"


def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def _normalize_animate_list_key(key):
  """Convert flattened animate list names into nnx.List-style tuple paths."""
  if not key:
    return key

  if isinstance(key[0], str) and key[0].startswith("face_adapter_"):
    adapter_idx = int(key[0].split("_")[-1])
    return ("face_adapter", adapter_idx) + key[1:]

  if len(key) >= 2 and key[0] == "motion_encoder" and isinstance(key[1], str) and key[1].startswith("motion_network_"):
    layer_idx = int(key[1].split("_")[-1])
    return ("motion_encoder", "motion_network", layer_idx) + key[2:]

  return key


def rename_for_nnx(key):
  new_key = key
  if "norm_k" in key or "norm_q" in key:
    new_key = key[:-1] + ("scale",)
  return new_key


def rename_for_custom_trasformer(key):
  renamed_pt_key = key.replace("model.diffusion_model.", "")

  renamed_pt_key = renamed_pt_key.replace("head.modulation", "scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("head.head", "proj_out")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "condition_embedder.text_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "condition_embedder.text_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "condition_embedder.time_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "condition_embedder.time_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_projection_1", "condition_embedder.time_proj")

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace("control_adapter.residual_blocks.", "control_adapter.residual_blocks_")
  renamed_pt_key = renamed_pt_key.replace("self_attn", "attn1")
  renamed_pt_key = renamed_pt_key.replace("cross_attn", "attn2")
  renamed_pt_key = renamed_pt_key.replace(".q.", ".query.")
  renamed_pt_key = renamed_pt_key.replace(".k.", ".key.")
  renamed_pt_key = renamed_pt_key.replace(".v.", ".value.")
  renamed_pt_key = renamed_pt_key.replace(".o.", ".proj_attn.")
  renamed_pt_key = renamed_pt_key.replace("ffn_0", "ffn.act_fn.proj")
  renamed_pt_key = renamed_pt_key.replace("ffn_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace(".modulation", ".scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("norm3", "norm2.layer_norm")

  return renamed_pt_key


def get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=40):
  block_index = None
  if scan_layers:
    if len(pt_tuple_key) >= 2 and pt_tuple_key[0] == "blocks":
      block_index = int(pt_tuple_key[1])
      pt_tuple_key = ("blocks",) + pt_tuple_key[2:]

  flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, random_flax_state_dict, scan_layers)

  flax_key = rename_for_nnx(flax_key)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers and block_index is not None:
    if flax_key in flax_state_dict:
      new_tensor = flax_state_dict[flax_key]
    else:
      new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype)
    flax_tensor = new_tensor.at[block_index].set(flax_tensor)
  return flax_key, flax_tensor


def _build_random_flax_state_dict(eval_shapes):
  flattened_dict = flatten_dict(eval_shapes)
  random_flax_state_dict = {}
  for key, value in flattened_dict.items():
    random_flax_state_dict[tuple(str(item) for item in key)] = value
  return random_flax_state_dict


def _rename_common_wan_transformer_key(renamed_pt_key: str) -> str:
  # DiffSynth/Captain-Safari / PAI Fun checkpoint aliases. These are no-ops
  # for ordinary Diffusers keys and let the same loader handle official Fun
  # camera-control safetensors.
  renamed_pt_key = renamed_pt_key.replace("model.diffusion_model.", "")
  renamed_pt_key = renamed_pt_key.replace("head.modulation", "scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("head.head", "proj_out")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "condition_embedder.text_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "condition_embedder.text_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "condition_embedder.time_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "condition_embedder.time_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_projection_1", "condition_embedder.time_proj")
  renamed_pt_key = renamed_pt_key.replace("self_attn", "attn1")
  renamed_pt_key = renamed_pt_key.replace("cross_attn", "attn2")
  renamed_pt_key = renamed_pt_key.replace(".q.", ".query.")
  renamed_pt_key = renamed_pt_key.replace(".k.", ".key.")
  renamed_pt_key = renamed_pt_key.replace(".v.", ".value.")
  renamed_pt_key = renamed_pt_key.replace(".o.", ".proj_attn.")
  renamed_pt_key = renamed_pt_key.replace("ffn_0", "ffn.act_fn.proj")
  renamed_pt_key = renamed_pt_key.replace("ffn_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace(".modulation", ".adaln_scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("norm3", "norm2.layer_norm")

  if "condition_embedder" in renamed_pt_key:
    renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "time_embedder.linear_1")
    renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "time_embedder.linear_2")
    renamed_pt_key = renamed_pt_key.replace("time_projection_1", "time_proj")
    renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "text_embedder.linear_1")
    renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "text_embedder.linear_2")

  if "image_embedder" in renamed_pt_key:
    if "net.0.proj" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net.0.proj", "net_0")
    elif "net_0.proj" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net_0.proj", "net_0")
    if "net.2" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net.2", "net_2")
    renamed_pt_key = renamed_pt_key.replace("norm1", "norm1.layer_norm")
    if "norm1" in renamed_pt_key or "norm2" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("weight", "scale")
      renamed_pt_key = renamed_pt_key.replace("kernel", "scale")

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace("control_adapter.residual_blocks.", "control_adapter.residual_blocks_")
  renamed_pt_key = renamed_pt_key.replace(".scale_shift_table", ".adaln_scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("to_out_0", "proj_attn")
  renamed_pt_key = renamed_pt_key.replace("ffn.net_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace("ffn.net_0", "ffn.act_fn")
  if "norm2" in renamed_pt_key and "norm2.layer_norm" not in renamed_pt_key:
    renamed_pt_key = renamed_pt_key.replace("norm2", "norm2.layer_norm")

  return renamed_pt_key


def _rename_wan_animate_pt_tuple_key(pt_key: str):
  renamed_pt_key = _rename_common_wan_transformer_key(rename_key(pt_key))
  is_motion_custom_weight = _is_motion_encoder_custom_weight(pt_key)

  renamed_pt_key = renamed_pt_key.replace(".activation.bias", ".act_fn.bias")
  if is_motion_custom_weight and renamed_pt_key.endswith(".kernel"):
    renamed_pt_key = renamed_pt_key[:-7] + ".weight"

  return tuple(renamed_pt_key.split(".")), is_motion_custom_weight


def get_wan_animate_key_and_value(
    pt_tuple_key,
    tensor,
    flax_state_dict,
    random_flax_state_dict,
    scan_layers,
    is_motion_custom_weight=False,
    num_layers=40,
):
  if is_motion_custom_weight:
    flax_key = _normalize_animate_list_key(_tuple_str_to_int(pt_tuple_key))
    return flax_key, tensor

  flax_key, flax_tensor = get_key_and_value(
      pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
  )
  flax_key = _normalize_animate_list_key(flax_key)
  return flax_key, flax_tensor


def load_fusionx_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, filename="Wan14BT2VFusioniX_fp16_.safetensors")
      tensors = {}
      with safe_open(ckpt_shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))

      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)

        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))

        flax_key, flax_tensor = get_key_and_value(
            pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
        )
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_causvid_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, filename="causal_model.pt")
      loaded_state_dict = torch.load(ckpt_shard_path)

      tensors = {}
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in loaded_state_dict.items():
        tensor = torch2jax(tensor)
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = get_key_and_value(
            pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
        )
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
):
  if pretrained_model_name_or_path == CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH:
    return load_causvid_transformer(pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers)
  elif pretrained_model_name_or_path == WAN_21_FUSION_X_MODEL_NAME_OR_PATH:
    return load_fusionx_transformer(pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers)
  else:
    return load_base_wan_transformer(
        pretrained_model_name_or_path, eval_shapes, device, hf_download, num_layers, scan_layers, subfolder
    )


def load_base_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
):
  device = jax.local_devices(backend=device)[0]
  filename = "diffusion_pytorch_model.safetensors.index.json"
  local_files = False
  local_model_files_are_absolute = False
  if os.path.isdir(pretrained_model_name_or_path):
    local_files = True
    search_dir = os.path.join(pretrained_model_name_or_path, subfolder)
    index_file_path = os.path.join(search_dir, filename)
    if os.path.isfile(index_file_path):
      with open(index_file_path, "r") as f:
        index_dict = json.load(f)
      model_files = sorted(set(index_dict["weight_map"].values()))
    else:
      model_files = sorted(glob.glob(os.path.join(search_dir, "diffusion_pytorch_model*.safetensors")))
      if not model_files and subfolder:
        model_files = sorted(glob.glob(os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model*.safetensors")))
      if not model_files:
        raise FileNotFoundError(
            f"Could not find {filename} or diffusion_pytorch_model*.safetensors under "
            f"{search_dir} (or repo root fallback)."
        )
      local_model_files_are_absolute = True
  else:
    remote_subfolder = subfolder
  if hf_download and not local_files:
    # Prefer the standard diffusers sharded index under subfolder, but Fun camera
    # checkpoints are commonly published as a single safetensors file at repo root.
    try:
      index_file_path = hf_hub_download(
          pretrained_model_name_or_path,
          subfolder=subfolder,
          filename=filename,
      )
      with open(index_file_path, "r") as f:
        index_dict = json.load(f)
      model_files = sorted(set(index_dict["weight_map"].values()))
    except Exception as exc:
      single_file = "diffusion_pytorch_model.safetensors"
      try:
        hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=single_file)
        model_files = [single_file]
      except Exception:
        remote_subfolder = ""
        try:
          hf_hub_download(pretrained_model_name_or_path, filename=single_file)
          model_files = [single_file]
        except Exception:
          raise exc
  with jax.default_device(device):
    tensors = {}
    for model_file in model_files:
      if local_files:
        ckpt_shard_path = model_file if local_model_files_are_absolute else os.path.join(pretrained_model_name_or_path, subfolder, model_file)
      else:
        ckpt_shard_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=remote_subfolder or None,
            filename=model_file,
        )
      # now get all the filenames for the model that need downloading
      max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")

      if ckpt_shard_path is not None:
        with safe_open(ckpt_shard_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    # turn all block numbers to strings just for matching weights.
    # Later they will be turned back to ints.
    random_flax_state_dict = _build_random_flax_state_dict(eval_shapes)
    for pt_key, tensor in tensors.items():
      # The diffusers implementation explicitly describes this key in keys to be ignored.
      if "norm_added_q" in pt_key:
        continue
      renamed_pt_key = rename_key(pt_key)
      renamed_pt_key = _rename_common_wan_transformer_key(renamed_pt_key)
      pt_tuple_key = tuple(renamed_pt_key.split("."))
      flax_key, flax_tensor = get_key_and_value(
          pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers
      )
      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def _is_motion_encoder_custom_weight(pt_key: str) -> bool:
  """Returns True for FlaxMotionConv2d/FlaxMotionLinear weight keys that must NOT be renamed to kernel."""
  prefixes = (
      "motion_encoder.conv_in.",
      "motion_encoder.conv_out.",
  )
  if any(pt_key.startswith(p) for p in prefixes) and pt_key.endswith(".weight"):
    return True
  if "motion_encoder.res_blocks." in pt_key and pt_key.endswith(".weight"):
    return True
  if "motion_encoder.motion_network." in pt_key and pt_key.endswith(".weight"):
    return True
  return False


def load_wan_animate_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "transformer",
):
  """Loads WanAnimate transformer weights from a HuggingFace checkpoint.

  Handles the additional key mappings for:
    - pose_patch_embedding (nnx.Conv3d → kernel)
    - motion_encoder.* (FlaxMotionConv2d/FlaxMotionLinear → keep as 'weight', no transpose)
    - activation.bias → act_fn.bias  (FusedLeakyReLU bias remapping)
    - face_encoder.* (nnx.Conv/Linear → standard rename to kernel)
    - face_adapter.* (nnx.Linear → standard rename to kernel)
  """
  device = jax.local_devices(backend=device)[0]
  filename = "diffusion_pytorch_model.safetensors.index.json"
  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
    index_file_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    if not os.path.isfile(index_file_path):
      raise FileNotFoundError(f"File {index_file_path} not found for local directory.")
    local_files = True
  elif hf_download:
    index_file_path = hf_hub_download(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        filename=filename,
    )
  with jax.default_device(device):
    with open(index_file_path, "r") as f:
      index_dict = json.load(f)
    model_files = set()
    for key in index_dict["weight_map"].keys():
      model_files.add(index_dict["weight_map"][key])

    model_files = list(model_files)
    tensors = {}
    for model_file in model_files:
      if local_files:
        ckpt_shard_path = os.path.join(pretrained_model_name_or_path, subfolder, model_file)
      else:
        ckpt_shard_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=model_file)
      max_logging.log(f"Load and port {pretrained_model_name_or_path} {subfolder} on {device}")
      if ckpt_shard_path is not None:
        with safe_open(ckpt_shard_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    random_flax_state_dict = _build_random_flax_state_dict(eval_shapes)

    for pt_key, tensor in tensors.items():
      if "norm_added_q" in pt_key:
        continue

      pt_tuple_key, is_motion_custom_weight = _rename_wan_animate_pt_tuple_key(pt_key)
      flax_key, flax_tensor = get_wan_animate_key_and_value(
          pt_tuple_key,
          tensor,
          flax_state_dict,
          random_flax_state_dict,
          scan_layers,
          is_motion_custom_weight=is_motion_custom_weight,
          num_layers=num_layers,
      )

      flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def _remap_wan_2p2_vae_key(key: str) -> str:
  """Map Wan2.2 high-compression VAE diffusers keys to the NNX VAE tree."""
  key = key.replace("downsamplers_", "downsamplers.")
  key = key.replace("upsamplers_", "upsamplers.")

  # Diffusers stores samplers separately; AutoencoderKLWan2p2 appends them
  # as the final residual block entry in each down/up block.
  key = key.replace(".downsamplers.0.", ".resnets.2.")
  key = key.replace(".downsampler.", ".resnets.2.")
  key = key.replace(".upsamplers.0.", ".resnets.3.")
  key = key.replace(".upsampler.", ".resnets.3.")
  return key


def _map_middle_index(prefix: str, idx: str, suffix: str) -> str:
  if idx == "0":
    return f"{prefix}.mid_block.resnets.0.{suffix}"
  if idx == "1":
    return f"{prefix}.mid_block.attentions.0.{suffix}"
  if idx == "2":
    return f"{prefix}.mid_block.resnets.1.{suffix}"
  return f"{prefix}.mid_block.{idx}.{suffix}"


def _remap_wan_2p2_pth_vae_key(key: str) -> str:
  """Map original Wan2.2_VAE.pth keys to the NNX VAE tree.

  PAI/Captain-Safari publish the original DiffSynth-style VAE weights where
  blocks are named `downsamples/middle/head`. MaxDiffusion's JAX port names the
  same modules `down_blocks/mid_block/norm_out/conv_out`.
  """
  if key.startswith("model."):
    key = key[len("model.") :]
  if key.startswith("conv1."):
    key = "quant_conv." + key[len("conv1.") :]
  elif key.startswith("conv2."):
    key = "post_quant_conv." + key[len("conv2.") :]

  parts = key.split(".")
  if len(parts) >= 2 and parts[0] in ("encoder", "decoder"):
    prefix = parts[0]
    if parts[1] == "conv1":
      parts[1] = "conv_in"
    elif parts[1] == "head" and len(parts) >= 3:
      if parts[2] == "0":
        parts = [prefix, "norm_out"] + parts[3:]
      elif parts[2] == "2":
        parts = [prefix, "conv_out"] + parts[3:]
    elif parts[1] == "middle" and len(parts) >= 4:
      key = _map_middle_index(prefix, parts[2], ".".join(parts[3:]))
      parts = key.split(".")
    elif prefix == "encoder" and parts[1] == "downsamples" and len(parts) >= 5:
      block_idx = parts[2]
      if parts[3] == "downsamples":
        parts = ["encoder", "down_blocks", block_idx, "resnets", parts[4]] + parts[5:]
      elif parts[3] == "avg_shortcut":
        parts = ["encoder", "down_blocks", block_idx, "avg_shortcut"] + parts[4:]
    elif prefix == "decoder" and parts[1] == "upsamples" and len(parts) >= 5:
      block_idx = parts[2]
      if parts[3] == "upsamples":
        parts = ["decoder", "up_blocks", block_idx, "resnets", parts[4]] + parts[5:]
      elif parts[3] == "avg_shortcut":
        parts = ["decoder", "up_blocks", block_idx, "avg_shortcut"] + parts[4:]

  key = ".".join(parts)
  key = key.replace(".residual.0.", ".norm1.")
  key = key.replace(".residual.2.", ".conv1.")
  key = key.replace(".residual.3.", ".norm2.")
  key = key.replace(".residual.6.", ".conv2.")
  key = key.replace(".shortcut.", ".conv_shortcut.")
  return key


# ---------------------------------------------------------------------------
# LoRA loading: DiffSynth/Captain-Safari PEFT safetensors -> our nnx LoRA params.
#
# CS keys (per layer N):
#   blocks.N.self_attn.{q,k,v,o}.lora_{A,B}.default.weight   -> attn1.lora_{q,k,v,o}
#   blocks.N.cross_attn.{q,k,v,o}.lora_{A,B}.default.weight  -> attn2.lora_{q,k,v,o}
#   blocks.N.ffn.0.lora_{A,B}.default.weight                 -> ffn.act_fn.lora_ffn0
#   blocks.N.ffn.2.lora_{A,B}.default.weight                 -> ffn.lora_ffn2
# Every CS weight is PyTorch [out, in]; our nnx Linear kernel is [in, out] -> transpose.
# Non-LoRA CS keys (memory_*, norm_memory, memory_cross_attn, memory_retriever,
# memory_emb) belong to the Captain-Safari memory augmentation we do NOT model;
# they are skipped.
# ---------------------------------------------------------------------------
_CS_ATTN_TO_NNX = {"self_attn": "attn1", "cross_attn": "attn2"}
_CS_PROJ_TO_NNX = {"q": "lora_q", "k": "lora_k", "v": "lora_v", "o": "lora_o"}


def _cs_lora_key_to_nnx_path(key: str):
  """Map one CS LoRA tensor key -> (block_index, nnx_path_tuple) or None to skip."""
  if "lora_" not in key:
    return None
  parts = key.split(".")
  # expect blocks.N.<module...>.lora_{A,B}.default.weight
  if parts[0] != "blocks" or parts[-1] != "weight" or parts[-2] != "default":
    return None
  ab = parts[-3]  # lora_A or lora_B
  if ab not in ("lora_A", "lora_B"):
    return None
  try:
    block_index = int(parts[1])
  except (IndexError, ValueError):
    return None
  module = parts[2]
  leaf = (ab, "kernel")
  if module in _CS_ATTN_TO_NNX:  # self_attn / cross_attn
    proj = parts[3]
    if proj not in _CS_PROJ_TO_NNX:
      return None
    return block_index, ("blocks", _CS_ATTN_TO_NNX[module], _CS_PROJ_TO_NNX[proj]) + leaf
  if module == "ffn":
    which = parts[3]  # "0" or "2"
    if which == "0":
      return block_index, ("blocks", "ffn", "act_fn", "lora_ffn0") + leaf
    if which == "2":
      return block_index, ("blocks", "ffn", "lora_ffn2") + leaf
    return None
  # memory_cross_attn / norm_memory / memory_* etc. -> skip
  return None


def init_wan_lora_params(eval_shapes: dict, seed: int = 0):
  """Fresh LoRA init for every lora_ param in the model (PEFT convention).

  lora_A -> variance_scaling(1/3, fan_in, uniform) == U(-1/sqrt(fan_in), +...);
  lora_B -> zeros (delta starts at 0). Returns {flat_path_tuple: jnp.array}.
  These must be filled because eval_shape leaves them abstract and the base
  checkpoint does not contain LoRA weights.
  """
  shapes = {tuple(str(p) for p in k): v for k, v in flatten_dict(eval_shapes).items()}
  out = {}
  key = jax.random.key(seed)
  for path, shaped in shapes.items():
    if "lora_A" in path and path[-1] == "kernel":
      key, sub = jax.random.split(key)
      fan_in = shaped.shape[-2]  # [.., in, rank] (or [L, in, rank] under scan)
      bound = 1.0 / (fan_in ** 0.5)
      out[path] = jax.random.uniform(sub, shaped.shape, jnp.float32, -bound, bound)
    elif "lora_B" in path and path[-1] == "kernel":
      out[path] = jnp.zeros(shaped.shape, dtype=jnp.float32)
  return out


def load_wan_lora(lora_path: str, eval_shapes: dict, scan_layers: bool = True, num_layers: int = 30):
  """Load a DiffSynth/CS LoRA safetensors into a flax LoRA param dict.

  Returns {nnx_path_tuple: jnp.array} with weights transposed to [in, out] and,
  when scan_layers, stacked to [num_layers, in, out]. eval_shapes is the model's
  param pytree (flattened) used to validate shapes and detect missing adapters.
  """
  if not os.path.isfile(lora_path):
    raise FileNotFoundError(f"LoRA file not found: {lora_path} (expected a local .safetensors path)")
  ckpt_path = lora_path
  max_logging.log(f"Loading WAN LoRA from {ckpt_path}")

  shapes = {tuple(str(p) for p in k): v for k, v in flatten_dict(eval_shapes).items()}

  out = {}
  skipped = 0
  n_lora = 0
  with safe_open(ckpt_path, framework="pt") as f:
    for key in f.keys():
      mapped = _cs_lora_key_to_nnx_path(key)
      if mapped is None:
        skipped += 1
        continue
      block_index, nnx_path = mapped
      tensor = torch2jax(f.get_tensor(key)).astype(jnp.float32)
      tensor = jnp.swapaxes(tensor, -1, -2)  # PyTorch [out,in] -> nnx [in,out]
      n_lora += 1
      if scan_layers:
        target = shapes.get(nnx_path)
        if target is None:
          raise ValueError(f"LoRA target {nnx_path} not found in model params (lora_rank set?)")
        if nnx_path not in out:
          out[nnx_path] = jnp.zeros(target.shape, dtype=jnp.float32)  # [num_layers, in, out]
        out[nnx_path] = out[nnx_path].at[block_index].set(tensor)
      else:
        # non-scan: per-layer modules live under blocks.<idx>.*
        path = ("blocks", str(block_index)) + nnx_path[1:]
        out[path] = tensor

  max_logging.log(f"WAN LoRA: mapped {n_lora} tensors -> {len(out)} stacked params, skipped {skipped} non-LoRA keys")
  return out


def load_wan_vae(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    is_wan_2p2: bool = False,
    subfolder: str = "vae",
    filename: str = "diffusion_pytorch_model.safetensors",
):
  device = jax.devices(device)[0]
  ckpt_path = None
  if os.path.isfile(pretrained_model_name_or_path):
    ckpt_path = pretrained_model_name_or_path
  elif os.path.isdir(pretrained_model_name_or_path):
    ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename) if subfolder else os.path.join(pretrained_model_name_or_path, filename)
    if not os.path.isfile(ckpt_path):
      raise FileNotFoundError(f"File {ckpt_path} not found for local directory.")
  elif hf_download:
    ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder or None, filename=filename)
  max_logging.log(f"Load and port {pretrained_model_name_or_path} VAE on {device}")
  with jax.default_device(device):
    if ckpt_path is not None:
      tensors = {}
      is_pth_vae = ckpt_path.endswith(".pth")
      if is_pth_vae:
        loaded = torch.load(ckpt_path, map_location="cpu")
        if isinstance(loaded, dict) and "model_state" in loaded:
          loaded = loaded["model_state"]
        if isinstance(loaded, dict) and "state_dict" in loaded:
          loaded = loaded["state_dict"]
        for k, v in loaded.items():
          if torch.is_tensor(v):
            tensors[k] = torch2jax(v)
      else:
        with safe_open(ckpt_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = _remap_wan_2p2_pth_vae_key(pt_key) if is_pth_vae else rename_key(pt_key)
        if is_wan_2p2 and not is_pth_vae:
          renamed_pt_key = _remap_wan_2p2_vae_key(renamed_pt_key)
        # Order matters
        renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
        renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
        renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

        renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_in.weight", "conv_in.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv_out.bias", "conv_out.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv_out.weight", "conv_out.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
        renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
        renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
        renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
        renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("time_conv.bias", "time_conv.conv.bias")
        renamed_pt_key = renamed_pt_key.replace("time_conv.weight", "time_conv.conv.weight")
        renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
        renamed_pt_key = renamed_pt_key.replace("conv_shortcut", "conv_shortcut.conv")
        if "decoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
          renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
        if "encoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
        flax_key = _tuple_str_to_int(flax_key)
        flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
    else:
      raise FileNotFoundError(f"Path {ckpt_path} was not found")

    return flax_state_dict
