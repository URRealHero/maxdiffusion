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

# V2V-1c: v2v concat TRAINER for the HyDRA-baseline reimplementation.
#
# Mirrors HyDRA/train_hydra.py training_step + _freeze_and_mark_trainables:
#   input latents = concat([cond(20 latent frames), tgt(20 latent frames)], axis=2)
#                   -> [B, 16, 40, h, w]  (cond first half, tgt second half)
#   noise         = randn_like(latents); noisy = apply_flow_match(noise, latents, t)
#                   over ALL 40 frames, then the cond (first) half is FORCED clean
#                   (noisy[:, :, :tgt_len] = latents[:, :, :tgt_len]).
#   camera        = cam_emb_con [B, 20, 12] + cam_emb_tgt [B, 20, 12], injected per
#                   latent-frame half INSIDE the blocks (V2V-1a); NO y-concat, NO
#                   control-adapter, NO extra input channels (Wan2.1 in_dim=out_dim=16).
#   loss          = flow-match MSE on the TGT (second) half only (frames tgt_len:).
#   trainable     = only params whose path contains cam_encoder_con / cam_encoder_tgt
#                   / projector / attn1 (self-attention); everything else FROZEN via
#                   optax.set_to_zero (no optimizer state for the frozen base).
#
# Everything else (optimizer build, schedule, logging, checkpoint cadence) is the
# verified WAN 2.1 training loop, unchanged.

import functools
import json
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jaxopt
import optax
import tensorflow as tf
from flax import nnx

from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.wan_trainer import WanTrainer
from maxdiffusion import max_logging, max_utils
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1

import orbax.checkpoint as ocp

from jax.sharding import PartitionSpec as P


# The 4 trainable-param substrings (HyDRA train_hydra.py:96, but maxdiffusion names
# the self-attention module `attn1`, not `self_attn`). Base params (patch_embedding,
# cross-attn attn2, ffn, condition_embedder, norms, proj_out) stay frozen.
V2V_DEFAULT_TRAINABLE_SUBSTRINGS = ("cam_encoder_con", "cam_encoder_tgt", "projector", "attn1")


def v2v_trainable_substrings(config):
  """The trainable-param substrings for the v2v partial fine-tune.

  Reads `v2v_trainable_param_substrings` (comma-separated); falls back to
  `lora_trainable_param_substrings`, then to the HyDRA-baseline default.
  """
  raw = str(getattr(config, "v2v_trainable_param_substrings", "") or "").strip()
  if not raw:
    raw = str(getattr(config, "lora_trainable_param_substrings", "") or "").strip()
  if not raw:
    return list(V2V_DEFAULT_TRAINABLE_SUBSTRINGS)
  return [s.strip() for s in raw.split(",") if s.strip()]


def _path_str(path):
  """Stable string for an nnx flat-state path tuple (save/overlay must agree)."""
  return "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path)


class WanCheckpointerV2V(WanCheckpointer2_1):
  """Wan2.1 base loader (with the v2v_concat scaffold filled by the load path) +
  the trainable-substring freezing optimizer (optax.multi_transform).

  Reuses WanCheckpointer2_1.load_checkpoint / load_diffusers_checkpoint, so the
  transformer is built through create_sharded_logical_transformer, which threads
  config.v2v_concat -> WanModel and fresh-inits cam_encoder_{con,tgt}/projector
  (V2V-1b). Only the optimizer construction is specialized here.
  """

  def _create_optimizer(self, model, config, learning_rate):
    """Freeze everything except the v2v trainable substrings (mirror
    WanCheckpointer2_2_FunCamera._create_optimizer): only params whose path
    contains one of the substrings are trained; the rest are frozen via
    optax.set_to_zero (no Adam state, so the frozen base costs no extra memory)."""
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    base_tx = max_utils.create_optimizer(config, learning_rate_scheduler)

    substrings = v2v_trainable_substrings(config)
    if not substrings:
      return base_tx, learning_rate_scheduler  # full fine-tune

    _, params, _ = nnx.split(model, nnx.Param, ...)

    def _label(path, _leaf):
      ps = jax.tree_util.keystr(path)
      return "train" if any(s in ps for s in substrings) else "freeze"

    labels = jax.tree_util.tree_map_with_path(_label, params)
    leaves = jax.tree_util.tree_leaves(labels)
    n_train = sum(1 for v in leaves if v == "train")
    max_logging.log(
        f"V2V partial fine-tune: training {n_train}/{len(leaves)} param tensors "
        f"(substrings={substrings}); base frozen via set_to_zero."
    )
    tx = optax.multi_transform({"train": base_tx, "freeze": optax.set_to_zero()}, labels)
    return tx, learning_rate_scheduler

  def load_diffusers_checkpoint(self):
    return WanPipeline2_1.from_pretrained(self.config)

  def _overlay_v2v_checkpoint(self, saved_flat):
    """Load the Wan2.1 base pipeline (with the fresh v2v scaffold) and overlay the
    saved trainable (adapter/attn1) params on top."""
    pipeline = self.load_diffusers_checkpoint()
    transformer = pipeline.transformer
    state = nnx.state(transformer, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    applied = 0
    for p, v in flat.items():
      sp = _path_str(p)
      if sp in saved_flat:
        target = v.value
        new_val = jnp.asarray(saved_flat[sp], dtype=target.dtype)
        sharding = getattr(target, "sharding", None)
        v.value = jax.device_put(new_val, sharding) if sharding is not None else new_val
        applied += 1
    if applied != len(saved_flat):
      missing = set(saved_flat) - {_path_str(p) for p in flat}
      raise ValueError(
          f"V2V overlay mismatch: applied {applied}/{len(saved_flat)} saved tensors; "
          f"unmatched saved keys: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
      )
    nnx.update(transformer, nnx.from_flat_state(flat))
    max_logging.log(f"V2V-only checkpoint: overlaid {applied} trainable tensors onto the Wan2.1 base.")
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_1, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      if restored_checkpoint.wan_config.get("_v2v_only", False):
        pipeline = self._overlay_v2v_checkpoint(restored_checkpoint.wan_state)
      else:
        pipeline = WanPipeline2_1.from_checkpoint(self.config, restored_checkpoint)
        if "opt_state" in restored_checkpoint.wan_state.keys():
          opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default Wan2.1 v2v pipeline.")
      pipeline = self.load_diffusers_checkpoint()
    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_1, train_states):
    """Full-state save, or (when save_lora_only=True) a compact save of only the
    v2v trainable params."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    wan_config = config_to_json(pipeline.transformer)

    substrings = v2v_trainable_substrings(self.config)
    save_v2v_only = bool(getattr(self.config, "save_lora_only", False)) and len(substrings) > 0
    if save_v2v_only:
      source = getattr(train_states, "params", train_states)
      flat = dict(nnx.to_flat_state(source))
      v2v_flat = {_path_str(p): v.value for p, v in flat.items() if any(s in _path_str(p) for s in substrings)}
      if len(v2v_flat) == 0:
        raise ValueError(
            f"save_lora_only=True but no params matched {substrings}; refusing to save an empty checkpoint."
        )
      wan_config["_v2v_only"] = True
      wan_config["_v2v_substrings"] = substrings
      state_save = ocp.args.StandardSave(v2v_flat)
      max_logging.log(f"  V2V-only checkpoint: {len(v2v_flat)} trainable tensors (base frozen, not saved).")
    else:
      state_save = ocp.args.StandardSave(train_states)

    items = {
        "wan_config": ocp.args.JsonSave(wan_config),
        "wan_state": state_save,
    }
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")


def v2v_concat_loss(model, batch, noise_rng, dropout_rng, timesteps, scheduler, config, _zero_cond_half_pred=False):
  """The V2V-1c forward + flow-match loss (HyDRA train_hydra.py:109-138).

  `batch` carries cond_latents/tgt_latents [B, 16, 20, h, w], cam_emb_con/tgt
  [B, 20, 12], encoder_hidden_states [B, 512, 4096].

  `_zero_cond_half_pred` is a GATE-C3 hook (default False, unused in training):
  when True, the cond (first) half of the model prediction is zeroed before the
  loss. Because the loss reads only the tgt (second) half, the result must be
  identical -- proving the loss is computed on the tgt half alone.
  """
  cond_latents = batch["cond_latents"].astype(config.weights_dtype)  # [B, 16, 20, h, w]
  tgt_latents = batch["tgt_latents"].astype(config.weights_dtype)  # [B, 16, 20, h, w]
  cam_emb_con = batch["cam_emb_con"].astype(config.weights_dtype)  # [B, 20, 12]
  cam_emb_tgt = batch["cam_emb_tgt"].astype(config.weights_dtype)  # [B, 20, 12]
  encoder_hidden_states = batch["encoder_hidden_states"].astype(config.weights_dtype)  # [B, 512, 4096]

  # V2V frame-concat along the latent-frame axis: cond FIRST, tgt SECOND.
  latents = jnp.concatenate([cond_latents, tgt_latents], axis=2)  # [B, 16, 40, h, w]
  tgt_len = latents.shape[2] // 2  # 20: [:tgt_len] = cond (kept clean), [tgt_len:] = tgt (loss).

  noise = jax.random.normal(key=noise_rng, shape=latents.shape, dtype=latents.dtype)
  noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)
  # FORCE the cond (first) half clean (HyDRA:114). The cond frames are the given
  # context; only the tgt frames are denoised.
  noisy_latents = noisy_latents.at[:, :, :tgt_len].set(latents[:, :, :tgt_len])

  with jax.named_scope("forward_pass"):
    model_pred = model(
        hidden_states=noisy_latents,  # 16-ch noisy latents only (NO y-concat / control-adapter).
        timestep=timesteps,  # scalar [B] timestep.
        encoder_hidden_states=encoder_hidden_states,
        cam_emb_con=cam_emb_con,
        cam_emb_tgt=cam_emb_tgt,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )

  if _zero_cond_half_pred:
    model_pred = model_pred.at[:, :, :tgt_len].set(0.0)

  with jax.named_scope("loss"):
    # Loss ONLY on the tgt (second) half (HyDRA:134-138).
    loss = (training_target[:, :, tgt_len:] - model_pred[:, :, tgt_len:]) ** 2
    if not config.disable_training_weights:
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight
    loss = jnp.mean(loss)

  return loss


class WanV2VConcatTrainer(WanTrainer):
  """Fine-tunes Wan2.1-T2V-1.3B with the V2V-1a per-block camera scaffold, on
  frame-concat (cond|tgt) latents, training only the camera/projector/self-attn
  params (HyDRA baseline)."""

  def _get_checkpointer(self):
    return WanCheckpointerV2V(config=self.config)

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    return {
        "cond_latents": data_sharding,
        "tgt_latents": data_sharding,
        "cam_emb_con": data_sharding,
        "cam_emb_tgt": data_sharding,
        "encoder_hidden_states": data_sharding,
    }

  def get_eval_data_shardings(self, mesh):
    # No eval-loss path (like the Fun camera trainer): the encoded dataset carries
    # no per-record `timesteps` feature. Returned only for API symmetry.
    return self.get_data_shardings(mesh)

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "V2V training only supports dataset_type=tfrecord with cache_latents_text_encoder_outputs=True"
      )

    feature_description = {
        "cond_latents": tf.io.FixedLenFeature([], tf.string),
        "tgt_latents": tf.io.FixedLenFeature([], tf.string),
        "cam_emb_con": tf.io.FixedLenFeature([], tf.string),
        "cam_emb_tgt": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    }

    def prepare_sample(features):
      return {
          "cond_latents": tf.io.parse_tensor(features["cond_latents"], out_type=tf.float32),
          "tgt_latents": tf.io.parse_tensor(features["tgt_latents"], out_type=tf.float32),
          "cam_emb_con": tf.io.parse_tensor(features["cam_emb_con"], out_type=tf.float32),
          "cam_emb_tgt": tf.io.parse_tensor(features["cam_emb_tgt"], out_type=tf.float32),
          "encoder_hidden_states": tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32),
      }

    return make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample,
        is_training=is_training,
    )

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(train_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    # No eval loss (dataset has no per-record timesteps). None => base loop skips it.
    return None


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    bsz = data["cond_latents"].shape[0]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    return v2v_concat_loss(model, data, new_rng, dropout_rng, timesteps, scheduler, config)

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = jax.tree_util.tree_reduce(
      lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))),
      grads,
      initializer=-1.0,
  )

  # Trainable (v2v adapter/self-attn) gradient norm, separated from the frozen
  # base. Flat-zero here would mean the camera/projector path is disconnected.
  substrings = tuple(v2v_trainable_substrings(config))

  def _trainable_sq(path, arr):
    ps = jax.tree_util.keystr(path)
    is_trainable = any(s in ps for s in substrings)
    return jnp.sum(arr.astype(jnp.float32) ** 2) if is_trainable else jnp.float32(0.0)

  v2v_grad_norm = jnp.sqrt(
      jax.tree_util.tree_reduce(
          lambda a, b: a + b,
          jax.tree_util.tree_map_with_path(_trainable_sq, grads),
          initializer=jnp.float32(0.0),
      )
  )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
          "learning/v2v_grad_norm": v2v_grad_norm,
      },
      "scalars": {},
  }

  new_state = state.apply_gradients(grads=grads)
  return new_state, scheduler_state, metrics, new_rng
