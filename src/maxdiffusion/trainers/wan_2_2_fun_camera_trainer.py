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

# Trainer for WAN 2.2 **Fun-5B camera-control** (PAI), fine-tuning from the
# official checkpoint on records produced by tpu_tfrecord_encoder/
# encode_concat_camera.py (with latent_condition).
#
# Mirrors the certified INFERENCE contract (wan_pipeline_2_2_fun_camera):
#   model input  = concat([noisy_latents(48), mask(4), latent_condition(48)]) = 100ch
#   camera       = per-frame (extrinsic, intrinsic) -> Plucker -> packed [B,24,F_lat,H,W]
#                  (computed ON DEVICE per step from the tiny stored matrices)
#   timesteps    = per-token: first-latent-frame tokens get t=0 (clean first
#                  frame), all others get the sampled t -- the TI2V convention.
#   loss         = flow-match MSE on latent frames 1.. (frame 0 is the clean
#                  conditioning frame, excluded, per DiffSynth FlowMatchSFTLoss).
#
# Everything else (optimizer, schedule, logging, checkpoint cadence) is the
# verified WAN 2.1/2.2-dense training loop, unchanged. No dtype changes.

import functools

import jax
import jax.numpy as jnp
import jaxopt
import tensorflow as tf
from flax import nnx

from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.wan.camera_plucker import build_control_camera_latents
from maxdiffusion.trainers.wan_trainer import WanTrainer
from maxdiffusion import max_utils

from jax.sharding import PartitionSpec as P


class Wan2_2FunCameraTrainer(WanTrainer):
  """Fine-tunes PAI Wan2.2-Fun-5B-Control-Camera with camera conditioning."""

  def _get_checkpointer(self):
    return WanCheckpointer2_2_FunCamera(config=self.config)

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    return {
        "latents": data_sharding,
        "latent_condition": data_sharding,
        "encoder_hidden_states": data_sharding,
        "camera_extrinsic": data_sharding,
        "camera_intrinsic": data_sharding,
    }

  def get_eval_data_shardings(self, mesh):
    shardings = self.get_data_shardings(mesh)
    shardings["timesteps"] = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    return shardings

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Fun camera training only supports dataset_type=tfrecord with cache_latents_text_encoder_outputs=True"
      )

    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "latent_condition": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        "camera_extrinsic": tf.io.FixedLenFeature([], tf.string),
        "camera_intrinsic": tf.io.FixedLenFeature([], tf.string),
    }
    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

    def prepare_sample(features):
      out = {
          "latents": tf.io.parse_tensor(features["latents"], out_type=tf.float32),
          "latent_condition": tf.io.parse_tensor(features["latent_condition"], out_type=tf.float32),
          "encoder_hidden_states": tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32),
          "camera_extrinsic": tf.io.parse_tensor(features["camera_extrinsic"], out_type=tf.float32),
          "camera_intrinsic": tf.io.parse_tensor(features["camera_intrinsic"], out_type=tf.float32),
      }
      if not is_training:
        out["timesteps"] = features["timesteps"]
      return out

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
    p_t, p_h, p_w = pipeline.transformer.config.patch_size
    return jax.jit(
        functools.partial(
            train_step,
            scheduler=pipeline.scheduler,
            config=self.config,
            patch_hw=(p_h, p_w),
        ),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    raise NotImplementedError("Fun camera eval step not wired yet; run with eval_every=-1.")


def _build_first_frame_mask(latents: jax.Array) -> jax.Array:
  """The 4-channel y mask, channel-first [B, 4, F_lat, h, w].

  Derivation from the inference pipeline's pixel-space construction
  (prepare_fun_camera_y_latents): the padded pixel mask is 1 for the first
  frame (repeated x4 by temporal compression) and 0 elsewhere; folding groups
  of 4 into channels makes every channel of latent frame 0 equal to 1 and
  everything else 0.
  """
  b, _, f_lat, h, w = latents.shape
  mask = jnp.zeros((b, 4, f_lat, h, w), dtype=latents.dtype)
  return mask.at[:, :, 0:1].set(1.0)


def _per_token_timesteps(timesteps: jax.Array, f_lat: int, tokens_per_frame: int) -> jax.Array:
  """[B] -> [B, seq] with first-latent-frame tokens at t=0 (TI2V convention).

  Token order matches WanModel: jax.lax.collapse over (F_lat, h/p, w/p),
  frame-major, so the first `tokens_per_frame` tokens belong to latent frame 0.
  """
  b = timesteps.shape[0]
  seq = f_lat * tokens_per_frame
  t_tok = jnp.broadcast_to(timesteps[:, None], (b, seq)).astype(jnp.float32)
  return t_tok.at[:, :tokens_per_frame].set(0.0)


def train_step(state, data, rng, scheduler_state, scheduler, config, patch_hw):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config, patch_hw)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config, patch_hw):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)                  # [B, 48, F_lat, h, w]
    latent_condition = data["latent_condition"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)

    bsz = latents.shape[0]
    f_lat, h_lat, w_lat = latents.shape[2], latents.shape[3], latents.shape[4]

    # Camera conditioning: tiny stored matrices -> packed Plucker, on device.
    control_camera_latents = build_control_camera_latents(
        data["camera_extrinsic"],
        data["camera_intrinsic"],
        height=config.height,
        width=config.width,
        moment_scale=float(getattr(config, "camera_moment_scale", 1.0)),
        dtype=config.weights_dtype,
    )

    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)
    # First latent frame is the CLEAN conditioning frame (TI2V): never noised.
    noisy_latents = noisy_latents.at[:, :, 0:1].set(latents[:, :, 0:1])

    # y-conditioning: [mask(4) | latent_condition(48)], concat after the noisy
    # latents -> the 100-channel Fun camera-control input.
    mask = _build_first_frame_mask(latents)
    hidden_states = jnp.concatenate([noisy_latents, mask, latent_condition], axis=1)

    tokens_per_frame = (h_lat // patch_hw[0]) * (w_lat // patch_hw[1])
    t_tok = _per_token_timesteps(timesteps, f_lat, tokens_per_frame)

    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=hidden_states,
          timestep=t_tok,
          encoder_hidden_states=encoder_hidden_states,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
          control_camera_latents_input=control_camera_latents,
      )

    with jax.named_scope("loss"):
      # Frame 0 is conditioning, not a prediction target (DiffSynth convention).
      loss = (training_target[:, :, 1:] - model_pred[:, :, 1:]) ** 2
      if not config.disable_training_weights:
        training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
        loss = loss * training_weight
      loss = jnp.mean(loss)

    return loss

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = jax.tree_util.tree_reduce(
      lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))),
      grads,
      initializer=-1.0,
  )

  # Adapter-only gradient norm: the camera path's learning signal, separated
  # from the (much larger) base model. Flat-zero here would mean the camera
  # conditioning is disconnected; growth precedes instability.
  def _adapter_sq(path, arr):
    is_adapter = any("control_adapter" in str(p) for p in path)
    return jnp.sum(arr.astype(jnp.float32) ** 2) if is_adapter else jnp.float32(0.0)

  adapter_grad_norm = jnp.sqrt(
      jax.tree_util.tree_reduce(
          lambda a, b: a + b,
          jax.tree_util.tree_map_with_path(_adapter_sq, grads),
          initializer=jnp.float32(0.0),
      )
  )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
          "learning/adapter_grad_norm": adapter_grad_norm,
      },
      "scalars": {},
  }

  new_state = state.apply_gradients(grads=grads)
  return new_state, scheduler_state, metrics, new_rng
