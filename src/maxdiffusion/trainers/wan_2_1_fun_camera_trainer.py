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

# Trainer for **Wan2.1-Fun-V1.1-1.3B-Control-Camera** (PAI), fine-tuning from
# the official checkpoint on records produced by tpu_tfrecord_encoder/
# encode_concat_camera.py --wan-version 2.1, merged with a `clip_feature`
# field ([257,1280] CLIP ViT-H of the conditioning frame).
#
# Mirrors the official DiffSynth training contract (config hash ac6a5aa7,
# in_dim=32, has_image_input=True):
#   model input  = concat([noisy_latents(16), y(16)]) = 32ch; y is ZERO except
#                  latent frame 0 = VAE(first frame). NO mask channels.
#   clip         = clip_feature -> encoder_hidden_states_image -> img_emb ->
#                  257 image tokens prepended to text in every cross-attn.
#   camera       = per-frame (extrinsic, intrinsic) -> Plucker -> packed
#                  [B,24,F_lat,H,W] on device per step (SimpleAdapter dscale 8).
#   timesteps    = SCALAR per batch (no per-token hold), loss over ALL frames
#                  (diffsynth FlowMatchSFTLoss: add_noise on the whole tensor).
#
# Everything else (loader resharding, optimizer masking, logging, checkpoint
# cadence) is inherited from the certified 2.2 Fun-camera trainer.

import functools
import time

import jax
import jax.numpy as jnp
import jaxopt
import tensorflow as tf
from flax import nnx

from maxdiffusion.checkpointing.wan_checkpointer_2_1_fun_camera import WanCheckpointer2_1_FunCamera
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.wan.camera_plucker import build_control_camera_latents
from maxdiffusion.trainers.wan_2_2_fun_camera_trainer import Wan2_2FunCameraTrainer, _save_eval_samples
from maxdiffusion import max_logging

from jax.sharding import PartitionSpec as P


class Wan2_1FunCameraTrainer(Wan2_2FunCameraTrainer):
  """Fine-tunes PAI Wan2.1-Fun-V1.1-1.3B-Control-Camera with camera conditioning."""

  def _get_checkpointer(self):
    return WanCheckpointer2_1_FunCamera(config=self.config)

  def get_data_shardings(self, mesh):
    # The 2.1 concat records have no `latent_condition` (y comes from
    # latents[:, :, 0], the VAE being causal) but do carry `clip_feature`.
    data_sharding = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))
    return {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "clip_feature": data_sharding,
        "camera_extrinsic": data_sharding,
        "camera_intrinsic": data_sharding,
    }

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Fun camera training only supports dataset_type=tfrecord with cache_latents_text_encoder_outputs=True"
      )

    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        "camera_extrinsic": tf.io.FixedLenFeature([], tf.string),
        "camera_intrinsic": tf.io.FixedLenFeature([], tf.string),
        "clip_feature": tf.io.FixedLenFeature([], tf.string),
    }
    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)
    if is_training and getattr(config, "exclude_sample_ids_path", ""):
      feature_description["sample_id"] = tf.io.FixedLenFeature([], tf.string, default_value="")

    def prepare_sample(features):
      out = {
          "latents": tf.io.parse_tensor(features["latents"], out_type=tf.float32),
          "encoder_hidden_states": tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32),
          "camera_extrinsic": tf.io.parse_tensor(features["camera_extrinsic"], out_type=tf.float32),
          "camera_intrinsic": tf.io.parse_tensor(features["camera_intrinsic"], out_type=tf.float32),
          "clip_feature": tf.io.parse_tensor(features["clip_feature"], out_type=tf.float32),
      }
      if not is_training:
        out["timesteps"] = features["timesteps"]
      return out

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample,
        is_training=is_training,
    )
    # Reshard OUTSIDE jit to ('data','fsdp') — same bs<1 corruption guard as the
    # 2.2 trainer (see its load_dataset comment; root-caused on v2v 2026-07-06).
    batch_sharding = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))

    def _reshard_batches(inner):
      for batch in inner:
        yield {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

    return _reshard_batches(data_iterator)

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

  def _generate_eval_videos(self, pipeline, mesh, example_batch, step):
    """Camera-control eval generation from a sampled dataset record (2.1 Fun)."""
    config = self.config
    gbs = int(config.global_batch_size_to_train_on)
    k = max(1, min(int(getattr(config, "eval_num_generate_samples", 2)), gbs))
    dtype = getattr(config, "activations_dtype", jnp.bfloat16)
    eval_steps = int(getattr(config, "eval_num_inference_steps", 0)) or int(config.num_inference_steps)
    eval_gs = float(getattr(config, "eval_guidance_scale", 1.0))

    latents = example_batch["latents"][:gbs].astype(dtype)
    encoder_hidden_states = example_batch["encoder_hidden_states"][:gbs].astype(dtype)
    clip_feature = example_batch["clip_feature"][:gbs].astype(dtype)
    camera_extrinsic = example_batch["camera_extrinsic"][:gbs]
    camera_intrinsic = example_batch["camera_intrinsic"][:gbs]

    y_latents = pipeline.prepare_fun_camera_y_latents(latents, dtype)
    control_camera_latents = build_control_camera_latents(
        camera_extrinsic,
        camera_intrinsic,
        height=config.height,
        width=config.width,
        moment_scale=float(getattr(config, "camera_moment_scale", 1.0)),
        dtype=dtype,
    )
    negative_prompt_embeds = jnp.zeros_like(encoder_hidden_states)

    max_logging.log(
        f"[eval-gen] step {step}: generating {k} camera-control sample(s), {eval_steps} steps, gs={eval_gs}"
    )
    t0 = time.perf_counter()
    videos, trace = pipeline(
        prompt_embeds=encoder_hidden_states,
        negative_prompt_embeds=negative_prompt_embeds,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        num_inference_steps=eval_steps,
        guidance_scale=eval_gs,
        y_latents=y_latents,
        image_embeds=clip_feature,
        control_camera_latents_input=control_camera_latents,
        use_kv_cache=config.use_kv_cache,
    )
    gen_s = time.perf_counter() - t0
    ext_host = jax.experimental.multihost_utils.process_allgather(camera_extrinsic[:k], tiled=True)
    int_host = jax.experimental.multihost_utils.process_allgather(camera_intrinsic[:k], tiled=True)

    if jax.process_index() == 0:
      _save_eval_samples(config, step, k, videos, ext_host, int_host, eval_steps, eval_gs, gen_s)
    max_logging.log(f"[eval-gen] step {step}: done in {gen_s:.1f}s ({trace})")


def train_step(state, data, rng, scheduler_state, scheduler, config, patch_hw):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config, patch_hw)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config, patch_hw):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)                  # [B, 16, F_lat, h, w]
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    clip_feature = data["clip_feature"].astype(config.weights_dtype)        # [B, 257, 1280]

    bsz = latents.shape[0]

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

    # Official in_dim=32 y: latent frame 0 = VAE(first frame), all later frames
    # ZERO (DiffSynth WanVideoUnit_FunCameraControl primary branch — NOT the
    # VAE(masked video) tail, and NO 4ch mask). The Wan2.1 VAE is temporally
    # causal, so the CLEAN latents' frame 0 already IS vae.encode(first frame)
    # (rel 2.2e-4, check_vae_causality.py) — no latent_condition field needed.
    y = jnp.zeros_like(latents)
    y = y.at[:, :, 0:1].set(latents[:, :, 0:1])
    hidden_states = jnp.concatenate([noisy_latents, y], axis=1)

    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=hidden_states,
          timestep=timesteps,  # scalar [B] timestep (official Fun training + our inference)
          encoder_hidden_states=encoder_hidden_states,
          encoder_hidden_states_image=clip_feature,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
          control_camera_latents_input=control_camera_latents,
      )

    with jax.named_scope("loss"):
      loss = (training_target - model_pred) ** 2  # all frames (diffsynth convention)
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

  # Adapter-only gradient norm (zero when the camera path is frozen or
  # disconnected; growth precedes instability).
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
