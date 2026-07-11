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
#   timesteps    = per-token: first-latent-frame tokens get t=0, clean frame-0
#                  latent (official VideoX-Fun convention; knob
#                  wan_fun_official_first_frame_training, False = CS scalar-t).
#   loss         = flow-match MSE over ALL frames, UNIFORM weight (official
#                  weighting_scheme="none"; frame-0 term is auxiliary — official
#                  inference clamps over frame-0 predictions).
#
# Everything else (optimizer, schedule, logging, checkpoint cadence) is the
# verified WAN 2.1/2.2-dense training loop, unchanged. No dtype changes.

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import tensorflow as tf
from flax import nnx
from jax.experimental import multihost_utils

from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.wan.camera_plucker import build_control_camera_latents
from maxdiffusion.trainers.wan_trainer import WanTrainer
from maxdiffusion import max_logging, max_utils
from maxdiffusion.utils import export_to_video

from jax.sharding import PartitionSpec as P


class Wan2_2FunCameraTrainer(WanTrainer):
  """Fine-tunes PAI Wan2.2-Fun-5B-Control-Camera with camera conditioning."""

  def _get_checkpointer(self):
    return WanCheckpointer2_2_FunCamera(config=self.config)

  def get_data_shardings(self, mesh):
    # Shard the input batch over ('data','fsdp') ONLY (the activation batch axes).
    # With per_device_batch_size<1 the loader delivers batch=num_devices and the
    # train step slices to global_batch_size_to_train_on IN-JIT; slicing a batch
    # dim sharded across the context axis silently corrupts the samples that must
    # migrate between context shards (root-caused + bitwise-validated on the v2v
    # trainer 2026-07-06; same loader + same slice pattern here). bs>=1 unaffected
    # (the slice is a no-op there).
    data_sharding = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))
    return {
        "latents": data_sharding,
        "latent_condition": data_sharding,
        "encoder_hidden_states": data_sharding,
        "camera_extrinsic": data_sharding,
        "camera_intrinsic": data_sharding,
    }

  def get_eval_data_shardings(self, mesh):
    shardings = self.get_data_shardings(mesh)
    shardings["timesteps"] = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))
    return shardings

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
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
    # Held-out training: parse sample_id so the pipeline can drop excluded (test) records.
    # Only when excluding + training, to avoid touching the normal (no-exclusion) path.
    if is_training and getattr(config, "exclude_sample_ids_path", ""):
      feature_description["sample_id"] = tf.io.FixedLenFeature([], tf.string)

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
    # Reshard each loaded batch to the activation batch sharding ('data','fsdp')
    # OUTSIDE jit. The multihost loader hardcodes batch -> ALL mesh axes
    # (multihost_dataloading._build_global_shape_and_sharding); with bs<1 the
    # in-step slice of that layout silently corrupts samples crossing the context
    # axis (v2v repro 2026-07-06: half the sliced batch = 100% NaN). device_put
    # here is the guaranteed-correct reshard; must match get_data_shardings.
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

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    # We do NOT compute eval-loss: the encoded dataset has no per-record
    # `timesteps` feature (only the is_training=False parser expects it). Eval =
    # camera-conditioned video generation, done in `_generate_eval_videos`.
    # Returning None makes the base loop skip the eval-loss path entirely.
    return None

  def _generate_eval_videos(self, pipeline, mesh, example_batch, step):
    """Camera-control eval generation from a sampled dataset record.

    Reuses the current (shuffled) training batch — a random in-distribution
    sample that already carries every conditioning tensor — and drives the
    certified Fun-camera inference pipeline with the SAME conditioning the
    training step builds (y = [mask | latent_condition], Plucker control,
    precomputed text embeds). Saves the generated video + the sample's camera
    trajectory/metadata under {output_dir}/{run_name}/eval/step_{step}/.
    """
    config = self.config
    gbs = int(config.global_batch_size_to_train_on)
    k = max(1, min(int(getattr(config, "eval_num_generate_samples", 2)), gbs))
    dtype = getattr(config, "activations_dtype", jnp.bfloat16)
    eval_steps = int(getattr(config, "eval_num_inference_steps", 0)) or int(config.num_inference_steps)
    eval_gs = float(getattr(config, "eval_guidance_scale", 1.0))

    # Slice the batch to the training global batch so input sharding matches the
    # mesh exactly (size-1 batches can't shard over the data axis).
    latents = example_batch["latents"][:gbs].astype(dtype)
    latent_condition = example_batch["latent_condition"][:gbs].astype(dtype)
    encoder_hidden_states = example_batch["encoder_hidden_states"][:gbs].astype(dtype)
    camera_extrinsic = example_batch["camera_extrinsic"][:gbs]
    camera_intrinsic = example_batch["camera_intrinsic"][:gbs]

    # y-conditioning + Plucker camera latents — identical to the training step.
    mask = _build_first_frame_mask(latent_condition)
    y_latents = jnp.concatenate([mask, latent_condition], axis=1).astype(dtype)
    control_camera_latents = build_control_camera_latents(
        camera_extrinsic,
        camera_intrinsic,
        height=config.height,
        width=config.width,
        moment_scale=float(getattr(config, "camera_moment_scale", 1.0)),
        dtype=dtype,
    )
    # Precomputed text embeds bypass the text encoder; zero negative is unused at
    # guidance_scale<=1 (and is a no-text null otherwise).
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
        control_camera_latents_input=control_camera_latents,
        use_kv_cache=config.use_kv_cache,
    )
    gen_s = time.perf_counter() - t0
    # `videos` is already process_allgather'd to host in _decode_latents_to_video.
    # Camera matrices are tiny — allgather the first k for metadata on host.
    ext_host = jax.experimental.multihost_utils.process_allgather(camera_extrinsic[:k], tiled=True)
    int_host = jax.experimental.multihost_utils.process_allgather(camera_intrinsic[:k], tiled=True)

    if jax.process_index() == 0:
      _save_eval_samples(config, step, k, videos, ext_host, int_host, eval_steps, eval_gs, gen_s)
    max_logging.log(f"[eval-gen] step {step}: done in {gen_s:.1f}s ({trace})")


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
  timesteps = jnp.atleast_1d(timesteps)
  b = timesteps.shape[0]
  seq = f_lat * tokens_per_frame
  t_tok = jnp.broadcast_to(timesteps[:, None], (b, seq)).astype(jnp.float32)
  return t_tok.at[:, :tokens_per_frame].set(0.0)


def _videox_fun_mse_loss(model_pred: jax.Array, training_target: jax.Array) -> jax.Array:
  """VideoX-Fun's FP32 MSE with elementwise errors above 50 ignored."""
  error = model_pred.astype(jnp.float32) - training_target.astype(jnp.float32)
  squared_error = error**2
  mask = (jnp.abs(error) <= 50.0).astype(jnp.float32)
  return jnp.mean(squared_error * mask)

def _upload_file_to_gcs(gcs_dir: str, local_path: str):
  """Upload one local file to gcs_dir/<basename> (gcs_dir = gs://bucket/prefix)."""
  from google.cloud import storage

  path_without_scheme = gcs_dir.removeprefix("gs://")
  bucket_name, _, prefix = path_without_scheme.partition("/")
  blob_name = os.path.join(prefix, os.path.basename(local_path))
  storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


def _save_eval_samples(config, step, k, videos, ext_host, int_host, eval_steps, eval_gs, gen_s):
  """Process-0 only: write the first k generated videos + camera metadata, then
  upload to {output_dir}/{run_name}/eval/step_{step}/."""
  videos = np.asarray(videos)
  out_root = os.path.join(config.output_dir, config.run_name, "eval", f"step_{step}")
  local_dir = os.path.join("/tmp", "eval_gen", str(config.run_name), f"step_{step}")
  os.makedirs(local_dir, exist_ok=True)
  is_gcs = str(config.output_dir).startswith("gs://")

  for i in range(k):
    base = os.path.join(local_dir, f"sample_{i}")
    mp4, ext_npy, int_npy, meta_json = base + ".mp4", base + "_extrinsic.npy", base + "_intrinsic.npy", base + ".json"
    export_to_video(videos[i], mp4, fps=config.fps)

    ext = np.asarray(ext_host[i])   # [F, 3, 4] world-to-camera extrinsics
    intr = np.asarray(int_host[i])  # [F, 3, 3] intrinsics
    np.save(ext_npy, ext)
    np.save(int_npy, intr)
    meta = {
        "step": int(step),
        "sample_index": int(i),
        "run_name": str(config.run_name),
        "height": int(config.height),
        "width": int(config.width),
        "num_frames": int(config.num_frames),
        "num_inference_steps": int(eval_steps),
        "guidance_scale": float(eval_gs),
        "gen_seconds": round(float(gen_s), 2),
        "camera_num_frames": int(ext.shape[0]),
        "camera_translation_start": ext[0, :, 3].tolist(),
        "camera_translation_end": ext[-1, :, 3].tolist(),
        "camera_translation_delta": (ext[-1, :, 3] - ext[0, :, 3]).tolist(),
        "intrinsic_first": intr[0].tolist(),
    }
    with open(meta_json, "w") as f:
      json.dump(meta, f, indent=2)

    if is_gcs:
      for p in (mp4, ext_npy, int_npy, meta_json):
        _upload_file_to_gcs(out_root, p)

  dest = out_root if is_gcs else local_dir
  max_logging.log(f"[eval-gen] step {step}: wrote {k} sample(s) -> {dest}")


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
    # OFFICIAL VideoX-Fun first-frame training convention for the 16x-VAE 5B family
    # (scripts/wan2.2_fun/train_control_lora.py ~L1995): the frame-0 latent is kept
    # CLEAN (never noised) and the timestep is PER-TOKEN with frame-0 tokens at t=0;
    # the loss stays over ALL frames (sigma-weighted, unmasked — the frame-0 term is
    # an auxiliary: official inference discards frame-0 predictions via the latent
    # clamp, wan_fun_first_frame_clamp). The 7416e130-era comment claimed diffsynth/
    # CS scalar-t all-noised matched base training; VideoX-Fun's code shows CS is
    # the deviation. wan_fun_official_first_frame_training=False restores CS-style.
    official_ff = bool(getattr(config, "wan_fun_official_first_frame_training", True))
    if official_ff:
      # Frame-0 source = the MASKED-video latent (latent_condition), exactly as
      # official (control_latents[:, -C:]); causally equal to latents[:, :, 0:1]
      # but byte-consistent with the y channels and the inference clamp source.
      noisy_latents = noisy_latents.at[:, :, 0:1].set(latent_condition[:, :, 0:1].astype(noisy_latents.dtype))
      f_lat = latents.shape[2]
      tokens_per_frame = (latents.shape[3] // 2) * (latents.shape[4] // 2)
      timestep_input = _per_token_timesteps(timesteps, f_lat, tokens_per_frame)
    else:
      timestep_input = timesteps  # scalar [B] (diffsynth/CS convention)

    # y-conditioning: [mask(4) | latent_condition(48)], concat after the noisy
    # latents -> the 100-channel Fun camera-control input.
    mask = _build_first_frame_mask(latents)
    hidden_states = jnp.concatenate([noisy_latents, mask, latent_condition], axis=1)

    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=hidden_states,
          timestep=timestep_input,
          encoder_hidden_states=encoder_hidden_states,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
          control_camera_latents_input=control_camera_latents,
      )

    with jax.named_scope("loss"):
      if official_ff:
        # Official VideoX-Fun recipe: uniform, guarded FP32 MSE over all frames.
        loss = _videox_fun_mse_loss(model_pred, training_target)
      else:
        # Preserve the existing diffsynth/CS compatibility behavior.
        loss = (training_target - model_pred) ** 2
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
