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

"""Inference for WAN 2.2 Dense **VACE** (control-video -> video).

Mirrors generate_wan.py, but:
  * loads the VACE transformer via WanVaceCheckpointer2_2_Dense,
  * builds the 160-ch VACE conditioning from a control video
    (inactive[48]=0 | reactive[48]=cond_latents | mask[64]=1), matching the
    Wan2_2DenseVaceTrainer data contract, and
  * runs a VACE denoising loop that calls WanVACEModel(..., control_hidden_states=...)
    exactly like the trainer (the shared transformer_forward_pass does NOT thread
    control_hidden_states, so we cannot reuse run_inference_2_2_dense here).

Example:
  python -u src/maxdiffusion/generate_wan_vace.py \
    src/maxdiffusion/configs/base_wan_ti2v_5b.yml \
    vace_layers=0,5,10,15,20,25 vace_in_channels=160 \
    height=480 width=832 num_frames=81 num_inference_steps=50 guidance_scale=5.0 \
    flow_shift=5.0 \
    checkpoint_dir=gs://data_us_east5_a/maxdiffusion/wan/ti2v5b-vace/vace-100k-bs64-lr-5e-5-zeroinit/checkpoints \
    checkpoint_step=50000 \
    vace_control_video_path=gs://data_us_east5_a/hmworld_raw/.../cond.mp4 \
    prompt="..." \
    output_dir=gs://data_us_east5_a/maxdiffusion/wan/ti2v5b-vace/infer
"""

from typing import Sequence
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from absl import app

from maxdiffusion import pyconfig, max_logging
from maxdiffusion.checkpointing.wan_vace_checkpointer_2_2_dense import WanVaceCheckpointer2_2_Dense
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
# Reuse helpers from the standard generate script.
from maxdiffusion.generate_wan import (
    upload_video_to_gcs,
    _load_video_frames,
    get_git_commit_hash,
)

# NOTE: do NOT enable Shardy here. Training uses the default GSPMD partitioner,
# which allows *uneven* sequence sharding (the patchified video length, e.g. 8190,
# is not divisible by context/fsdp). Shardy is strict and rejects that split.
# jax.config.update("jax_use_shardy_partitioner", True)


def _stats(name, arr):
  """Log finite fraction + max abs of a (possibly sharded) array — NaN debugging."""
  a = np.asarray(jax.device_get(arr))
  finite = float(np.isfinite(a).mean())
  mx = float(np.nanmax(np.abs(a))) if finite > 0 else float("nan")
  max_logging.log(f"[debug] {name}: finite={finite:.4f} max_abs={mx:.4f} shape={tuple(a.shape)}")
  return finite, mx


def make_vace_conditioning(cond_latents: jnp.ndarray) -> jnp.ndarray:
  """[B, C, F, H, W] cond_latents -> [B, 2C+64, F, H, W] VACE conditioning.

  Matches wan_vace_trainer.make_vace_conditioning: inactive(=zeros) + reactive
  (=cond_latents) + all-active mask(64). For WAN2.2 (C=48) -> 160 channels.
  """
  b, c, f, h, w = cond_latents.shape
  inactive = jnp.zeros_like(cond_latents)
  mask = jnp.ones((b, 64, f, h, w), dtype=cond_latents.dtype)
  return jnp.concatenate([inactive, cond_latents, mask], axis=1)


def run_vace_inference(config, pipeline, prompt, negative_prompt, conditioning_latents):
  """Flow-match denoising with VACE control conditioning. Returns decoded video."""
  do_cfg = config.guidance_scale > 1.0

  # Reuse the dense pipeline's input prep (noise latents, text embeds, scheduler).
  latents, prompt_embeds, negative_prompt_embeds, scheduler_state, _ = pipeline._prepare_model_inputs(
      prompt,
      negative_prompt,
      config.height,
      config.width,
      config.num_frames,
      config.num_inference_steps,
      1,            # num_videos_per_prompt
      getattr(config, "max_sequence_length", 512),
      None,         # latents
      None,         # prompt_embeds
      None,         # negative_prompt_embeds
      False,        # vae_only
  )
  bsz = latents.shape[0]
  # (2) input check: are the noise latents / text embeds from _prepare_model_inputs finite?
  _stats("init_latents", latents)
  _stats("prompt_embeds", prompt_embeds)
  _stats("negative_prompt_embeds", negative_prompt_embeds)
  graphdef, state, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

  # JIT the transformer forward: under jit, the model's with_sharding_constraint is a
  # COMPILE-TIME hint, so GSPMD pads/handles the uneven sequence sharding (e.g. 8190
  # over context=8) exactly like training. Run eagerly, that constraint would demand
  # exact divisibility and crash.
  @jax.jit
  def predict(state, lat, t_b, text_emb, cond):
    model = nnx.merge(graphdef, state, rest_of_state)
    out = model(
        hidden_states=lat,
        timestep=t_b,
        encoder_hidden_states=text_emb,
        control_hidden_states=cond,
        return_dict=False,
        deterministic=True,
    )
    return out[0] if isinstance(out, tuple) else out

  with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    cond = jnp.asarray(conditioning_latents, dtype=latents.dtype)
    # Keep the scheduler's native timestep dtype. Some WAN/flow schedulers use
    # shifted fractional timesteps; forcing int32 here makes the model see a
    # different time embedding than the scheduler state represents.
    timesteps = jnp.asarray(scheduler_state.timesteps)
    timesteps_np = np.asarray(jax.device_get(timesteps[: min(5, timesteps.shape[0])]))
    max_logging.log(f"[debug] scheduler_timesteps dtype={timesteps.dtype} first={timesteps_np.tolist()}")

    for step in range(config.num_inference_steps):
      t = timesteps[step]
      t_b = jnp.broadcast_to(t, (bsz,))
      noise_cond = predict(state, latents, t_b, prompt_embeds, cond)
      if step < 5:
        _stats(f"step{step}_noise_cond", noise_cond)
      if do_cfg:
        noise_uncond = predict(state, latents, t_b, negative_prompt_embeds, cond)
        if step < 5:
          _stats(f"step{step}_noise_uncond", noise_uncond)
        noise_pred = noise_uncond + config.guidance_scale * (noise_cond - noise_uncond)
      else:
        noise_pred = noise_cond
      latents, scheduler_state = pipeline.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
      # (3) per-step check (first few steps): pinpoint whether NaN is born in the
      # forward (noise_pred) or in the scheduler.step (latents), and at which step.
      if step < 5:
        _stats(f"step{step}_noise_pred", noise_pred)
        _stats(f"step{step}_latents", latents)

    latents = pipeline._denormalize_latents(latents)
    latents.block_until_ready()

  _lat = np.asarray(jax.device_get(latents))
  max_logging.log(
      f"[debug] denoised latents: min={_lat.min():.4f} max={_lat.max():.4f} "
      f"mean={_lat.mean():.4f} std={_lat.std():.4f} finite_frac={np.isfinite(_lat).mean():.4f}"
  )

  video = pipeline._decode_latents_to_video(latents)
  if hasattr(video, "block_until_ready"):
    video.block_until_ready()
  _v = np.asarray(video)
  max_logging.log(
      f"[debug] decoded video: shape={_v.shape} dtype={_v.dtype} "
      f"min={_v.min():.4f} max={_v.max():.4f} mean={_v.mean():.4f}"
  )
  return video


def run(config, commit_hash=None):
  # --- load VACE pipeline from checkpoint ---
  load_start = time.perf_counter()
  loader = WanVaceCheckpointer2_2_Dense(config=config)
  ckpt_step = getattr(config, "checkpoint_step", -1)
  ckpt_step = int(ckpt_step) if ckpt_step is not None and int(ckpt_step) >= 0 else None
  if ckpt_step is not None:
    max_logging.log(f"Loading VACE checkpoint step: {ckpt_step}")
  pipeline, _, _ = loader.load_checkpoint(step=ckpt_step)
  max_logging.log(f"load_time: {time.perf_counter() - load_start:.1f}s")

  # --- build the 160-ch VACE conditioning from the control video ---
  control_path = getattr(config, "vace_control_video_path", "") or ""
  if not control_path:
    raise ValueError("Set vace_control_video_path=<cond.mp4> (the control video).")
  max_logging.log(f"Loading VACE control video: {control_path}")
  pixel_video = _load_video_frames(control_path, config.height, config.width, num_frames=config.num_frames)
  cond_jnp = jnp.asarray(pixel_video)  # [1, 3, F, H, W]
  activations_dtype = getattr(config, "activations_dtype", jnp.bfloat16)
  cond_latents = pipeline._encode_cond_video(cond_jnp, dtype=activations_dtype)  # [1, 48, f, h, w]
  conditioning_latents = make_vace_conditioning(cond_latents)
  max_logging.log(f"cond_latents {cond_latents.shape} -> conditioning {conditioning_latents.shape}")
  # (1) cond-video-encoding check: is the control conditioning finite?
  _stats("cond_latents", cond_latents)
  _stats("conditioning_latents", conditioning_latents)

  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on
  if conditioning_latents.shape[0] == 1 and len(prompt) > 1:
    conditioning_latents = jnp.concatenate([conditioning_latents] * len(prompt), axis=0)

  max_logging.log(
      f"VACE infer: steps={config.num_inference_steps} {config.width}x{config.height} "
      f"frames={config.num_frames} cfg={config.guidance_scale}"
  )

  # --- denoise (compile run, then timed run) ---
  s0 = time.perf_counter()
  video = run_vace_inference(config, pipeline, prompt, negative_prompt, conditioning_latents)
  max_logging.log(f"compile+generate time: {time.perf_counter() - s0:.1f}s")

  saved = []
  for i in range(len(video)):
    video_path = f"vace_output_{config.seed}_{i}.mp4"
    export_to_video(video[i], video_path, fps=config.fps)
    saved.append(video_path)
    # checkpoint_dir is forced to output_dir/run_name by pyconfig, so use a
    # dedicated key for video output (falls back to output_dir/run_name).
    video_out_dir = getattr(config, "vace_output_video_dir", "") or (
        os.path.join(config.output_dir, config.run_name) if config.run_name else config.output_dir
    )
    if video_out_dir.startswith("gs://"):
      upload_video_to_gcs(video_out_dir, video_path)
  max_logging.log(f"saved: {saved}")
  return saved


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
