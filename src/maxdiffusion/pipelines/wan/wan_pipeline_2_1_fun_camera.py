# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

# Pipeline for PAI **Wan2.1-Fun-V1.1-1.3B-Control-Camera** (model_type I2V-CC).
#
# Official contract (DiffSynth wan_video.py WanVideoUnit_FunCameraControl +
# model_fn_wan_video, config hash ac6a5aa7):
#   model input  = concat([noisy_latents(16), y(16)]) = 32ch, where y is ZERO
#                  everywhere except latent frame 0 = VAE(first frame). NO mask
#                  channels (in_dim=32 primary branch, unlike 2.2 Fun's 100ch).
#   clip         = has_image_input=True: CLIP ViT-H [B,257,1280] through img_emb,
#                  prepended to the text context in every cross-attention.
#   camera       = packed Plucker [B,24,F_lat,H,W] through SimpleAdapter
#                  (PixelUnshuffle 8 + conv stride 2 = /16 = token grid for the
#                  8x-spatial 2.1 VAE), added to the patch embedding.
#   timestep     = scalar per step (no per-token TI2V hold; conditioning comes
#                  solely from y + clip).
#
# Everything else is the verified WAN 2.1 single-transformer loop.

from functools import partial
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import time
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from .wan_pipeline_2_1 import WanPipeline2_1
from .wan_pipeline import transformer_forward_pass, transformer_forward_pass_full_cfg
from ...pyconfig import HyperParameters
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler


class WanPipeline2_1_FunCamera(WanPipeline2_1):
  """Official PAI Wan2.1-Fun-V1.1-1.3B-Control-Camera pipeline."""

  def _get_num_channel_latents(self) -> int:
    # Noise latents are out_channels (16); in_channels (32) includes the y half.
    return getattr(self.transformer.config, "out_channels", None) or self.transformer.config.in_channels

  def prepare_fun_camera_y_latents(self, latents: jax.Array, dtype) -> jax.Array:
    """DiffSynth in_dim=32 y: latent frame 0 = VAE(first frame), rest zero.

    `latents` are the CLEAN VAE latents of the video [B, 16, F_lat, h, w]. The
    Wan2.1 VAE is temporally causal, so latents[:, :, 0] IS vae.encode(first
    frame) — the exact official y — verified on real HM-World frames at
    480x832: rel 2.2e-4 vs a separate first-frame encode, while latent frame 1
    differs by 98% (check_vae_causality.py). So no separate `latent_condition`
    encode is needed.
    """
    y = jnp.zeros_like(latents)
    y = y.at[:, :, 0:1].set(latents[:, :, 0:1])
    return y.astype(dtype)

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 153,
      num_inference_steps: int = 40,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      y_latents: Optional[jax.Array] = None,
      image_embeds: Optional[jax.Array] = None,
      control_camera_latents_input: Optional[jax.Array] = None,
      vae_only: bool = False,
      use_kv_cache: bool = False,
  ):
    if y_latents is None:
      raise ValueError("Wan2.1-Fun camera inference requires y_latents (first-frame latent conditioning).")
    if control_camera_latents_input is None:
      raise ValueError("Wan2.1-Fun camera inference requires control_camera_latents_input.")
    if image_embeds is None:
      raise ValueError(
          "Wan2.1-Fun camera inference requires image_embeds (CLIP [B,257,1280] of the first frame; "
          "precomputed in the dataset records or via the torch CLIP encoder)."
      )

    trace = {}
    t_cond_start = time.perf_counter()
    latents, prompt_embeds, negative_prompt_embeds, scheduler_state, num_frames = self._prepare_model_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        num_videos_per_prompt,
        max_sequence_length,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        vae_only,
    )
    latents.block_until_ready()
    prompt_embeds.block_until_ready()
    trace["conditioning"] = time.perf_counter() - t_cond_start

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)
    p_run_inference = partial(
        run_inference_2_1_fun_camera,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
        config=self.config,
        use_kv_cache=use_kv_cache,
    )

    t_denoise_start = time.perf_counter()
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      latents = p_run_inference(
          graphdef=graphdef,
          sharded_state=state,
          rest_of_state=rest_of_state,
          latents=latents,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
          y_latents=y_latents,
          image_embeds=image_embeds,
          control_camera_latents_input=control_camera_latents_input,
      )
      latents = self._denormalize_latents(latents)
      latents.block_until_ready()
    trace["denoise_total"] = time.perf_counter() - t_denoise_start

    t_decode_start = time.perf_counter()
    video = self._decode_latents_to_video(latents)
    if hasattr(video, "block_until_ready"):
      video.block_until_ready()
    trace["vae_decode"] = time.perf_counter() - t_decode_start
    return video, trace


def run_inference_2_1_fun_camera(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    y_latents: jnp.array,
    image_embeds: jnp.array,
    control_camera_latents_input: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    config=None,
    use_kv_cache: bool = False,
):
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]
  prompt_cond_embeds = prompt_embeds
  prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0) if do_cfg else None

  transformer_obj = nnx.merge(graphdef, sharded_state, rest_of_state)

  # RoPE spans the full 32-channel concat input grid (channel count only sets
  # the last dummy dim; the grid is what matters).
  rope_channels = latents.shape[1] + y_latents.shape[1]
  dummy_hidden_states = jnp.zeros((latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4], rope_channels))
  rotary_emb = transformer_obj.rope(dummy_hidden_states)

  kv_cache = None
  encoder_attention_mask = None
  if use_kv_cache:
    kv_cache, encoder_attention_mask = transformer_obj.compute_kv_cache(
        prompt_embeds_combined if do_cfg else prompt_cond_embeds,
        jnp.concatenate([image_embeds] * 2, axis=0) if do_cfg else image_embeds,
    )

  def _for_batch(x, ref):
    batch_mult = ref.shape[0] // x.shape[0]
    return x if batch_mult == 1 else jnp.concatenate([x] * batch_mult, axis=0)

  def _append_y(x):
    y_rep = _for_batch(y_latents, x)
    if x.shape[2:] != y_rep.shape[2:]:
      raise ValueError(f"y_latents shape {y_rep.shape} is incompatible with latents {x.shape}")
    return jnp.concatenate([x, y_rep.astype(x.dtype)], axis=1)

  # Official DiffSynth Fun-Camera inference: scalar timestep, all frames
  # denoised; first-frame + camera conditioning enter via y, clip and Plucker.
  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    if do_cfg:
      latents_doubled = jnp.concatenate([latents] * 2)
      transformer_input = _append_y(latents_doubled)
      timestep = jnp.broadcast_to(t, (bsz * 2,))
      noise_pred, _, _ = transformer_forward_pass_full_cfg(
          graphdef,
          sharded_state,
          rest_of_state,
          transformer_input,
          timestep,
          prompt_embeds_combined,
          guidance_scale=guidance_scale,
          kv_cache=kv_cache,
          rotary_emb=rotary_emb,
          encoder_attention_mask=encoder_attention_mask,
          encoder_hidden_states_image=_for_batch(image_embeds, transformer_input),
          control_camera_latents_input=_for_batch(control_camera_latents_input, transformer_input),
      )
    else:
      transformer_input = _append_y(latents)
      timestep = jnp.broadcast_to(t, (bsz,))
      noise_pred, _ = transformer_forward_pass(
          graphdef,
          sharded_state,
          rest_of_state,
          transformer_input,
          timestep,
          prompt_cond_embeds,
          do_classifier_free_guidance=False,
          guidance_scale=guidance_scale,
          kv_cache=kv_cache,
          rotary_emb=rotary_emb,
          encoder_attention_mask=encoder_attention_mask,
          encoder_hidden_states_image=_for_batch(image_embeds, transformer_input),
          control_camera_latents_input=_for_batch(control_camera_latents_input, transformer_input),
      )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

  return latents
