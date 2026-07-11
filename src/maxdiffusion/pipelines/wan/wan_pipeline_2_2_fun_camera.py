# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

from functools import partial
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import time
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from .wan_pipeline_2_2_dense import WanPipeline2_2_Dense
from .wan_pipeline import WanPipeline, transformer_forward_pass, transformer_forward_pass_full_cfg
from ...models.wan.transformers.transformer_wan import WanModel
from ...pyconfig import HyperParameters
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler


class WanPipeline2_2_FunCamera(WanPipeline2_2_Dense):
  """Official PAI/Captain-Safari Wan2.2-Fun-5B-Control-Camera pipeline.

  This is intentionally separate from the frame-concat TV2V dense path. The
  official Fun camera model uses transformer input channels:
    noisy latent 48ch + y conditioning 52ch = 100ch
  and camera Plucker latents through a SimpleAdapter after patch embedding.
  """

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only and load_transformer:
      transformer = WanPipeline.load_transformer(
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

  def _get_num_channel_latents(self) -> int:
    return getattr(self.transformer.config, "out_channels", self.transformer.config.in_channels)

  def prepare_fun_camera_y_latents(self, image, height: int, width: int, num_frames: int, batch_size: int, dtype):
    tensor = self.video_processor.preprocess(image, height=height, width=width)
    image_tensor = jnp.array(tensor.cpu().numpy())
    if image_tensor.ndim == 3:
      image_tensor = image_tensor[None, ...]
    if image_tensor.shape[0] == 1 and batch_size > 1:
      image_tensor = jnp.repeat(image_tensor, batch_size, axis=0)

    latent_condition, _ = self.prepare_latents_i2v_base(image_tensor, num_frames, dtype)
    latent_height = latent_condition.shape[2]
    latent_width = latent_condition.shape[3]
    num_latent_frames = latent_condition.shape[1]

    mask = jnp.ones((image_tensor.shape[0], 1, num_frames, latent_height, latent_width), dtype=dtype)
    mask = mask.at[:, :, 1:, :, :].set(0)
    first_frame_mask = jnp.repeat(mask[:, :, 0:1], self.vae_scale_factor_temporal, axis=2)
    mask = jnp.concatenate([first_frame_mask, mask[:, :, 1:]], axis=2)
    mask = mask.reshape(
        image_tensor.shape[0],
        1,
        num_latent_frames,
        self.vae_scale_factor_temporal,
        latent_height,
        latent_width,
    )
    mask = jnp.transpose(mask, (0, 2, 4, 5, 3, 1)).squeeze(-1)
    y_latents = jnp.concatenate([mask, latent_condition.astype(dtype)], axis=-1)
    return jnp.transpose(y_latents, (0, 4, 1, 2, 3)).astype(dtype)

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 704,
      width: int = 1248,
      num_frames: int = 121,
      num_inference_steps: int = 40,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      y_latents: Optional[jax.Array] = None,
      control_camera_latents_input: Optional[jax.Array] = None,
      vae_only: bool = False,
      use_kv_cache: bool = False,
      memory: Optional[jax.Array] = None,
      memory_pose_token: Optional[jax.Array] = None,
      memory_key_pose_token: Optional[jax.Array] = None,
  ):
    if y_latents is None:
      raise ValueError("Wan2.2-Fun camera inference requires y_latents from the first input image.")
    if control_camera_latents_input is None:
      raise ValueError("Wan2.2-Fun camera inference requires control_camera_latents_input.")

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
        run_inference_2_2_fun_camera,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
        config=self.config,
        use_kv_cache=use_kv_cache,
        first_frame_clamp=bool(getattr(self.config, "wan_fun_first_frame_clamp", True)),
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
          control_camera_latents_input=control_camera_latents_input,
          memory=memory,
          memory_pose_token=memory_pose_token,
          memory_key_pose_token=memory_key_pose_token,
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


@jax.jit
def _clamp_frame0(latents, cond_frame0):
  """Official first-frame hold, as ONE cached executable (compiled on first call).

  Dispatching this as a per-step eager `.at[].set(...)` wedged the TPU runtime:
  every eval worker blocked inside TpuLoadedExecutable::Execute mid-denoise
  (py-spy verified, 2026-07-11). Keep it jitted."""
  return latents.at[:, :, 0:1].set(cond_frame0)


def run_inference_2_2_fun_camera(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    y_latents: jnp.array,
    control_camera_latents_input: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    config=None,
    use_kv_cache: bool = False,
    memory: Optional[jnp.array] = None,
    memory_pose_token: Optional[jnp.array] = None,
    memory_key_pose_token: Optional[jnp.array] = None,
    first_frame_clamp: bool = True,
):
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]
  # Official VideoX-Fun (pipeline_wan2_2_fun_control.py) hard-holds the anchor
  # frame for the 16x-VAE 5B family: after init and after EVERY scheduler step,
  # latent frame 0 is reset to the conditioning latent (frames 1+ keep denoising).
  # y_latents layout is [B, 4 mask + 48 latent_condition, F, h, w]; the causal VAE
  # makes latent_condition frame 0 == VAE(first frame).
  cond_frame0 = y_latents[:, 4:, 0:1].astype(latents.dtype)
  if first_frame_clamp:
    latents = _clamp_frame0(latents, cond_frame0)
  # Part 2 of the official mechanism: frame-0 tokens at t=0, others at t
  # (VideoX-Fun pipeline_wan2_2_fun_control.py:833). The timestep only ever takes
  # these TWO values, so it is passed per-FRAME [rows, f_lat]; the transformer's
  # grouped-modulation path (transformer_wan.py) broadcasts within each frame's
  # tokens — the materialized per-token [B, seq, 6, dim] layout wedged v6e HBM in
  # the unrolled inference loop, and a per-step eager scatter here wedged the
  # runtime execute queue (both 2026-07-11). Mask built ONCE; per step the
  # timestep is a single cached multiply.
  f_lat = latents.shape[2]
  rows_used = bsz * 2 if do_cfg else bsz
  frame_t_mask = None
  if first_frame_clamp:
    frame_t_mask = jnp.ones((rows_used, f_lat), jnp.float32).at[:, 0].set(0.0)

  def _timestep_for(t, rows):
    if not first_frame_clamp:
      return jnp.broadcast_to(t, (rows,))
    assert rows == rows_used, (rows, rows_used)
    return t.astype(jnp.float32) * frame_t_mask
  prompt_cond_embeds = prompt_embeds
  prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0) if do_cfg else None

  transformer_obj = nnx.merge(graphdef, sharded_state, rest_of_state)

  # Captain-Safari 3D memory: retrieve + lift ONCE (independent of the noisy latents and
  # timestep), then reuse every denoise step. memory_context is text-independent, so under
  # CFG both the cond and uncond halves get the SAME memory -> just double it.
  memory_context = None
  if memory is not None:
    memory_context = transformer_obj.compute_memory_context(
        memory, memory_pose_token, memory_key_pose_token
    ).astype(latents.dtype)

  def _double_mem(mc, x):
    if mc is None:
      return None
    batch_mult = x.shape[0] // mc.shape[0]
    return mc if batch_mult == 1 else jnp.concatenate([mc] * batch_mult, axis=0)
  rope_channels = latents.shape[1] + y_latents.shape[1]
  dummy_hidden_states = jnp.zeros((latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4], rope_channels))
  rotary_emb = transformer_obj.rope(dummy_hidden_states)

  kv_cache = None
  encoder_attention_mask = None
  if use_kv_cache:
    kv_cache, encoder_attention_mask = transformer_obj.compute_kv_cache(prompt_embeds_combined if do_cfg else prompt_cond_embeds)

  def _append_y(x):
    batch_mult = x.shape[0] // y_latents.shape[0]
    y_rep = y_latents if batch_mult == 1 else jnp.concatenate([y_latents] * batch_mult, axis=0)
    if x.shape[2:] != y_rep.shape[2:]:
      raise ValueError(f"y_latents shape {y_rep.shape} is incompatible with latents {x.shape}")
    return jnp.concatenate([x, y_rep.astype(x.dtype)], axis=1)

  def _camera_for_batch(x):
    batch_mult = x.shape[0] // control_camera_latents_input.shape[0]
    if batch_mult == 1:
      return control_camera_latents_input
    return jnp.concatenate([control_camera_latents_input] * batch_mult, axis=0)

  # Fun-5B-Control-Camera is seperated_timestep=False: scalar timestep per step, all frames
  # denoised. First-frame conditioning = the y-concat (mask + latent_condition) PLUS,
  # when first_frame_clamp is on (official VideoX-Fun behavior for the 16x-VAE 5B),
  # the per-step hard hold of latent frame 0 above. diffsynth/CS omit the hold —
  # set wan_fun_first_frame_clamp=False to reproduce the old CS-parity behavior.
  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    if do_cfg:
      latents_doubled = jnp.concatenate([latents] * 2)
      transformer_input = _append_y(latents_doubled)
      timestep = _timestep_for(t, bsz * 2)
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
          control_camera_latents_input=_camera_for_batch(transformer_input),
          memory_context=_double_mem(memory_context, transformer_input),
      )
    else:
      transformer_input = _append_y(latents)
      timestep = _timestep_for(t, bsz)
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
          control_camera_latents_input=_camera_for_batch(transformer_input),
          memory_context=_double_mem(memory_context, transformer_input),
      )

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    if first_frame_clamp:
      latents = _clamp_frame0(latents, cond_frame0)

  return latents
