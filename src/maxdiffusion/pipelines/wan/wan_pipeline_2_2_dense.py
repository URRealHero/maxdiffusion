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

from .wan_pipeline import WanPipeline, transformer_forward_pass, transformer_forward_pass_full_cfg, transformer_forward_pass_cfg_cache, init_magcache, magcache_step
from ...models.wan.transformers.transformer_wan import WanModel
from ...models.wan.autoencoder_kl_wan_2p2 import AutoencoderKLWan2p2
from ...models.wan.autoencoder_kl_wan import AutoencoderKLWanCache
from ...models.wan.wan_utils import load_wan_vae
from typing import List, Union, Optional
from ...pyconfig import HyperParameters
from functools import partial
import flax
import flax.traverse_util
from flax import nnx
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler
from ...video_processor import VideoProcessor
import numpy as np
import time
from ... import max_utils
from ...max_utils import device_put_replicated


class WanPipeline2_2_Dense(WanPipeline):
  """Pipeline for WAN 2.2 Dense 5B Model with a single transformer."""

  def __init__(self, config: HyperParameters, transformer: Optional[WanModel], **kwargs):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer
    # The WAN 2.2 VAE applies a 2x pixel-shuffle patchify on top of the conv
    # down-blocks, so its true spatial compression is 8 * patch_size = 16. The
    # base pipeline computes 8, which doubles the latent (and output) resolution.
    if getattr(self, "vae", None) is not None:
      vae_spatial_patch = getattr(self.vae, "patch_size", 1) or 1
      self.vae_scale_factor_spatial = (2 ** len(self.vae.temperal_downsample)) * vae_spatial_patch
      self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    common_components = cls._create_common_components(config, vae_only)
    transformer = None
    if not vae_only and load_transformer:
      transformer = super().load_transformer(
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
  def from_checkpoint(
      cls,
      config: HyperParameters,
      restored_checkpoint=None,
      vae_only=False,
      load_transformer=True,
  ):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  @classmethod
  def load_vae(
      cls,
      devices_array: np.array,
      mesh: Mesh,
      rngs: nnx.Rngs,
      config: HyperParameters,
      vae_logical_axis_rules: tuple = None,
  ):
    """Override: the TI2V-5B dense model uses the WAN 2.2 high-compression VAE
    (AutoencoderKLWan2p2, 48 latent channels, 16x spatial), not the 2.1 VAE the
    base pipeline loads. Mirrors WanPipeline.load_vae otherwise."""

    def create_model(rngs: nnx.Rngs, config: HyperParameters):
      if getattr(config, "model_type", "") == "TI2V-CC" and getattr(config, "wan_vae_filename", ""):
        # PAI Fun camera-control repos publish the VAE as Wan2.2_VAE.pth at repo
        # root (no diffusers vae/ subfolder; the root config.json belongs to the
        # transformer). Build the native Wan2.2 VAE from class defaults and load
        # the PAI weights via the filename override below.
        wan_vae = AutoencoderKLWan2p2(
            rngs=rngs,
            mesh=mesh,
            dtype=config.vae_dtype,
            weights_dtype=config.vae_weights_dtype,
        )
      else:
        wan_vae = AutoencoderKLWan2p2.from_config(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            rngs=rngs,
            mesh=mesh,
            dtype=config.vae_dtype,
            weights_dtype=config.vae_weights_dtype,
        )
      return wan_vae

    p_model_factory = partial(create_model, config=config)
    wan_vae = nnx.eval_shape(p_model_factory, rngs=rngs)
    graphdef, state = nnx.split(wan_vae, nnx.Param)

    logical_state_spec = nnx.get_partition_spec(state)
    logical_rules = vae_logical_axis_rules if vae_logical_axis_rules is not None else config.logical_axis_rules
    logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, logical_rules)
    logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
    params = state.to_pure_dict()
    state = dict(nnx.to_flat_state(state))

    params = load_wan_vae(
        config.pretrained_model_name_or_path,
        params,
        "cpu",
        is_wan_2p2=True,
        subfolder=getattr(config, "wan_vae_subfolder", "vae"),
        filename=getattr(config, "wan_vae_filename", "") or "diffusion_pytorch_model.safetensors",
    )
    params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
    for path, val in flax.traverse_util.flatten_dict(params).items():
      sharding = logical_state_sharding[path].value
      if config.replicate_vae:
        sharding = NamedSharding(mesh, P())
      state[path].value = device_put_replicated(val, sharding)
    state = nnx.from_flat_state(state)

    wan_vae = nnx.merge(graphdef, state)
    vae_cache = AutoencoderKLWanCache(wan_vae)
    return wan_vae, vae_cache

  def _get_num_channel_latents(self) -> int:
    return self.transformer.config.in_channels

  def _encode_cond_video(self, video: jax.Array, dtype) -> jax.Array:
    """VAE-encode a pixel-space video to normalized channel-first latents.

    Args:
      video: float32 array in [-1, 1], shape [B, C, F, H, W] (channel-first,
             matching the WAN VAE encoder's expected input format).
      dtype: target dtype for the returned latent.

    Returns:
      Latent array in channel-first format [B, C, F_lat, H_lat, W_lat].
    """
    vae_dtype = getattr(self.vae, "dtype", jnp.float32)
    video = video.astype(vae_dtype)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      # VAE encodes [B, C, F, H, W] → [B, F_lat, H_lat, W_lat, C] (channel-last output)
      encoded = self.vae.encode(video, self.vae_cache)[0].mode()
    latents_mean = jnp.array(self.vae.latents_mean).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents_std = jnp.array(self.vae.latents_std).reshape(1, 1, 1, 1, self.vae.z_dim)
    latents = (encoded - latents_mean) / latents_std  # [B, F_lat, H_lat, W_lat, C]
    latents = jnp.transpose(latents, (0, 4, 1, 2, 3))  # → [B, C, F_lat, H_lat, W_lat]
    return latents.astype(dtype)

  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      negative_prompt: Union[str, List[str]] = None,
      height: int = 480,
      width: int = 832,
      num_frames: int = 81,
      num_inference_steps: int = 50,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: Optional[int] = 1,
      max_sequence_length: int = 512,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      vae_only: bool = False,
      use_cfg_cache: bool = False,
      use_magcache: bool = False,
      magcache_thresh: Optional[float] = None,
      magcache_K: Optional[int] = None,
      retention_ratio: Optional[float] = None,
      use_kv_cache: bool = False,
      cond_latents: Optional[jax.Array] = None,
  ):
    config = getattr(self, "config", None)
    if magcache_thresh is None:
      magcache_thresh = getattr(config, "magcache_thresh", 0.12)
    if magcache_K is None:
      magcache_K = getattr(config, "magcache_K", 2)
    if retention_ratio is None:
      retention_ratio = getattr(config, "retention_ratio", 0.2)

    if use_cfg_cache and guidance_scale <= 1.0:
      raise ValueError(
          f"use_cfg_cache=True requires guidance_scale > 1.0 (got {guidance_scale}). "
          "CFG cache accelerates classifier-free guidance, which is disabled when guidance_scale <= 1.0."
      )
    if cond_latents is not None and (use_cfg_cache or use_magcache):
      raise ValueError(
          "cond_latents (frame-concat conditioning) is not supported with use_cfg_cache or use_magcache. "
          "Set use_cfg_cache=False and use_magcache=False."
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
        run_inference_2_2_dense,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        scheduler_state=scheduler_state,
        use_cfg_cache=use_cfg_cache,
        use_magcache=use_magcache,
        magcache_thresh=magcache_thresh,
        magcache_K=magcache_K,
        retention_ratio=retention_ratio,
        height=height,
        mag_ratios_base=getattr(config, "mag_ratios_base", None),
        config=self.config,
        use_kv_cache=use_kv_cache,
        cond_latents=cond_latents,
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


def run_inference_2_2_dense(
    graphdef,
    sharded_state,
    rest_of_state,
    latents: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    use_cfg_cache: bool = False,
    use_magcache: bool = False,
    magcache_thresh: float = 0.12,
    magcache_K: int = 2,
    retention_ratio: float = 0.2,
    height: int = 480,
    mag_ratios_base: Optional[List[float]] = None,
    config=None,
    use_kv_cache: bool = False,
    cond_latents: Optional[jnp.ndarray] = None,
):
  """Denoising loop for Wan2.2 dense single-transformer models."""
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]

  # Resolution-dependent CFG cache config (FasterCache / MixCache guidance)
  if height >= 720:
    # 720p: conservative — protect last 40%, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = int(num_inference_steps * 0.9)
    cfg_cache_alpha = 0.2
  else:
    # 480p: moderate — protect last 2 steps, interval=5
    cfg_cache_interval = 5
    cfg_cache_start_step = int(num_inference_steps / 3)
    cfg_cache_end_step = num_inference_steps - 2
    cfg_cache_alpha = 0.2

  # Pre-split embeds once, outside the loop.
  prompt_cond_embeds = prompt_embeds
  prompt_embeds_combined = None
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

  # Pre-compute cache schedule and phase-dependent weights.
  # t₀ = midpoint step; before t₀ boost low-freq, after boost high-freq.
  t0_step = num_inference_steps // 2
  first_full_step_seen = False
  step_is_cache = []
  step_w1w2 = []
  for s in range(num_inference_steps):
    is_cache = (
        use_cfg_cache
        and do_cfg
        and first_full_step_seen
        and s >= cfg_cache_start_step
        and s < cfg_cache_end_step
        and (s - cfg_cache_start_step) % cfg_cache_interval != 0
    )
    step_is_cache.append(is_cache)
    if not is_cache:
      first_full_step_seen = True
    # Phase-dependent weights: w = 1 + α·I(condition)
    if s < t0_step:
      step_w1w2.append((1.0 + cfg_cache_alpha, 1.0))  # early: boost low-freq
    else:
      step_w1w2.append((1.0, 1.0 + cfg_cache_alpha))  # late: boost high-freq

  # Cache tensors (on-device JAX arrays, initialised to None).
  cached_noise_cond = None
  cached_noise_uncond = None

  transformer_obj = nnx.merge(graphdef, sharded_state, rest_of_state)

  num_cond = cond_latents.shape[2] if cond_latents is not None else 0

  # Helpers for prepending cond frames and slicing noise predictions.
  def _prepend_cond(x):
    """Prepend cond_latents to x along temporal axis (axis=2)."""
    if num_cond == 0:
      return x
    # x may be [B, C, T, H, W] or [2B, C, T, H, W] for CFG-doubled inputs.
    batch_mult = x.shape[0] // latents.shape[0]
    cond_rep = jnp.concatenate([cond_latents] * batch_mult, axis=0)
    return jnp.concatenate([cond_rep, x], axis=2)

  def _slice_target(pred):
    """Remove cond-frame positions from a noise prediction."""
    if num_cond == 0:
      return pred
    return pred[:, :, num_cond:, :, :]

  # Compute RoPE once; use full temporal dim (cond + target) when conditioning.
  T_full = num_cond + latents.shape[2]
  dummy_hidden_states = jnp.zeros((
      latents.shape[0],
      T_full,
      latents.shape[3],
      latents.shape[4],
      latents.shape[1],
  ))
  rotary_emb = transformer_obj.rope(dummy_hidden_states)

  kv_cache = None
  encoder_attention_mask = None

  if use_kv_cache:
    kv_cache, encoder_attention_mask = transformer_obj.compute_kv_cache(
        prompt_embeds_combined if do_cfg else prompt_cond_embeds
    )

  if use_magcache and do_cfg:
    magcache_init = init_magcache(num_inference_steps, retention_ratio, mag_ratios_base)
    accumulated_state = magcache_init[:6]
    cached_residual = magcache_init[6]
    skip_warmup = magcache_init[7]
    mag_ratios = magcache_init[8]

  first_profiling_step = config.skip_first_n_steps_for_profiler if config else 0
  profiler_steps = config.profiler_steps if config else 0
  last_profiling_step = np.clip(
      first_profiling_step + profiler_steps - 1,
      first_profiling_step,
      num_inference_steps - 1,
  )

  scan_diffusion_loop = getattr(config, "scan_diffusion_loop", False) if config else False

  if scan_diffusion_loop and not use_magcache and not use_cfg_cache:
    if num_cond > 0:
      raise ValueError(
          "scan_diffusion_loop=True is not supported with cond_latents. "
          "Set scan_diffusion_loop=False."
      )
    timesteps = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)

    scheduler_state = scheduler_state.replace(last_sample=jnp.zeros_like(latents), step_index=jnp.array(0, dtype=jnp.int32))

    def scan_body(carry, t):
      current_latents, current_scheduler_state = carry

      if do_cfg:
        latents_doubled = jnp.concatenate([current_latents] * 2)
        timestep = jnp.broadcast_to(t, bsz * 2)
        noise_pred, _, _ = transformer_forward_pass_full_cfg(
            graphdef,
            sharded_state,
            rest_of_state,
            latents_doubled,
            timestep,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
      else:
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred, _ = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            current_latents,
            timestep,
            prompt_cond_embeds,
            do_classifier_free_guidance=False,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

      new_latents, new_scheduler_state = scheduler.step(
          current_scheduler_state, noise_pred, t, current_latents, return_dict=False
      )

      return (new_latents, new_scheduler_state), None

    initial_carry = (latents, scheduler_state)

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, timesteps)

    final_latents, _ = final_carry
    return final_latents

  profiler = None
  for step in range(num_inference_steps):
    if config and max_utils.profiler_enabled(config) and step == first_profiling_step:
      profiler = max_utils.Profiler(config)
      profiler.start()

    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

    if use_magcache and do_cfg:
      timestep = jnp.broadcast_to(t, bsz * 2 if do_cfg else bsz)

      skip_blocks, accumulated_state = magcache_step(
          step,
          mag_ratios,
          accumulated_state,
          magcache_thresh,
          magcache_K,
          skip_warmup,
      )

      latents_for_magcache = _prepend_cond(jnp.concatenate([latents] * 2) if do_cfg else latents)
      noise_pred, latents, residual_x_cur = transformer_forward_pass(
          graphdef,
          sharded_state,
          rest_of_state,
          latents_for_magcache,
          timestep,
          prompt_embeds_combined if do_cfg else prompt_cond_embeds,
          do_classifier_free_guidance=do_cfg,
          guidance_scale=guidance_scale,
          skip_blocks=bool(skip_blocks),
          cached_residual=cached_residual,
          return_residual=True,
          kv_cache=kv_cache,
          rotary_emb=rotary_emb,
          encoder_attention_mask=encoder_attention_mask,
      )
      noise_pred = _slice_target(noise_pred)
      latents = _slice_target(latents)
      residual_x_cur = _slice_target(residual_x_cur)

      if not skip_blocks:
        cached_residual = residual_x_cur

    else:
      is_cache_step = step_is_cache[step]

      if is_cache_step:
        w1, w2 = step_w1w2[step]
        timestep = jnp.broadcast_to(t, bsz)
        kv_cache_cond = jax.tree.map(lambda x: x[:, :bsz], kv_cache) if kv_cache is not None else None
        encoder_attention_mask_cond = encoder_attention_mask[:bsz] if encoder_attention_mask is not None else None
        noise_pred, cached_noise_cond = transformer_forward_pass_cfg_cache(
            graphdef,
            sharded_state,
            rest_of_state,
            _prepend_cond(latents),
            timestep,
            prompt_cond_embeds,
            cached_noise_cond,
            cached_noise_uncond,
            guidance_scale=guidance_scale,
            w1=jnp.float32(w1),
            w2=jnp.float32(w2),
            kv_cache=kv_cache_cond,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask_cond,
        )
        noise_pred = _slice_target(noise_pred)
        cached_noise_cond = _slice_target(cached_noise_cond)

      elif do_cfg:
        latents_doubled = jnp.concatenate([latents] * 2)
        timestep = jnp.broadcast_to(t, bsz * 2)
        (
            noise_pred,
            cached_noise_cond,
            cached_noise_uncond,
        ) = transformer_forward_pass_full_cfg(
            graphdef,
            sharded_state,
            rest_of_state,
            _prepend_cond(latents_doubled),
            timestep,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
        noise_pred = _slice_target(noise_pred)
        cached_noise_cond = _slice_target(cached_noise_cond)
        cached_noise_uncond = _slice_target(cached_noise_uncond)

      else:
        timestep = jnp.broadcast_to(t, bsz)
        noise_pred, latents = transformer_forward_pass(
            graphdef,
            sharded_state,
            rest_of_state,
            _prepend_cond(latents),
            timestep,
            prompt_cond_embeds,
            do_classifier_free_guidance=False,
            guidance_scale=guidance_scale,
            kv_cache=kv_cache,
            rotary_emb=rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
        noise_pred = _slice_target(noise_pred)
        latents = _slice_target(latents)

    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

    if config and max_utils.profiler_enabled(config) and step == last_profiling_step:
      if profiler:
        latents.block_until_ready()
        profiler.stop()

  return latents
