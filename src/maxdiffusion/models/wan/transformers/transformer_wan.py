"""
Copyright 2025 Google LLC

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

from typing import Tuple, Optional, Dict, Union, Any
import contextlib
import math
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from flax import nnx
import flax.linen as nn
import numpy as np
from .... import common_types
from ...modeling_flax_utils import FlaxModelMixin, get_activation
from ....configuration_utils import ConfigMixin, register_to_config
from ...embeddings_flax import (
    NNXWanImageEmbedding,
    get_1d_rotary_pos_embed,
    NNXFlaxTimesteps,
    NNXTimestepEmbedding,
    NNXPixArtAlphaTextProjection,
)
from ...normalization_flax import FP32LayerNorm
from ...attention_flax import FlaxWanAttention
from ...gradient_checkpoint import GradientCheckpointType
from ..wan_lora import WanLoRAAdapter
from ..memory_retriever import CrossAttention as MemoryCrossAttention, MemoryRetriever

BlockSizes = common_types.BlockSizes


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freq = get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32, use_real=False)
    freqs.append(freq)
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6

  dims = [t_size, hw_size, hw_size]

  # Calculate split indices as a static list of integers
  cumulative_sizes = np.cumsum(dims)
  split_indices = cumulative_sizes[:-1].tolist()
  freqs_split = jnp.split(freqs, split_indices, axis=1)
  return freqs_split


class WanRotaryPosEmbed(nnx.Module):

  def __init__(
      self,
      attention_head_dim: int,
      patch_size: Tuple[int, int, int],
      max_seq_len: int,
      theta: float = 10000.0,
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)

    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))
    return freqs_final


class WanTimeTextImageEmbedding(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      time_freq_dim: int,
      time_proj_dim: int,
      text_embed_dim: int,
      image_embed_dim: Optional[int] = None,
      pos_embed_seq_len: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      flash_min_seq_length: int = 4096,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = NNXTimestepEmbedding(
        rngs=rngs,
        in_channels=time_freq_dim,
        time_embed_dim=dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=time_proj_dim,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
        rngs=rngs,
        in_features=text_embed_dim,
        hidden_size=dim,
        act_fn="gelu_tanh",
    )

    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = NNXWanImageEmbedding(
          rngs=rngs,
          in_features=image_embed_dim,
          out_features=dim,
          pos_embed_seq_len=pos_embed_seq_len,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          flash_min_seq_length=flash_min_seq_length,
      )

  def __call__(
      self,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      skip_embeddings: bool = False,
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))

    if not skip_embeddings:
      encoder_hidden_states = self.text_embedder(encoder_hidden_states)
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None:
        (
            encoder_hidden_states_image,
            encoder_attention_mask,
        ) = self.image_embedder(encoder_hidden_states_image)
    else:
      encoder_attention_mask = None
      if (
          encoder_hidden_states_image is not None
          and encoder_hidden_states_image.shape[-1] != encoder_hidden_states.shape[-1]
      ):
        img_dim = encoder_hidden_states_image.shape[-1]
        text_dim = encoder_hidden_states.shape[-1]
        if img_dim < text_dim:
          pad_shape = (
              encoder_hidden_states_image.shape[0],
              encoder_hidden_states_image.shape[1],
              text_dim - img_dim,
          )
          encoder_hidden_states_image = jnp.concatenate(
              [
                  encoder_hidden_states_image,
                  jnp.zeros(pad_shape, dtype=encoder_hidden_states_image.dtype),
              ],
              axis=-1,
          )
        else:
          pad_shape = (
              encoder_hidden_states.shape[0],
              encoder_hidden_states.shape[1],
              img_dim - text_dim,
          )
          encoder_hidden_states = jnp.concatenate(
              [encoder_hidden_states, jnp.zeros(pad_shape, dtype=encoder_hidden_states.dtype)], axis=-1
          )

    return (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        encoder_attention_mask,
    )


class ApproximateGELU(nnx.Module):
  r"""
  The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim_in: int,
      dim_out: int,
      bias: bool,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      lora_rank: int = 0,
      lora_alpha: float = 0.0,
      lora_target_modules: Optional[tuple] = None,
  ):
    self.proj = nnx.Linear(
        rngs=rngs,
        in_features=dim_in,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    # LoRA on ffn.0 (the first FFN linear); matches DiffSynth/Captain-Safari.
    _targets = lora_target_modules if lora_target_modules is not None else ("ffn.0",)
    self.lora_ffn0 = nnx.data(None)
    if lora_rank > 0 and "ffn.0" in _targets:
      effective_alpha = float(lora_rank) if lora_alpha <= 0.0 else lora_alpha
      self.lora_ffn0 = WanLoRAAdapter(dim_in, dim_out, lora_rank, effective_alpha, dtype, weights_dtype, precision, rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    with jax.named_scope("gelu"):
      h = self.proj(x)
      if self.lora_ffn0 is not None:
        h = h + self.lora_ffn0(x)
    return nnx.gelu(h)


class WanFeedForward(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      dropout: float = 0.0,
      activation_fn: str = "geglu",
      final_dropout: bool = False,
      inner_dim: int = None,
      bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      enable_jax_named_scopes: bool = False,
      lora_rank: int = 0,
      lora_alpha: float = 0.0,
      lora_target_modules: Optional[tuple] = None,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.act_fn = nnx.data(None)
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(
          rngs=rngs,
          dim_in=dim,
          dim_out=inner_dim,
          bias=bias,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          lora_rank=lora_rank,
          lora_alpha=lora_alpha,
          lora_target_modules=lora_target_modules,
      )
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")

    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "mlp",
                "embed",
            ),
        ),
    )
    # LoRA on ffn.2 (proj_out); matches DiffSynth/Captain-Safari.
    _ffn2_targets = lora_target_modules if lora_target_modules is not None else ("ffn.2",)
    self.lora_ffn2 = nnx.data(None)
    if lora_rank > 0 and "ffn.2" in _ffn2_targets:
      effective_alpha = float(lora_rank) if lora_alpha <= 0.0 else lora_alpha
      self.lora_ffn2 = WanLoRAAdapter(inner_dim, dim_out, lora_rank, effective_alpha, dtype, weights_dtype, precision, rngs)

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    act = self.act_fn(hidden_states)  # Output is (4, 75600, 13824)
    act = checkpoint_name(act, "ffn_activation")
    if self.drop_out.rate > 0:
      act = self.drop_out(act, deterministic=deterministic, rngs=rngs)
    with jax.named_scope("proj_out"):
      out = self.proj_out(act)  # output is (4, 75600, 5120)
      if self.lora_ffn2 is not None:
        out = out + self.lora_ffn2(act)
      return out


class WanCameraResidualBlock(nnx.Module):
  """DiffSynth/Captain-Safari residual block for camera Plucker features."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out"))
    bias_init = nnx.with_partitioning(nnx.initializers.zeros, ("conv_out",))
    self.conv1 = nnx.Conv(
        dim,
        dim,
        rngs=rngs,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )
    self.conv2 = nnx.Conv(
        dim,
        dim,
        rngs=rngs,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x
    x = jax.nn.relu(self.conv1(x))
    x = self.conv2(x)
    return x + residual


class WanCameraSimpleAdapter(nnx.Module):
  """DiffSynth SimpleAdapter port for official WAN Fun camera-control.

  Input: packed Plucker control latents [B, 24, T_lat, H, W].
  Output: patch-token feature map [B, T_lat, H/32, W/32, dim] for PAI 5B
  when downscale_factor=16 and patch stride=(2,2).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_dim: int,
      out_dim: int,
      kernel_size: Tuple[int, int],
      stride: Tuple[int, int],
      downscale_factor: int = 8,
      num_residual_blocks: int = 1,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.downscale_factor = downscale_factor
    kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, None, "conv_out"))
    bias_init = nnx.with_partitioning(nnx.initializers.zeros, ("conv_out",))
    self.conv = nnx.Conv(
        in_dim * downscale_factor * downscale_factor,
        out_dim,
        rngs=rngs,
        kernel_size=kernel_size,
        strides=stride,
        padding="VALID",
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )
    self.num_residual_blocks = num_residual_blocks
    for i in range(num_residual_blocks):
      setattr(self, f"residual_blocks_{i}", WanCameraResidualBlock(rngs, out_dim, dtype, weights_dtype, precision))

  def _pixel_unshuffle(self, x: jax.Array) -> jax.Array:
    factor = self.downscale_factor
    bf, height, width, channels = x.shape
    if height % factor != 0 or width % factor != 0:
      raise ValueError(f"Camera control spatial shape {(height, width)} must be divisible by {factor}.")
    x = x.reshape(bf, height // factor, factor, width // factor, factor, channels)
    # Match PyTorch nn.PixelUnshuffle channel order [C, r, r] (channel-major), not
    # [r, r, C]. The checkpoint's control_adapter.conv weights assume PyTorch order;
    # using [r, r, C] permutes the conv's input channels and scrambles the camera
    # control signal (model ignores camera -> falls back to prompt-default motion).
    # dims here are (bf, h, i, w, j, c) -> reorder to (bf, h, w, c, i, j).
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape(bf, height // factor, width // factor, channels * factor * factor)

  def __call__(self, x: jax.Array) -> jax.Array:
    batch, channels, frames, height, width = x.shape
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    x = x.reshape(batch * frames, height, width, channels)
    x = self._pixel_unshuffle(x)
    x = self.conv(x)
    for i in range(self.num_residual_blocks):
      x = getattr(self, f"residual_blocks_{i}")(x)
    out_height, out_width, out_channels = x.shape[1], x.shape[2], x.shape[3]
    return x.reshape(batch, frames, out_height, out_width, out_channels)


class WanTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      added_kv_proj_dim: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      mask_padding_tokens: bool = True,
      enable_jax_named_scopes: bool = False,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
      lora_rank: int = 0,
      lora_alpha: float = 0.0,
      lora_target_modules: Optional[tuple] = None,
      use_memory: bool = False,
  ):
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.use_memory = use_memory

    # 1. Self-attention
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=True,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="self_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
    )

    # 1. Cross-attention
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        added_kv_proj_dim=added_kv_proj_dim,
        image_seq_len=image_seq_len,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="cross_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
    )
    assert cross_attn_norm is True
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)

    # 3. Feed-forward
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        dropout=dropout,
        enable_jax_named_scopes=enable_jax_named_scopes,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

    # Captain-Safari memory cross-attention (optional). Mirrors attn2 but reads
    # the retrieved memory_context; structurally == CS DiTBlock.memory_cross_attn
    # (a CrossAttention) + norm_memory (LayerNorm). Injected after text cross-attn.
    if use_memory:
      self.memory_cross_attn = MemoryCrossAttention(dim, num_heads, eps, rngs=rngs)
      self.norm_memory = nnx.LayerNorm(dim, epsilon=eps, dtype=dtype, param_dtype=weights_dtype, rngs=rngs)
    else:
      self.memory_cross_attn = nnx.data(None)
      self.norm_memory = nnx.data(None)

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
      memory_context: Optional[jax.Array] = None,
  ):
    with self.conditional_named_scope("transformer_block"):
      # Support both global [B, 6, dim] and per-token [B, seq_len, 6, dim] temb.
      # Per-token temb is used by TI2V where first-frame tokens have timestep=0.
      if temb.ndim == 4:  # Per-token: [B, seq_len, 6, dim]
        adaln = jnp.expand_dims(self.adaln_scale_shift_table, 0)  # [1, 1, 6, dim]
        combined = adaln + temb.astype(jnp.float32)  # [B, seq_len, 6, dim]
        parts = jnp.split(combined, 6, axis=2)
        shift_msa = parts[0].squeeze(2)
        scale_msa = parts[1].squeeze(2)
        gate_msa = parts[2].squeeze(2)
        c_shift_msa = parts[3].squeeze(2)
        c_scale_msa = parts[4].squeeze(2)
        c_gate_msa = parts[5].squeeze(2)
      else:  # Global: [B, 6, dim]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = jnp.split(
            (self.adaln_scale_shift_table + temb.astype(jnp.float32)),
            6,
            axis=1,
        )
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
      hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
      hidden_states = checkpoint_name(hidden_states, "hidden_states")
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_kv"))
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)

      # 1. Self-attention
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("self_attn_attn"):
          attn_output = self.attn1(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=norm_hidden_states,
              rotary_emb=rotary_emb,
              deterministic=deterministic,
              rngs=rngs,
          )
        with self.conditional_named_scope("self_attn_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)

      # 2. Cross-attention
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          attn_output = self.attn2(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              deterministic=deterministic,
              rngs=rngs,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=cached_kv,
          )
        with self.conditional_named_scope("cross_attn_residual"):
          hidden_states = hidden_states + attn_output

      # 2b. Memory cross-attention (Captain-Safari): video tokens attend to the
      # retrieved 3D-memory tokens. norm_memory(x) is the query; memory_context
      # is K/V. Added as a residual, same place as CS DiTBlock.
      if self.use_memory and memory_context is not None:
        with self.conditional_named_scope("memory_cross_attn"):
          norm_hidden_states = self.norm_memory(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
          mem_out = self.memory_cross_attn(norm_hidden_states, memory_context).astype(hidden_states.dtype)
          hidden_states = hidden_states + mem_out

      # 3. Feed-forward
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(
              hidden_states.dtype
          )
      return hidden_states

  def compute_kv(
      self,
      encoder_hidden_states: jax.Array,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    return self.attn2.compute_kv(encoder_hidden_states, encoder_attention_mask)


class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size: Tuple[int] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      eps: float = 1e-6,
      image_dim: Optional[int] = None,
      added_kv_proj_dim: Optional[int] = None,
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
      add_control_adapter: bool = False,
      in_dim_control_adapter: int = 24,
      downscale_factor_control_adapter: int = 8,
      lora_rank: int = 0,
      lora_alpha: float = 0.0,
      lora_target_modules: Optional[tuple] = None,
      use_memory: bool = False,
      memory_dim: int = 1024,
      memory_heads: int = 8,
      memory_blocks: int = 1,
  ):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.use_memory = use_memory

    # 1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )
    # Camera control adapter (PAI WAN Fun camera-control). Plucker control
    # latents are mapped to patch-token features and added to the patch
    # embedding output before the transformer blocks.
    self.control_adapter = nnx.data(None)
    if add_control_adapter:
      self.control_adapter = WanCameraSimpleAdapter(
          rngs=rngs,
          in_dim=in_dim_control_adapter,
          out_dim=inner_dim,
          kernel_size=patch_size[1:],
          stride=patch_size[1:],
          downscale_factor=downscale_factor_control_adapter,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
        flash_min_seq_length=flash_min_seq_length,
    )

    # 3. Transformer blocks
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )
    def init_block(rngs):
      return WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          dropout=dropout,
          mask_padding_tokens=mask_padding_tokens,
          enable_jax_named_scopes=enable_jax_named_scopes,
          added_kv_proj_dim=added_kv_proj_dim,
          image_seq_len=image_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          lora_rank=lora_rank,
          lora_alpha=lora_alpha,
          lora_target_modules=lora_target_modules,
          use_memory=use_memory,
      )

    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved
    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(num_layers):
        block = WanTransformerBlock(
            rngs=rngs,
            dim=inner_dim,
            ffn_dim=ffn_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            image_seq_len=image_seq_len,
            flash_min_seq_length=flash_min_seq_length,
            flash_block_sizes=flash_block_sizes,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            attention=attention,
            enable_jax_named_scopes=enable_jax_named_scopes,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            use_memory=use_memory,
        )
        blocks.append(block)
      self.blocks = nnx.data(blocks)

    # Captain-Safari memory (optional): the pose-conditioned retriever + the
    # memory_emb MLP (1024 -> inner_dim, SiLU) that lifts retrieved memory into
    # the DiT token space for the per-block memory cross-attention.
    if use_memory:
      self.memory_retriever = MemoryRetriever(
          dim=memory_dim, num_heads=memory_heads, num_blocks=memory_blocks, rngs=rngs)
      self.memory_emb_0 = nnx.Linear(memory_dim, inner_dim, dtype=dtype, param_dtype=weights_dtype, rngs=rngs)
      self.memory_emb_2 = nnx.Linear(inner_dim, inner_dim, dtype=dtype, param_dtype=weights_dtype, rngs=rngs)
    else:
      self.memory_retriever = nnx.data(None)
      self.memory_emb_0 = nnx.data(None)
      self.memory_emb_2 = nnx.data(None)

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "embed")),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def compute_kv_cache(
      self,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      timestep: Optional[jax.Array] = None,
  ) -> Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Optional[jax.Array]]:
    if timestep is None:
      batch_size = encoder_hidden_states.shape[0]
      timestep = jnp.zeros((batch_size,), dtype=jnp.int32)

    with self.conditional_named_scope("condition_embedder"):
      (
          temb,
          timestep_proj,
          encoder_hidden_states,
          encoder_hidden_states_image,
          encoder_attention_mask,
      ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)
      if encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)

    if self.scan_layers:

      @nnx.vmap(
          in_axes=(0, None, None),
          out_axes=0,
          transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
      )
      def _compute_kv(block, enc_states, enc_mask):
        return block.compute_kv(enc_states, enc_mask)

      kv_cache = _compute_kv(self.blocks, encoder_hidden_states, encoder_attention_mask)
    else:
      kv_cache_list = []
      for block in self.blocks:
        kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
      keys = kv_cache_list[0].keys()
      kv_cache = {}
      for k in keys:
        k_list = [d[k][0] for d in kv_cache_list]
        v_list = [d[k][1] for d in kv_cache_list]
        kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))

    return kv_cache, encoder_attention_mask

  @jax.named_scope("WanModel")
  def compute_memory_context(self, memory, memory_pose_token, memory_key_pose_token):
    """Retrieve pose-aligned memory and lift it to the DiT dim.

    Independent of the noisy latents and timestep, so the generation loop computes this
    ONCE and feeds it back in as `memory_context` (skipping the per-step retriever).
    Returns [B, 4*782, inner_dim]. Requires use_memory=True."""
    mem_pred = self.memory_retriever(memory_pose_token, memory_key_pose_token, memory)
    return self.memory_emb_2(nnx.silu(self.memory_emb_0(mem_pred)))

  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      skip_blocks: Optional[jax.Array] = None,
      cached_residual: Optional[jax.Array] = None,
      return_residual: bool = False,
      kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
      rotary_emb: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      control_camera_latents_input: Optional[jax.Array] = None,
      memory: Optional[jax.Array] = None,
      memory_pose_token: Optional[jax.Array] = None,
      memory_key_pose_token: Optional[jax.Array] = None,
      memory_context: Optional[jax.Array] = None,
  ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], Dict[str, jax.Array]]:
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      if rotary_emb is None:
        rotary_emb = self.rope(hidden_states)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      if self.control_adapter is not None and control_camera_latents_input is not None:
        camera_states = self.control_adapter(control_camera_latents_input)
        hidden_states = hidden_states + camera_states.astype(hidden_states.dtype)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    per_token_t = timestep.ndim == 2  # [B, seq_len] for TI2V
    with self.conditional_named_scope("condition_embedder"):
      if per_token_t:
        # Per-token timestep: process time and text embeddings separately.
        # This matches the official WAN 2.2 TI2V pipeline where first-frame
        # tokens receive timestep=0 (clean) and other tokens receive timestep=t.
        bt, sl = timestep.shape
        t_flat = timestep.reshape(-1)  # [B*seq_len]
        t_sinusoidal = self.condition_embedder.timesteps_proj(t_flat)  # [B*sl, freq_dim]
        t_sinusoidal = t_sinusoidal.reshape(bt, sl, -1)  # [B, sl, freq_dim]
        temb = self.condition_embedder.time_embedder(t_sinusoidal)  # [B, sl, dim]
        with jax.named_scope("time_proj"):
          timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))  # [B, sl, dim*6]
        timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)  # [B, sl, 6, dim]
        # Text processing
        if kv_cache is None:
          encoder_hidden_states_out = self.condition_embedder.text_embedder(encoder_hidden_states)
        else:
          encoder_hidden_states_out = encoder_hidden_states
        encoder_hidden_states_image_out = None
        encoder_attention_mask_out = None
      else:
        (
            temb,
            timestep_proj,
            encoder_hidden_states_out,
            encoder_hidden_states_image_out,
            encoder_attention_mask_out,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            skip_embeddings=(kv_cache is not None),
        )
        timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_attention_mask is None:
      encoder_attention_mask = encoder_attention_mask_out

    if encoder_hidden_states_image_out is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image_out, encoder_hidden_states_out], axis=1)
      if kv_cache is None and encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image_out.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)
      encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)
    else:
      encoder_hidden_states = encoder_hidden_states_out.astype(hidden_states.dtype)

    # Captain-Safari memory: retrieve pose-aligned memory once, lift to DiT space.
    # Block-constant (same for every layer), so it's closed over like the text states.
    if self.use_memory and memory_context is None and memory is not None:
      mem_pred = self.memory_retriever(memory_pose_token, memory_key_pose_token, memory)
      memory_context = self.memory_emb_2(nnx.silu(self.memory_emb_0(mem_pred))).astype(hidden_states.dtype)

    def _run_all_blocks(h):
      if self.scan_layers:

        def scan_fn(carry, block_input):
          hidden_states_carry, rngs_carry = carry
          if kv_cache is not None:
            block, layer_kv_cache = block_input
          else:
            block = block_input
            layer_kv_cache = None

          hidden_states = block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask,
              cached_kv=layer_kv_cache,
              memory_context=memory_context,
          )
          new_carry = (hidden_states, rngs_carry)
          return new_carry, None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        initial_carry = (h, rngs)

        if kv_cache is not None:
          scan_input = (self.blocks, kv_cache)
        else:
          scan_input = self.blocks

        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_layers,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, scan_input)

        h_out, _ = final_carry
      else:
        h_out = h
        for i, block in enumerate(self.blocks):
          layer_kv_cache = None
          if kv_cache is not None:
            layer_kv_cache = jax.tree.map(lambda x: x[i], kv_cache)

          def layer_forward(hidden_states, l_kv):
            return block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                deterministic,
                rngs,
                encoder_attention_mask=encoder_attention_mask,
                cached_kv=l_kv,
                memory_context=memory_context,
            )

          rematted_layer_forward = self.gradient_checkpoint.apply(
              layer_forward,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          h_out = rematted_layer_forward(h_out, layer_kv_cache)
      return h_out

    hidden_states_before_blocks = hidden_states

    if skip_blocks:
      if cached_residual is None:
        raise ValueError("cached_residual must be provided when skip_blocks is True")
      hidden_states = hidden_states + cached_residual
    else:
      hidden_states = _run_all_blocks(hidden_states)

    residual_x = hidden_states - hidden_states_before_blocks

    if per_token_t:
      # temb: [B, seq_len, dim] — per-token modulation for final head
      combined_head = jnp.expand_dims(self.scale_shift_table, 0) + jnp.expand_dims(temb, 2)  # [B, sl, 2, dim]
      shift, scale = jnp.split(combined_head, 2, axis=2)
      shift = shift.squeeze(2)  # [B, sl, dim]
      scale = scale.squeeze(2)  # [B, sl, dim]
    else:
      shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if return_residual:
      return hidden_states, residual_x
    return hidden_states
