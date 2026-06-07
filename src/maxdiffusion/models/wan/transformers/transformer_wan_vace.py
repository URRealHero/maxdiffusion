"""Copyright 2025 Google LLC

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

import contextlib
import math
from typing import Any, Dict, Optional, Tuple

from flax import nnx
import flax.linen as nn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from .... import common_types
from ....configuration_utils import register_to_config
from ...attention_flax import FlaxWanAttention
from ...gradient_checkpoint import GradientCheckpointType
from ...normalization_flax import FP32LayerNorm
from .transformer_wan import WanFeedForward, WanModel, WanRotaryPosEmbed, WanTimeTextImageEmbedding, WanTransformerBlock


BlockSizes = common_types.BlockSizes


class WanVACETransformerBlock(nnx.Module):
  """Attention block for VACE.

  Processes the conditioning signals and produces latent codes that can be
  summed to the main branch of WAN.

  Based on
  https://github.com/huggingface/diffusers/blob/be3c2a0667493022f17d756ca3dba631d28dfb40/src/diffusers/models/transformers/transformer_wan_vace.py#L41C7-L41C30
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      *,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes | None = None,
      mesh: jax.sharding.Mesh | None = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision | None = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      mask_padding_tokens: bool = True,
      enable_jax_named_scopes: bool = False,
      apply_input_projection: bool = False,
      apply_output_projection: bool = False,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
  ):
    """Sets up the model.

    Args:
      rngs: Random number generator.
      dim: Internal dimension of the block.
      ffn_dim: Dimension of the feed-forward network.
      num_heads: Number of attention heads.
      qk_norm: Whether to apply RMSNorm to the query and key vectors.
      cross_attn_norm: Whether to apply layer normalization before
        cross-attention (only True supported).
      eps: Epsilon value for normalization.
      flash_min_seq_length: Minimum sequence length for flash attention.
      flash_block_sizes: Block sizes for flash attention.
      mesh: Sharding topology.
      dtype: Data type for the computation.
      weights_dtype: Data type for parameter initializers (see param_dtype in
        nnx.Linear).
      precision: Precision for the computation.
      attention: Type of attention to use.
      dropout: Dropout rate.
      apply_input_projection: Whether to apply a linear projection to the
        inputs.
      apply_output_projection: Whether to apply an output projection before
        outputting the result.
    """
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.apply_input_projection = apply_input_projection
    self.apply_output_projection = apply_output_projection

    # 1. Input projection
    self.proj_in = nnx.data([None])
    if apply_input_projection:
      self.proj_in = nnx.Linear(
          rngs=rngs,
          in_features=dim,
          out_features=dim,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
      )

    # 2. Self-attention
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
    )

    # 3. Cross-attention
    self.attn2 = FlaxWanAttention(
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
        is_self_attention=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="cross_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        use_base2_exp=use_base2_exp,
        use_experimental_scheduler=use_experimental_scheduler,
    )
    assert cross_attn_norm is True, "cross_attn_norm must be True"
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)

    # 4. Feed-forward
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
    )

    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)

    # 5. Output projection
    self.proj_out = nnx.data([None])
    if apply_output_projection:
      self.proj_out = nnx.Linear(
          rngs=rngs,
          in_features=dim,
          out_features=dim,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
      )

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None):
    return self.attn2.compute_kv(encoder_hidden_states, encoder_attention_mask)

  def __call__(
      self,
      *,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      control_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs | None = None,
      input_projection_scale: Optional[jax.Array] = None,
  ) -> Tuple[jax.Array, jax.Array]:
    with self.conditional_named_scope("vace_transformer_block"):
      with self.conditional_named_scope("input_projection"):
        if self.apply_input_projection:
          projected_control_hidden_states = self.proj_in(control_hidden_states) + hidden_states
          if input_projection_scale is None:
            control_hidden_states = projected_control_hidden_states
          else:
            projection_scale = input_projection_scale.astype(control_hidden_states.dtype)
            control_hidden_states = (
                projected_control_hidden_states * projection_scale + control_hidden_states * (1 - projection_scale)
            ).astype(control_hidden_states.dtype)

      shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
          (self.adaln_scale_shift_table + temb.astype(jnp.float32)), 6, axis=1
      )

      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
      control_hidden_states = jax.lax.with_sharding_constraint(control_hidden_states, axis_names)
      control_hidden_states = checkpoint_name(control_hidden_states, "control_hidden_states")
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_kv"))
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)

      # 1. Self-attention
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(control_hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
              control_hidden_states.dtype
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
          control_hidden_states = (control_hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(
              control_hidden_states.dtype
          )

      # 2. Cross-attention
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(control_hidden_states.astype(jnp.float32)).astype(control_hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          attn_output = self.attn2(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              cached_kv=kv_cache,
              encoder_attention_mask=encoder_attention_mask,
              deterministic=deterministic,
              rngs=rngs,
          )
        with self.conditional_named_scope("cross_attn_residual"):
          control_hidden_states = control_hidden_states + attn_output

      # 3. Feed-forward
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (
              self.norm3(control_hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa
          ).astype(control_hidden_states.dtype)
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          control_hidden_states = (
              control_hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa
          ).astype(control_hidden_states.dtype)

      with self.conditional_named_scope("output_projection"):
        conditioning_states = None
        if self.apply_output_projection:
          conditioning_states = self.proj_out(control_hidden_states)

      return conditioning_states, control_hidden_states


class WanVACEModel(WanModel):
  """Extension of Wan to include VACE conditioning."""

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      vace_layers: list[int],
      vace_in_channels: int,
      model_type="t2v",
      patch_size: Tuple[int, ...] = (1, 2, 2),
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
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: list[str] = [],
      names_which_can_be_offloaded: list[str] = [],
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
      debug_vace_numerics: bool = False,
  ):
    """Initializes the VACE model.

    All arguments are similar to WanModel with the exception of:
      vace_layers: Indices of the layers at which the VACE conditioning is
      injected.
      vace_in_channels: Number of channels in the VACE conditioning.
    """
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.debug_vace_numerics = debug_vace_numerics

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

    # 2. Condition embeddings
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

    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved

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
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
      )

    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = nnx.List([])
      for _ in range(num_layers):
        block = WanTransformerBlock(
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
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
        )
        blocks.append(block)
      self.blocks = blocks

    # Sparse VACE scan: scan_layers=True but only a subset of layers have VACE
    # blocks. Main blocks stay vmapped (scan-efficient), VACE blocks are kept as
    # an nnx.List and run sequentially before the main-block scan (only 6 of them
    # so the extra collective count is negligible vs the 40 scanned main blocks).
    self._is_sparse_vace = scan_layers and (list(self.config.vace_layers) != list(range(num_layers)))

    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )
    def init_vace_block(rngs):
      return WanVACETransformerBlock(
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
          apply_input_projection=True,
          apply_output_projection=True,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
      )

    if scan_layers and not self._is_sparse_vace:
      # Dense VACE scan: every layer has a VACE block; use vmapped blocks.
      self.vace_blocks = init_vace_block(rngs)
    else:
      # Non-scan path or sparse-VACE-scan: VACE blocks as plain nnx.List.
      vace_blocks = nnx.List([])
      for vace_block_id in self.config.vace_layers:
        vace_block = WanVACETransformerBlock(
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
            apply_input_projection=vace_block_id == 0,
            apply_output_projection=True,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
        )
        vace_blocks.append(vace_block)
      self.vace_blocks = vace_blocks

    self.vace_patch_embedding = nnx.Conv(
        rngs=rngs,
        in_features=vace_in_channels,
        out_features=inner_dim,
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

  def debug_finite(self, name: str, value: jax.Array):
    if self.debug_vace_numerics:
      value_f32 = value.astype(jnp.float32)
      finite = jnp.mean(jnp.isfinite(value_f32))
      max_abs = jnp.max(jnp.abs(value_f32))
      jax.debug.print("VACE numerics {name}: finite={finite} max_abs={max_abs}", name=name, finite=finite, max_abs=max_abs)

  def compute_kv_cache(
      self,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      timestep: Optional[jax.Array] = None,
  ) -> Tuple[Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Dict[str, Tuple[jax.Array, jax.Array]]], Optional[jax.Array]]:
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

      if self._is_sparse_vace:
        # VACE blocks are nnx.List (not vmapped); compute KV caches with a loop.
        vace_kv_cache_list = []
        for block in self.vace_blocks:
          vace_kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
        vace_kv_cache = {}
        if vace_kv_cache_list:
          for k in vace_kv_cache_list[0].keys():
            vace_kv_cache[k] = (
                jnp.stack([d[k][0] for d in vace_kv_cache_list], axis=0),
                jnp.stack([d[k][1] for d in vace_kv_cache_list], axis=0),
            )
      else:
        vace_kv_cache = _compute_kv(self.vace_blocks, encoder_hidden_states, encoder_attention_mask)
      main_kv_cache = _compute_kv(self.blocks, encoder_hidden_states, encoder_attention_mask)
    else:
      vace_kv_cache_list = []
      for block in self.vace_blocks:
        vace_kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
      vace_kv_cache = {}
      if vace_kv_cache_list:
        keys = vace_kv_cache_list[0].keys()
        for k in keys:
          k_list = [d[k][0] for d in vace_kv_cache_list]
          v_list = [d[k][1] for d in vace_kv_cache_list]
          vace_kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))

      main_kv_cache_list = []
      for block in self.blocks:
        main_kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
      main_kv_cache = {}
      if main_kv_cache_list:
        keys = main_kv_cache_list[0].keys()
        for k in keys:
          k_list = [d[k][0] for d in main_kv_cache_list]
          v_list = [d[k][1] for d in main_kv_cache_list]
          main_kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))

    return (vace_kv_cache, main_kv_cache), encoder_attention_mask

  @jax.named_scope("WanVACEModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      control_hidden_states: jax.Array,
      control_hidden_states_scale: Optional[jax.Array] = None,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      kv_cache: Optional[Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Dict[str, Tuple[jax.Array, jax.Array]]]] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    if control_hidden_states_scale is None:
      control_hidden_states_scale = jnp.ones((len(self.config.vace_layers),), dtype=control_hidden_states.dtype)
    if control_hidden_states_scale.shape[0] != len(self.config.vace_layers):
      raise ValueError(
          "Length of `control_hidden_states_scale`"
          f" {len(control_hidden_states_scale)} should be equal to"
          f" {len(self.config.vace_layers)}."
      )

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    control_hidden_states = jnp.transpose(control_hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      rotary_emb = self.rope(hidden_states)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
      self.debug_finite("main_patch_embedding", hidden_states)

      control_hidden_states = self.vace_patch_embedding(control_hidden_states)
      control_hidden_states = jax.lax.collapse(control_hidden_states, 1, -1)
      self.debug_finite("vace_patch_embedding", control_hidden_states)
    if control_hidden_states.shape[1] < hidden_states.shape[1]:
      control_hidden_states_padding = jnp.zeros(
          (
              batch_size,
              hidden_states.shape[1] - control_hidden_states.shape[1],
              control_hidden_states.shape[2],
          ),
          dtype=control_hidden_states.dtype,
      )
      control_hidden_states = jnp.concatenate([control_hidden_states, control_hidden_states_padding], axis=1)
    elif control_hidden_states.shape[1] > hidden_states.shape[1]:
      raise ValueError(
          "VACE control sequence is longer than the noisy latent sequence: "
          f"{control_hidden_states.shape[1]} > {hidden_states.shape[1]}"
      )

    # Condition embedder is a FC layer.
    with self.conditional_named_scope("condition_embedder"):
      (
          temb,
          timestep_proj,
          encoder_hidden_states,
          encoder_hidden_states_image,
          _,
      ) = self.condition_embedder(  # We will need to mask out the text embedding.
          timestep, encoder_hidden_states, encoder_hidden_states_image, skip_embeddings=(kv_cache is not None)
      )
      timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)
      self.debug_finite("temb", temb)
      self.debug_finite("timestep_proj", timestep_proj)
      self.debug_finite("encoder_hidden_states", encoder_hidden_states)

    if encoder_hidden_states_image is not None:
      raise NotImplementedError("img2vid is not yet implemented.")

    vace_kv_cache, main_kv_cache = kv_cache if kv_cache is not None else (None, None)
    vace_base_hidden_states = hidden_states

    if self.scan_layers:
      control_scales = control_hidden_states_scale.astype(hidden_states.dtype)

      if self._is_sparse_vace:
        # ---- Sparse VACE scan ----
        # Phase 1: run the small VACE block set sequentially (only len(vace_layers)
        # blocks, e.g. 6). control_hidden_states flows through as a chain carry.
        conditioning_states_list = []
        ctrl = control_hidden_states
        for i, vace_block in enumerate(self.vace_blocks):
          _vb = vace_block
          _vkvc = jax.tree.map(lambda x, _i=i: x[_i], vace_kv_cache) if vace_kv_cache is not None else None

          def _vace_fwd(hidden_states, ctrl, rngs, _vb=_vb, _vkvc=_vkvc):
            return _vb(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                control_hidden_states=ctrl,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                kv_cache=_vkvc,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                rngs=rngs,
            )

          rematted_vace = self.gradient_checkpoint.apply(
              _vace_fwd,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          conditioning_state, ctrl = rematted_vace(vace_base_hidden_states, ctrl, rngs)
          self.debug_finite(f"sparse_vace_block_{i}_conditioning", conditioning_state)
          conditioning_states_list.append(conditioning_state)

        # Stack all conditioning outputs: [num_vace_layers, B, S, D]
        vace_skips = jnp.stack(conditioning_states_list, axis=0)

        # Build static per-main-layer lookup arrays (Python loops over static config).
        vace_layers_list = list(self.config.vace_layers)
        vace_layers_set = set(vace_layers_list)
        _vace_idx, _is_vace, _ctrl_scale = [], [], []
        for j in range(self.num_layers):
          if j in vace_layers_set:
            vi = vace_layers_list.index(j)
            _vace_idx.append(vi)
            _is_vace.append(True)
            _ctrl_scale.append(control_scales[vi])
          else:
            _vace_idx.append(0)  # safe index — unused when is_vace=False
            _is_vace.append(False)
            _ctrl_scale.append(jnp.zeros((), dtype=hidden_states.dtype))
        layer_vace_idx = jnp.array(_vace_idx, dtype=jnp.int32)    # [num_layers]
        layer_is_vace = jnp.array(_is_vace, dtype=jnp.bool_)       # [num_layers]
        layer_ctrl_scale = jnp.stack(_ctrl_scale, axis=0)           # [num_layers]

        # Phase 2: scan all main blocks with conditional VACE skip injection.
        def scan_fn_sparse(carry, block_input):
          hidden_states_carry, rngs_carry = carry
          if main_kv_cache is not None:
            main_block, main_layer_kv_cache, vace_idx, is_vace, ctrl_scale = block_input
          else:
            main_block, vace_idx, is_vace, ctrl_scale = block_input
            main_layer_kv_cache = None

          hidden_states_out = main_block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=main_layer_kv_cache,
          )
          # Dynamic index into pre-computed conditioning states; safe-index (0)
          # for non-VACE layers (masked out by is_vace below).
          skip = jnp.take(vace_skips, vace_idx, axis=0)
          hidden_states_out = jnp.where(
              is_vace,
              hidden_states_out + skip * ctrl_scale.astype(skip.dtype),
              hidden_states_out,
          )
          return (hidden_states_out, rngs_carry), None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn_sparse,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        initial_carry = (hidden_states, rngs)
        if main_kv_cache is not None:
          scan_input = (self.blocks, main_kv_cache, layer_vace_idx, layer_is_vace, layer_ctrl_scale)
        else:
          scan_input = (self.blocks, layer_vace_idx, layer_is_vace, layer_ctrl_scale)
        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_layers,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, scan_input)
        hidden_states, _ = final_carry

      else:
        # ---- Dense VACE scan (every layer has a VACE block) ----
        layer_indices = jnp.arange(self.num_layers, dtype=jnp.int32)

        def scan_fn(carry, block_input):
          hidden_states_carry, control_hidden_states_carry, rngs_carry = carry
          if main_kv_cache is not None:
            main_block, vace_block, main_layer_kv_cache, vace_layer_kv_cache, layer_idx, control_scale = block_input
          else:
            main_block, vace_block, layer_idx, control_scale = block_input
            main_layer_kv_cache = None
            vace_layer_kv_cache = None

          input_projection_scale = (layer_idx == 0).astype(control_hidden_states_carry.dtype)
          conditioning_states, control_hidden_states_out = vace_block(
              hidden_states=vace_base_hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              control_hidden_states=control_hidden_states_carry,
              temb=timestep_proj,
              rotary_emb=rotary_emb,
              kv_cache=vace_layer_kv_cache,
              encoder_attention_mask=encoder_attention_mask,
              deterministic=deterministic,
              rngs=rngs_carry,
              input_projection_scale=input_projection_scale,
          )

          hidden_states_out = main_block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=main_layer_kv_cache,
          )
          hidden_states_out = hidden_states_out + conditioning_states * control_scale.astype(conditioning_states.dtype)
          return (hidden_states_out, control_hidden_states_out, rngs_carry), None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        initial_carry = (hidden_states, control_hidden_states, rngs)
        if main_kv_cache is not None:
          scan_input = (self.blocks, self.vace_blocks, main_kv_cache, vace_kv_cache, layer_indices, control_scales)
        else:
          scan_input = (self.blocks, self.vace_blocks, layer_indices, control_scales)
        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_layers,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, scan_input)
        hidden_states, _, _ = final_carry
    else:
      # Prepare VACE hints.
      control_hidden_states_list = []
      for i, vace_block in enumerate(self.vace_blocks):
        layer_kv_cache = None
        if vace_kv_cache is not None:
          layer_kv_cache = jax.tree.map(lambda x: x[i], vace_kv_cache)

        def layer_forward(hidden_states, control_hidden_states, rngs):
          return vace_block(
              hidden_states=hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              control_hidden_states=control_hidden_states,
              temb=timestep_proj,
              rotary_emb=rotary_emb,
              kv_cache=layer_kv_cache,
              encoder_attention_mask=encoder_attention_mask,
              deterministic=deterministic,
              rngs=rngs,
          )

        rematted_layer_forward = self.gradient_checkpoint.apply(
            layer_forward,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        conditioning_states, control_hidden_states = rematted_layer_forward(hidden_states, control_hidden_states, rngs)
        self.debug_finite(f"vace_block_{i}_conditioning", conditioning_states)
        self.debug_finite(f"vace_block_{i}_state", control_hidden_states)
        control_hidden_states_list.append(conditioning_states)

      control_hidden_states_list = [
          (control_hidden_states_list[i], control_hidden_states_scale[i])
          for i in range(len(control_hidden_states_list) - 1, -1, -1)
      ]
      for i, block in enumerate(self.blocks):
        layer_kv_cache = None
        if main_kv_cache is not None:
          layer_kv_cache = jax.tree.map(lambda x: x[i], main_kv_cache)

        def layer_forward_vace(hidden_states, rngs):
          return block(
              hidden_states,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=layer_kv_cache,
          )

        rematted_layer_forward = self.gradient_checkpoint.apply(
            layer_forward_vace,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        hidden_states = rematted_layer_forward(hidden_states, rngs)
        self.debug_finite(f"main_block_{i}", hidden_states)
        if i in self.config.vace_layers:
          control_hint, scale = control_hidden_states_list.pop()
          self.debug_finite(f"control_hint_layer_{i}", control_hint)
          hidden_states = hidden_states + control_hint * scale
          self.debug_finite(f"main_block_{i}_after_vace", hidden_states)

    # 6. Output norm, projection & unpatchify
    shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)

    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    self.debug_finite("final_norm", hidden_states)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)  # Linear layer.
    self.debug_finite("final_proj_out", hidden_states)

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
    hidden_states = jax.lax.collapse(hidden_states, 6, None)
    hidden_states = jax.lax.collapse(hidden_states, 4, 6)
    hidden_states = jax.lax.collapse(hidden_states, 2, 4)
    return hidden_states
