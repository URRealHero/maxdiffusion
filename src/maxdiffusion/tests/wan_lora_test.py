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

"""Smoke tests for standard-matrix LoRA on WAN transformer blocks.

Tests three key properties:
  1. Param paths — lora_ params exist at q/k/v/o/ffn0/ffn2 per block.
  2. Zero-delta init — B=0 means lora_rank forward == base forward at init.
  3. Scale — WanLoRAAdapter.scale = alpha/rank (or 1.0 when alpha <= 0).

Run locally (CPU, no TPU):
  cd /home/spu9/maxdiffusion
  python -m pytest src/maxdiffusion/tests/wan_lora_test.py -v
"""

import os
import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning

from .. import pyconfig
from ..max_utils import create_device_mesh, get_flash_block_sizes
from ..models.wan.transformers.transformer_wan import (
    WanTransformerBlock,
    WanFeedForward,
)
from ..models.attention_flax import FlaxWanAttention
from ..models.wan.wan_lora import WanLoRAAdapter
from ..models.wan.wan_utils import _cs_lora_key_to_nnx_path


class CSLoRAKeyRemapTest(unittest.TestCase):
  """The DiffSynth/Captain-Safari PEFT key -> our nnx path mapping (F2)."""

  def test_self_attn(self):
    bi, path = _cs_lora_key_to_nnx_path("blocks.7.self_attn.q.lora_A.default.weight")
    self.assertEqual(bi, 7)
    self.assertEqual(path, ("blocks", "attn1", "lora_q", "lora_A", "kernel"))

  def test_cross_attn(self):
    bi, path = _cs_lora_key_to_nnx_path("blocks.0.cross_attn.v.lora_B.default.weight")
    self.assertEqual((bi, path), (0, ("blocks", "attn2", "lora_v", "lora_B", "kernel")))

  def test_ffn0_and_ffn2(self):
    _, p0 = _cs_lora_key_to_nnx_path("blocks.3.ffn.0.lora_A.default.weight")
    _, p2 = _cs_lora_key_to_nnx_path("blocks.3.ffn.2.lora_B.default.weight")
    self.assertEqual(p0, ("blocks", "ffn", "act_fn", "lora_ffn0", "lora_A", "kernel"))
    self.assertEqual(p2, ("blocks", "ffn", "lora_ffn2", "lora_B", "kernel"))

  def test_memory_keys_skipped(self):
    for k in (
        "blocks.6.memory_cross_attn.k.bias",
        "blocks.6.norm_memory.bias",
        "memory_emb.0.weight",
        "memory_retriever.learnable_query",
    ):
      self.assertIsNone(_cs_lora_key_to_nnx_path(k), f"{k} should be skipped")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_mesh_and_config():
  """Initialize config with single-device mesh, returns (config, mesh)."""
  pyconfig.initialize(
      [None, os.path.join(THIS_DIR, "..", "configs", "base_wan_ti2v_5b.yml")],
      unittest=True,
  )
  config = pyconfig.config
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  return config, mesh


def _lora_param_names(module):
  """Return set of param-path strings containing 'lora_'."""
  graphdef, state = nnx.split(module)
  names = set()
  for path, _ in jax.tree_util.tree_leaves_with_path(state):
    path_str = jax.tree_util.keystr(path)
    if "lora_" in path_str:
      names.add(path_str)
  return names


class WanLoRAAdapterTest(unittest.TestCase):

  def setUp(self):
    config, mesh = _make_mesh_and_config()
    self.config = config
    self.mesh = mesh

  def test_scale_with_explicit_alpha(self):
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      adapter = WanLoRAAdapter(
          in_features=64, out_features=64, rank=8, alpha=16.0,
          dtype=jnp.float32, weights_dtype=jnp.float32, precision=None,
          rngs=nnx.Rngs(0),
      )
    self.assertAlmostEqual(adapter.scale, 16.0 / 8, places=5)

  def test_scale_alpha_equals_rank(self):
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      adapter = WanLoRAAdapter(
          in_features=64, out_features=64, rank=8, alpha=8.0,
          dtype=jnp.float32, weights_dtype=jnp.float32, precision=None,
          rngs=nnx.Rngs(0),
      )
    self.assertAlmostEqual(adapter.scale, 1.0, places=5)

  def test_zero_output_at_init(self):
    """B=0 init means adapter output is exactly zero before any training."""
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      adapter = WanLoRAAdapter(
          in_features=64, out_features=128, rank=4, alpha=4.0,
          dtype=jnp.float32, weights_dtype=jnp.float32, precision=None,
          rngs=nnx.Rngs(0),
      )
      x = jax.random.normal(jax.random.key(1), (2, 16, 64))
      out = adapter(x)
    self.assertEqual(out.shape, (2, 16, 128))
    self.assertTrue(jnp.allclose(out, 0.0), f"Expected zero output, got max={jnp.abs(out).max()}")


class WanFeedForwardLoRATest(unittest.TestCase):

  def setUp(self):
    config, mesh = _make_mesh_and_config()
    self.config = config
    self.mesh = mesh

  def test_lora_param_names(self):
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      ffn = WanFeedForward(
          rngs=nnx.Rngs(0), dim=64, inner_dim=256,
          activation_fn="gelu-approximate",
          lora_rank=4, lora_alpha=0.0,
      )
    names = _lora_param_names(ffn)
    self.assertTrue(any("lora_ffn0" in n for n in names), f"lora_ffn0 missing; got: {names}")
    self.assertTrue(any("lora_ffn2" in n for n in names), f"lora_ffn2 missing; got: {names}")

  def test_zero_delta_at_init(self):
    """ffn lora (B=0 at init) must equal the same ffn with lora disabled.

    Toggling lora on the SAME module isolates the lora delta; comparing two
    separately-constructed modules would instead diff their base weights, since
    building lora consumes extra rng draws and shifts proj_out's init key.
    """
    key = jax.random.key(42)
    x = jax.random.normal(key, (1, 8, 64))
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      ffn = WanFeedForward(rngs=nnx.Rngs(0), dim=64, inner_dim=256, activation_fn="gelu-approximate",
                           lora_rank=4, lora_alpha=0.0)
      out_with = ffn(x)
      # nonzero B -> output MUST change (proves the lora path is actually live)
      ffn.lora_ffn2.lora_B.kernel.value = jnp.ones_like(ffn.lora_ffn2.lora_B.kernel.value)
      ffn.act_fn.lora_ffn0.lora_B.kernel.value = jnp.ones_like(ffn.act_fn.lora_ffn0.lora_B.kernel.value)
      out_active = ffn(x)
      # disable lora entirely -> must match the at-init (B=0) output exactly
      ffn.lora_ffn2 = nnx.data(None)
      ffn.act_fn.lora_ffn0 = nnx.data(None)
      out_without = ffn(x)
    self.assertTrue(
        jnp.allclose(out_with, out_without, atol=1e-5),
        f"LoRA delta non-zero at init: max diff={jnp.abs(out_with - out_without).max()}"
    )
    self.assertGreater(
        float(jnp.abs(out_with - out_active).max()), 1e-3,
        "nonzero LoRA B did not change output — lora path not wired",
    )


class WanTransformerBlockLoRATest(unittest.TestCase):

  def setUp(self):
    config, mesh = _make_mesh_and_config()
    self.config = config
    self.mesh = mesh

  def _make_block(self, lora_rank=0, dim=128, num_heads=2, ffn_dim=256):
    return WanTransformerBlock(
        rngs=nnx.Rngs(0),
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        cross_attn_norm=True,
        attention="dot_product",
        lora_rank=lora_rank,
        lora_alpha=0.0,
    )

  def test_lora_param_names_present(self):
    """Each of the 6 LoRA targets should appear in param names."""
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = self._make_block(lora_rank=4)
    names = _lora_param_names(block)
    expected = ["lora_q", "lora_k", "lora_v", "lora_o", "lora_ffn0", "lora_ffn2"]
    for target in expected:
      self.assertTrue(
          any(target in n for n in names),
          f"'{target}' not found in lora param names.\nAll lora names: {sorted(names)}"
      )

  def test_no_lora_params_when_rank_zero(self):
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = self._make_block(lora_rank=0)
    names = _lora_param_names(block)
    self.assertEqual(len(names), 0, f"Expected no lora params, got: {names}")

  def test_zero_delta_forward(self):
    """Block lora (B=0 at init) must equal the same block with lora disabled.

    Toggle on the SAME module so the base weights are identical (separate
    modules would diff their base init due to extra lora rng draws).
    """
    dim, num_heads, ffn_dim = 128, 2, 256
    seq = 16
    B = 1
    x = jax.random.normal(jax.random.key(7), (B, seq, dim))
    enc = jax.random.normal(jax.random.key(8), (B, seq, dim))
    temb = jax.random.normal(jax.random.key(9), (B, 6, dim))
    # WAN rotary is COMPLEX freqs of head_dim//2 (use_real=False); ones = identity
    # rotation, enough to exercise the block. (head_dim = dim // num_heads.)
    rotary = jnp.ones((B, 1, seq, (dim // num_heads) // 2), dtype=jnp.complex64)

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = self._make_block(lora_rank=4, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
      out_with = block(x, enc, temb, rotary)
      # disable every lora adapter in the block, keep base weights
      for attn in (block.attn1, block.attn2):
        attn.lora_q = attn.lora_k = attn.lora_v = attn.lora_o = nnx.data(None)
      block.ffn.lora_ffn2 = nnx.data(None)
      block.ffn.act_fn.lora_ffn0 = nnx.data(None)
      out_without = block(x, enc, temb, rotary)

    max_diff = float(jnp.abs(out_with - out_without).max())
    self.assertLess(max_diff, 1e-4,
        f"LoRA delta non-zero at init: max_diff={max_diff:.2e} (expected < 1e-4)")

  def test_lora_count(self):
    """Exactly 6 lora_ leaf tensors per block: lora_A+lora_B for q,k,v,o (attn1+attn2), ffn0, ffn2.
    attn1 has q,k,v,o (4 adapters × 2 leaves = 8).
    attn2 has q,k,v,o (4 adapters × 2 leaves = 8).
    ffn has ffn0 + ffn2 (2 adapters × 2 leaves = 4).
    Total = 20 leaves (kernel tensors)."""
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      block = self._make_block(lora_rank=4)
    graphdef, state = nnx.split(block)
    lora_leaves = [
        v for path, v in jax.tree_util.tree_leaves_with_path(state)
        if "lora_" in jax.tree_util.keystr(path)
    ]
    self.assertEqual(len(lora_leaves), 20,
        f"Expected 20 lora leaf tensors, got {len(lora_leaves)}")
