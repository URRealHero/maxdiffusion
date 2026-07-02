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

"""Regression tests for the V2V-2a/2b HyDRA gate in FlaxWanAttention.

WanTransformerBlock calls attn1 with encoder_hidden_states=norm_hidden_states
(non-None). The hydra branch must therefore NOT be gated on a runtime
`encoder_hidden_states is None` check: with that gate the memory path is
unreachable from WanModel and hydra=True silently trains/infers as the
baseline (the bug that invalidated the first hydra retrain).

Tests:
  1. Reachability — hydra=True + grid_thw, called in the BLOCK's convention
     (encoder_hidden_states=hidden_states), must produce a DIFFERENT output
     than the standard self-attention path (grid_thw=None).
  2. hydra=False no-op — passing grid_thw to a non-hydra module changes nothing.
  3. Cross-attention safety — hydra=True with is_self_attention=False never
     enters the memory path (construction-time gate).

Run locally (CPU, no TPU):
  cd /home/spu9/maxdiffusion
  python -m pytest src/maxdiffusion/tests/wan_hydra_gate_test.py -v
"""

import os
import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from .. import pyconfig
from ..max_utils import create_device_mesh
from ..models.attention_flax import FlaxWanAttention

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Small but hydra-valid geometry: F=20 latent frames (>= the hardcoded
# window_size=5), compressed F'=10 (>= the hardcoded top_k=10), H=W=2.
HEADS = 2
DIM_HEAD = 64
DIM = HEADS * DIM_HEAD
GRID_THW = (20, 2, 2)
SEQ = GRID_THW[0] * GRID_THW[1] * GRID_THW[2]


def _make_mesh_and_config():
  pyconfig.initialize(
      [None, os.path.join(THIS_DIR, "..", "configs", "base_wan_2_1_v2v.yml")],
      unittest=True,
  )
  config = pyconfig.config
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  return config, mesh


class WanHydraGateTest(unittest.TestCase):

  def setUp(self):
    self.config, self.mesh = _make_mesh_and_config()
    self.x = jax.random.normal(jax.random.key(1), (1, SEQ, DIM), dtype=jnp.float32)
    # Identity RoPE (cos=1, sin=0) in the [1, 1, seq, head_dim//2] complex layout.
    self.rotary = jnp.ones((1, 1, SEQ, DIM_HEAD // 2), dtype=jnp.complex64)

  def _build(self, hydra, is_self_attention=True):
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      return FlaxWanAttention(
          rngs=nnx.Rngs(0),
          query_dim=DIM,
          heads=HEADS,
          dim_head=DIM_HEAD,
          attention_kernel="dot_product",
          mesh=self.mesh,
          dtype=jnp.float32,
          weights_dtype=jnp.float32,
          is_self_attention=is_self_attention,
          hydra=hydra,
      )

  def _block_call(self, attn, grid_thw):
    """Call exactly the way WanTransformerBlock calls attn1: encoder_hidden_states
    is the SAME (non-None) tensor as hidden_states."""
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      return attn(
          hidden_states=self.x,
          encoder_hidden_states=self.x,
          rotary_emb=self.rotary,
          grid_thw=grid_thw,
      )

  def test_hydra_path_reachable_via_block_call_convention(self):
    """hydra=True + grid_thw must route to the memory path (different output
    than the standard path) even though encoder_hidden_states is non-None."""
    attn = self._build(hydra=True)
    out_hydra = self._block_call(attn, grid_thw=GRID_THW)
    out_standard = self._block_call(attn, grid_thw=None)
    self.assertEqual(out_hydra.shape, out_standard.shape)
    max_diff = float(jnp.max(jnp.abs(out_hydra - out_standard)))
    self.assertGreater(
        max_diff,
        0.0,
        "hydra=True with grid_thw produced the SAME output as the standard "
        "self-attention path — the memory branch is unreachable from the "
        "block's call convention (encoder_hidden_states non-None).",
    )

  def test_hydra_off_is_noop(self):
    """hydra=False: grid_thw must be ignored entirely."""
    attn = self._build(hydra=False)
    out_with = self._block_call(attn, grid_thw=GRID_THW)
    out_without = self._block_call(attn, grid_thw=None)
    self.assertTrue(
        jnp.array_equal(out_with, out_without),
        "hydra=False output changed when grid_thw was passed.",
    )

  def test_chunked_retrieval_matches_full_vmap(self):
    """The frame-chunked retrieval (HBM fix) must equal the full-vmap path."""
    attn = self._build(hydra=True)
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      q, k, v, k_comp, v_comp, sim, _ = attn._hydra_memory_encode(self.x, self.rotary, GRID_THW)
      out_full = attn._dynamic_retrieval_attention(q, k, v, k_comp, v_comp, sim, GRID_THW, _frame_chunk=GRID_THW[0])
      out_chunked = attn._dynamic_retrieval_attention(q, k, v, k_comp, v_comp, sim, GRID_THW)
    max_diff = float(jnp.max(jnp.abs(out_full - out_chunked)))
    self.assertLess(max_diff, 1e-5, f"chunked retrieval diverges from full vmap: max|diff|={max_diff}")

  def test_cross_attention_never_enters_hydra(self):
    """Construction-time gate: is_self_attention=False disables hydra even if
    requested, so cross-attention can never take the memory path."""
    attn = self._build(hydra=True, is_self_attention=False)
    self.assertFalse(attn.hydra)
    out_with = self._block_call(attn, grid_thw=GRID_THW)
    out_without = self._block_call(attn, grid_thw=None)
    self.assertTrue(
        jnp.array_equal(out_with, out_without),
        "is_self_attention=False attention entered the hydra path.",
    )


if __name__ == "__main__":
  unittest.main()
