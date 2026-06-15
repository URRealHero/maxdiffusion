"""Block-by-block parity vs CS: feed CS's exact block-stack input through OUR DiT blocks
one at a time and diff each block's output against CS's dumped block_NN.

Two modes per block i (isolated, so errors don't accumulate):
  input to our block i = CS's block_(i-1) output  (block_00 input = blkin_x)
  compare our output -> CS block_i
The first block whose output diverges / goes NaN is the culprit.

Runs in float32 (avoid bf16 NaN masking the comparison). scan_layers=False so blocks are a
plain list we can call individually. PARITY_DIR holds the CS dumps (blkin_*, block_NN).
"""
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from absl import app

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera

D = os.environ.get("PARITY_DIR", "/tmp/cs_parity")


def _load(n):
  p = os.path.join(D, n)
  return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def _cmp(name, ours, ref):
  ours = np.asarray(ours, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
  nan = bool(np.isnan(ours).any())
  d = np.abs(ours - ref)
  denom = np.abs(ref).max() + 1e-8
  cos = float((ours.flatten() @ ref.flatten()) / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-8))
  max_logging.log(f"  {name}: nan={nan} max_abs={d.max():.4e} mean={d.mean():.4e} "
                  f"rel={d.max()/denom:.4e} cos={cos:.5f}  (our std={ours.std():.3f})")
  return nan, cos


def run(config):
  # bf16 matches real generation (and halves the naive memory-attn matrix so it fits HBM).
  # CS block dumps are fp16; feeding CS's exact block input tests real per-block behavior.
  dtype = jnp.bfloat16
  pipeline, _, _ = WanCheckpointer2_2_FunCamera(config=config).load_checkpoint()
  t = pipeline.transformer
  blocks = list(t.blocks)
  max_logging.log(f"[blockparity] {len(blocks)} blocks (scan_layers={config.scan_layers})")

  x0 = _load("blkin_x.npy")
  context = jnp.asarray(_load("blkin_context.npy"), dtype=dtype)
  temb = jnp.asarray(_load("blkin_t_mod.npy"), dtype=dtype)
  mc_np = _load("blkin_memory_context.npy")
  memory_context = jnp.asarray(mc_np, dtype=dtype) if mc_np is not None else None
  max_logging.log(f"[blockparity] blkin_x={None if x0 is None else x0.shape} context={context.shape} "
                  f"temb={temb.shape} memory_context={None if mc_np is None else mc_np.shape}")

  # rope: recompute ours from the dumped block input seq layout (same model -> should match CS)
  # x0 is post-patchify [B, seq, dim]; our rope wants the 5D latent. Reconstruct via the dump's
  # latent dims (31,44,80) + rope_channels 100 (48 latents + 52 y), matching run_inference.
  dummy = jnp.zeros((1, 31, 44, 80, 100))
  rotary = t.rope(dummy)

  # jit the block forward so XLA FUSES the naive memory-attn softmax (else the eager
  # 27280x3128 fp32 matrix = 8GB OOMs; the real jitted/scan generation fuses it).
  @nnx.jit
  def _fwd(block, h, ctx, tb, rot, mc):
    o = block(h, ctx, tb, rot, memory_context=mc)
    return o[0] if isinstance(o, tuple) else o

  cur = jnp.asarray(x0, dtype=dtype)
  with pipeline.mesh:
    for i, blk in enumerate(blocks):
      out = _fwd(blk, cur, context, temb, rotary, memory_context)
      cs_i = _load(f"block_{i:02d}.npy")
      if cs_i is None:
        max_logging.log(f"  block {i:02d}: (no CS dump)")
      else:
        nan, cos = _cmp(f"block_{i:02d}", out, cs_i)
        if nan or cos < 0.9:
          max_logging.log(f"  >>> FIRST DIVERGENCE at block {i:02d} (nan={nan} cos={cos:.4f}) <<<")
      # isolated: next block gets CS's block_i output (so errors don't accumulate)
      cur = jnp.asarray(cs_i, dtype=dtype) if cs_i is not None else out


def main(argv):
  pyconfig.initialize(argv, validate_training=False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
