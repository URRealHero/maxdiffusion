"""Parity probe: tokamax_ring vs flash vs dense reference, under a ctx-sharded mesh.

Motivation (2026-07-04): fresh-init v2v training with attention=tokamax_ring showed
~2x the loss of attention=flash on identical data/seed -> the ring path is suspected
of a semantic bug for our shapes; prime suspect is padding/segment-id handling (ring
pads each sequence shard to a block multiple; if pad tokens leak into the softmax
on any ring hop, outputs are systematically diluted).

Runs on a ctx>1 mesh (e.g. debug-8, fsdp=2 x context=4):
  TPU_NAME=tpu-v6e-8-debug: python src/maxdiffusion/probe_ring_parity.py \
      src/maxdiffusion/configs/base_wan_2_1_v2v.yml run_name=ringparity \
      ici_data_parallelism=1 ici_fsdp_parallelism=2 ici_context_parallelism=4 \
      ici_tensor_parallelism=1 skip_jax_distributed_system=True

For each seq length in PROBE_SEQS (alignable 16384 vs v2v-like non-alignable 15600
and the real 62400), compares:
  flash-vs-ref, ring-vs-ref, ring-vs-flash  (max|diff| and mean|diff| over outputs)
A padding bug shows as: ring matches at 16384 but diverges at 15600/62400.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.max_utils import create_device_mesh, get_flash_block_sizes
from maxdiffusion.models.attention_flax import _tpu_flash_attention

HEADS = 12
DIM_HEAD = 128
INNER = HEADS * DIM_HEAD
BATCH = 2
# (label, seq_len): 16384 = 4 shards x 4096, block-alignable at 512/1024/2048.
# 15600 = the real v2v shard size as a total (not alignable); 62400 = real v2v seq.
PROBE_SEQS = [("alignable-16384", 16384), ("v2v-like-15600", 15600), ("real-62400", 62400)]

AXIS_Q = ("activation_batch", "activation_self_attn_heads", "activation_self_attn_q_length", "activation_kv")
AXIS_KV = ("activation_batch", "activation_self_attn_heads", "activation_kv_length", "activation_kv")


def dense_reference(q, k, v):
  """fp32 full-softmax attention on unsharded arrays. [B, S, H*D] layout in/out."""
  b, s, _ = q.shape
  qh = q.reshape(b, s, HEADS, DIM_HEAD).transpose(0, 2, 1, 3).astype(jnp.float32)
  kh = k.reshape(b, s, HEADS, DIM_HEAD).transpose(0, 2, 1, 3).astype(jnp.float32)
  vh = v.reshape(b, s, HEADS, DIM_HEAD).transpose(0, 2, 1, 3).astype(jnp.float32)
  scores = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) / jnp.sqrt(DIM_HEAD)
  probs = jax.nn.softmax(scores, axis=-1)
  out = jnp.einsum("bhqk,bhkd->bhqd", probs, vh)
  return out.transpose(0, 2, 1, 3).reshape(b, s, INNER)


def run_kernel(kernel, q, k, v, mesh, config):
  block_sizes = get_flash_block_sizes(config)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    out = _tpu_flash_attention(
        q.astype(jnp.bfloat16),
        k.astype(jnp.bfloat16),
        v.astype(jnp.bfloat16),
        heads=HEADS,
        mesh=mesh,
        axis_names_q=AXIS_Q,
        axis_names_kv=AXIS_KV,
        flash_block_sizes=block_sizes,
        dtype=jnp.bfloat16,
        attention_kernel=kernel,
        mask_padding_tokens=bool(getattr(config, "mask_padding_tokens", True)),
    )
  return np.asarray(jax.device_get(out)).astype(np.float32)


def main(argv):
  pyconfig.initialize(argv, validate_training=False)
  config = pyconfig.config
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  max_logging.log(f"probe mesh: {dict(zip(config.mesh_axes, mesh.devices.shape))}")

  for label, seq in PROBE_SEQS:
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    # bf16-representable inputs so the dense fp32 reference sees identical values.
    q = jax.random.normal(kq, (BATCH, seq, INNER), dtype=jnp.bfloat16).astype(jnp.float32)
    k = jax.random.normal(kk, (BATCH, seq, INNER), dtype=jnp.bfloat16).astype(jnp.float32)
    v = jax.random.normal(kv, (BATCH, seq, INNER), dtype=jnp.bfloat16).astype(jnp.float32)

    ref = np.asarray(dense_reference(q, k, v)).astype(np.float32)
    results = {}
    for kernel in ("flash", "tokamax_ring"):
      try:
        results[kernel] = run_kernel(kernel, q, k, v, mesh, config)
      except Exception as e:  # OOM / unsupported shape: report and continue
        max_logging.log(f"[{label}] {kernel} FAILED: {type(e).__name__}: {str(e)[:200]}")

    def stats(a, b):
      d = np.abs(a - b)
      return f"max|d|={d.max():.4f} mean|d|={d.mean():.6f}"

    for kernel, out in results.items():
      max_logging.log(f"[{label}] {kernel:13s} vs dense-ref : {stats(out, ref)}")
    if len(results) == 2:
      max_logging.log(f"[{label}] ring vs flash          : {stats(results['tokamax_ring'], results['flash'])}")
  max_logging.log("PROBE DONE")


if __name__ == "__main__":
  main(sys.argv)
