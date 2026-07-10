"""JAX side of the MemoryRetriever parity check (run on CPU).

Builds our nnx MemoryRetriever, loads the SAME Captain-Safari weights, runs on the
inputs dumped by cs_memory_ref_dump_torch.py, and compares to the PyTorch
reference output. Validates the port (esp. the 3D RoPE) numerically.

  CS_ST=<epoch-4.safetensors> python cs_memory_parity.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from flax import nnx

from maxdiffusion.models.wan.memory_retriever import (
    MemoryRetriever,
    load_cs_memory_retriever_weights,
)

ST = os.environ["CS_ST"]
PD = os.environ.get("PARITY_DIR", "/tmp/parity")

m = MemoryRetriever(dim=1024, num_heads=8, num_blocks=1, rngs=nnx.Rngs(0))
n = load_cs_memory_retriever_weights(m, ST)
print(f"[jax] loaded {n} CS params into nnx MemoryRetriever")

qp = np.load(os.path.join(PD, "qp.npy"))
kp = np.load(os.path.join(PD, "kp.npy"))
mem = np.load(os.path.join(PD, "mem.npy"))
ref = np.load(os.path.join(PD, "out_ref.npy"))

import jax.numpy as jnp  # noqa: E402

out = np.asarray(m(jnp.asarray(qp), jnp.asarray(kp), jnp.asarray(mem)))
diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
rel = diff.max() / (np.abs(ref).max() + 1e-8)
print(f"[jax] out {out.shape}  ref {ref.shape}")
print(f"[jax] max_abs_diff={diff.max():.3e}  mean_abs_diff={diff.mean():.3e}  ref|max|={np.abs(ref).max():.3e}  rel={rel:.3e}")
max_abs = float(diff.max())
finite = bool(np.isfinite(out).all() and np.isfinite(ref).all() and np.isfinite(max_abs))
print("PARITY:", "PASS (<1e-3)" if finite and max_abs < 1e-3 else ("CLOSE (<1e-2)" if finite and max_abs < 1e-2 else "MISMATCH"))
if not finite or max_abs >= 1e-2:
  raise SystemExit(1)
