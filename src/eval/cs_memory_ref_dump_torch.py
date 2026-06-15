"""PyTorch reference for the MemoryRetriever parity check (run on a CPU torch env).

Uses the REAL Captain-Safari MemoryRetriever (imported from a tiny stub package
'csmem' that wraps the actual wan_video_dit.py; flash_attention falls back to
torch SDPA on CPU). Runs in float32 on real memory features + seeded-random pose
tokens, and dumps inputs + output so the JAX port can be compared bit-for-bit.

Setup (done by the launcher): /tmp/csmem_pkg/csmem/{__init__.py, wan_video_dit.py,
utils.py(stub), wan_video_camera_controller.py(stub)}.

  CS_ST=<epoch-4.safetensors> CS_MEM=<key memory .npy> python cs_memory_ref_dump_torch.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/csmem_pkg")
from csmem.wan_video_dit import MemoryRetriever  # noqa: E402
from safetensors import safe_open  # noqa: E402

ST = os.environ["CS_ST"]
MEM = os.environ["CS_MEM"]
OUT = os.environ.get("PARITY_DIR", "/tmp/parity")
os.makedirs(OUT, exist_ok=True)

# 1024-dim, 8 heads, single retrieval block (matches the checkpoint).
m = MemoryRetriever(dim=1024, num_heads=8, num_blocks=1, rope_mode="3d").float().eval()
sd = {}
with safe_open(ST, framework="pt") as f:
  for k in f.keys():
    if k.startswith("memory_retriever."):
      sd[k[len("memory_retriever."):]] = f.get_tensor(k).float()
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"[ref] loaded memory_retriever: {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
if missing:
  print("  missing[:5]:", missing[:5])

# real memory features: take the first 4 keyframes -> flat [1, 4*3128, 1024].
# seeded-random poses (identical on both sides -> a clean architecture parity).
mem_np = np.asarray(np.load(MEM, allow_pickle=True), dtype=np.float32)
print(f"[ref] raw memory npy shape: {mem_np.shape}")
mem = torch.tensor(mem_np[:4]).reshape(1, -1, 1024)  # first 4 frames
assert mem.shape[1] == 4 * 4 * 782, f"expected 4*4*782={4*4*782} memory tokens, got {mem.shape[1]}"
rng = np.random.RandomState(0)
qp = torch.tensor(rng.randn(1, 1, 9).astype(np.float32))
kp = torch.tensor(rng.randn(1, 4, 9).astype(np.float32))

with torch.no_grad():
  out = m(qp, kp, mem)  # [1, 3128, 1024]
print(f"[ref] out {tuple(out.shape)}  mean={out.mean().item():.5f}  std={out.std().item():.5f}")

np.save(os.path.join(OUT, "qp.npy"), qp.numpy())
np.save(os.path.join(OUT, "kp.npy"), kp.numpy())
np.save(os.path.join(OUT, "mem.npy"), mem.numpy())
np.save(os.path.join(OUT, "out_ref.npy"), out.numpy())
print(f"[ref] dumped inputs + reference output to {OUT}")
