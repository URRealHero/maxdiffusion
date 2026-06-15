"""Captain-Safari MEMORY-BRANCH reference (real demo poses), CPU/GPU torch.

Reproduces CS's exact memory pathway from model_fn_wan_video on the demo sample:
  raw camera matrices -> convert_to_local_coordinates -> extri_intri_to_pose_encoding
  -> dit.memory_retriever(target_pose_token, key_pose_token, memory) -> dit.memory_emb
This is the small part (retriever + 2-layer MLP, no 5B DiT), so it runs on CPU.

Imports CS's REAL functions/modules from the installed diffsynth (conda captain_safari).
Dumps inputs + target/key pose tokens + memory_context so the JAX port can be
compared. The full-DiT (LoRA + memory cross-attn -> noise pred) parity is a separate
GPU script.

Run (conda captain_safari env):
  CS_ST=<epoch-4.safetensors> CS_DATA=<.../captain_safari/data> \
  PARITY_DIR=/tmp/parity_mem python cs_memory_full_ref_torch.py
"""
import os
import numpy as np
import pandas as pd
import torch

from safetensors import safe_open
from diffsynth.models.wan_video_dit import MemoryRetriever
from diffsynth.models.wan_video_camera_controller import (
    convert_to_local_coordinates,
    extri_intri_to_pose_encoding,
)

ST = os.environ["CS_ST"]
DATA = os.environ.get("CS_DATA", "/home/spu9/Captain-Safari/captain_safari/data")
OUT = os.environ.get("PARITY_DIR", "/tmp/parity_mem")
os.makedirs(OUT, exist_ok=True)
NKEY = 4  # first 4 keyframes (matches retriever [B,4,9] key tokens)
HW = (256, 512)


def _load(rel):
  return np.load(os.path.join(DATA, rel), allow_pickle=True)


# --- demo sample (row 0) ---
row = pd.read_csv(os.path.join(DATA, "metadata.csv")).iloc[0]
mem_np = np.asarray(_load(row["memory"]), dtype=np.float32)            # (16,4,782,1024)
extr_key = np.asarray(_load(row["extrinsic_key"]), dtype=np.float32)   # (16,3,4)
intr_key = np.asarray(_load(row["intrinsic_key"]), dtype=np.float32)   # (16,3,3)
extr_q = np.asarray(_load(row["extrinsic_query"]), dtype=np.float32)   # (3,4)
intr_q = np.asarray(_load(row["intrinsic_query"]), dtype=np.float32)   # (3,3)
print(f"[ref] mem {mem_np.shape} extr_key {extr_key.shape} extr_q {extr_q.shape}")

# memory: first 4 keyframes -> flat [1, 4*4*782, 1024]
mem = torch.tensor(mem_np[:NKEY]).reshape(1, -1, 1024).float()
assert mem.shape[1] == NKEY * 4 * 782, mem.shape

# --- pose tokens, exactly as model_fn_wan_video ---
extr_key_t = torch.tensor(extr_key[:NKEY]).float()        # [4,3,4]
extr_q_t = torch.tensor(extr_q).float().unsqueeze(0)      # [1,3,4]
combined = torch.cat([extr_key_t, extr_q_t], dim=0)       # [5,3,4]
combined_local = convert_to_local_coordinates(combined)   # [5,3,4]
extr_key_local = combined_local[:NKEY].unsqueeze(0)       # [1,4,3,4]
extr_q_local = combined_local[NKEY:].squeeze(0)           # [3,4]

intr_key_t = torch.tensor(intr_key[:NKEY]).float().unsqueeze(0)  # [1,4,3,3]
intr_q_t = torch.tensor(intr_q).float()                          # [3,3]

target_pose_token = extri_intri_to_pose_encoding(
    extr_q_local.unsqueeze(0).unsqueeze(0), intr_q_t.unsqueeze(0).unsqueeze(0), image_size_hw=HW
)  # [1,1,9]
key_pose_token = extri_intri_to_pose_encoding(
    extr_key_local, intr_key_t, image_size_hw=HW
)  # [1,4,9]
print(f"[ref] target_pose_token {tuple(target_pose_token.shape)} key_pose_token {tuple(key_pose_token.shape)}")

# --- retriever + memory_emb (Sequential: Linear, SiLU, Linear) ---
m = MemoryRetriever(dim=1024, num_heads=8, num_blocks=1, rope_mode="3d").float().eval()
sd = {}
with safe_open(ST, framework="pt") as f:
  for k in f.keys():
    if k.startswith("memory_retriever."):
      sd[k[len("memory_retriever."):]] = f.get_tensor(k).float()
mr_missing, mr_unexpected = m.load_state_dict(sd, strict=False)
print(f"[ref] retriever: {len(sd)} tensors loaded; missing={len(mr_missing)} unexpected={len(mr_unexpected)}")

memory_emb = torch.nn.Sequential(
    torch.nn.Linear(1024, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072)
).float().eval()
emb_sd = {}
with safe_open(ST, framework="pt") as f:
  for k in f.keys():
    if k.startswith("memory_emb."):
      emb_sd[k[len("memory_emb."):]] = f.get_tensor(k).float()
memory_emb.load_state_dict(emb_sd)
print(f"[ref] memory_emb: {len(emb_sd)} tensors loaded")

with torch.no_grad():
  retrieved = m(target_pose_token.float(), key_pose_token.float(), mem)  # [1,3128,1024]
  memory_context = memory_emb(retrieved)                                 # [1,3128,3072]
print(f"[ref] retrieved {tuple(retrieved.shape)} memory_context {tuple(memory_context.shape)} "
      f"mean={memory_context.mean().item():.5f} std={memory_context.std().item():.5f}")

np.save(os.path.join(OUT, "mem.npy"), mem.numpy())
np.save(os.path.join(OUT, "target_pose_token.npy"), target_pose_token.float().numpy())
np.save(os.path.join(OUT, "key_pose_token.npy"), key_pose_token.float().numpy())
np.save(os.path.join(OUT, "retrieved.npy"), retrieved.numpy())
np.save(os.path.join(OUT, "memory_context_ref.npy"), memory_context.numpy())
# also dump the raw inputs so the JAX side encodes poses from identical matrices
np.save(os.path.join(OUT, "extr_key4.npy"), extr_key[:NKEY])
np.save(os.path.join(OUT, "intr_key4.npy"), intr_key[:NKEY])
np.save(os.path.join(OUT, "extr_query.npy"), extr_q)
np.save(os.path.join(OUT, "intr_query.npy"), intr_q)
print(f"[ref] dumped to {OUT}")
