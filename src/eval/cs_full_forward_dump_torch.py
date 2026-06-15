"""CS FULL-FORWARD reference dump for stage-by-stage TPU parity (run on GPU, conda captain_safari).

Sets up the exact Captain-Safari validate pipeline (PAI base + LoRA + memory), then
monkeypatches model_fn_wan_video to capture, on the FIRST denoise step (fixed seed=1),
every model input + the model output (noise_pred). Dumps them as .npy so the JAX/TPU
side can feed the IDENTICAL inputs and diff each stage:
  latents (noised, step 0), timestep, context (prompt emb), y, control_camera_latents_input,
  memory, intrinsic/extrinsic query+key, and noise_pred.

This isolates WHERE TPU diverges from CS: if our noise_pred matches given the same inputs,
the gap is the sampler/decode; if an input differs, that conditioning stage is the culprit.

Run (GPU box, conda captain_safari env, from the captain_safari dir so ./data + ./models resolve):
  CS_MODEL=./models/train/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors \
  CS_DATA=./data PARITY_DIR=/tmp/cs_parity \
  python <path>/cs_full_forward_dump_torch.py
Then: gsutil -m cp /tmp/cs_parity/*.npy gs://data_us_central1_a/maxdiffusion/wan/cs_memory/parity/
"""
import os
import numpy as np
import pandas as pd
import torch

from diffsynth import VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import diffsynth.pipelines.wan_video_new as wv

MODEL = os.environ["CS_MODEL"]
DATA = os.environ.get("CS_DATA", "./data")
OUT = os.environ.get("PARITY_DIR", "/tmp/cs_parity")
GPU = os.environ.get("CS_GPU", "cuda:0")
os.makedirs(OUT, exist_ok=True)


def _save(name, t):
  if t is None:
    print(f"  [skip] {name} is None"); return
  a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
  np.save(os.path.join(OUT, f"{name}.npy"), a)
  print(f"  saved {name}: {tuple(a.shape)} {a.dtype}")


# ---- build pipeline exactly like validate ----
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device=GPU,
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-5B-Control-Camera", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.dit.use_memory_retrieval = True
pipe.dit.use_memory_cross_attn = True
pipe.load_lora(pipe.dit, MODEL, alpha=1)
pipe.load_memory(pipe.dit, MODEL)
pipe.load_cross_attn(pipe.dit, MODEL)
pipe.load_memory_retriever_from_dit(pipe.dit, MODEL)
pipe.enable_vram_management()
if hasattr(pipe.dit, "memory_retriever"):
  pipe.dit.memory_retriever.to(device=GPU, dtype=torch.bfloat16)

# ---- demo sample (row 0) ----
row = pd.read_csv(os.path.join(DATA, "metadata.csv")).iloc[0]
ld = lambda c: np.load(os.path.join(DATA, str(row[c])), allow_pickle=True)
video_name = str(row["video"]).split("/")[-1]
memory = torch.tensor(ld("memory"), dtype=torch.bfloat16).unsqueeze(0)
intr_q = torch.tensor(ld("intrinsic_query"), dtype=torch.bfloat16).unsqueeze(0)
extr_q = torch.tensor(ld("extrinsic_query"), dtype=torch.bfloat16).unsqueeze(0)
intr_k = torch.tensor(ld("intrinsic_key"), dtype=torch.bfloat16).unsqueeze(0)
extr_k = torch.tensor(ld("extrinsic_key"), dtype=torch.bfloat16).unsqueeze(0)
intr_c = torch.tensor(ld("intrinsic_clip"), dtype=torch.bfloat16).unsqueeze(0)
extr_c = torch.tensor(ld("extrinsic_clip"), dtype=torch.bfloat16).unsqueeze(0)
input_image = VideoData(f"{DATA}/videos/{video_name}", height=704, width=1280)[0]

# ---- capture the FIRST model_fn call ----
captured = {}
orig = wv.model_fn_wan_video
def patched(*args, **kw):
  out = orig(*args, **kw)
  if not captured:
    for k in ("latents", "timestep", "context", "y", "memory", "control_camera_latents_input"):
      captured[k] = kw.get(k)
    captured["noise_pred"] = out
    captured["_done"] = True
    print("[dump] captured first model_fn call")
    for k, v in captured.items():
      if torch.is_tensor(v): _save(k, v)
  return out
wv.model_fn_wan_video = patched

NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
print("[dump] running 1-step generation to trigger model_fn capture...")
_ = pipe(
    prompt=str(row["prompt"]), negative_prompt=NEG,
    memory=memory, intrinsic_query=intr_q, extrinsic_query=extr_q,
    intrinsic_key=intr_k, extrinsic_key=extr_k, intrinsic_clip=intr_c, extrinsic_clip=extr_c,
    input_image=input_image, height=704, width=1280, num_frames=121,
    seed=1, tiled=True, num_inference_steps=2,
)
# also dump the raw camera matrices so the JAX side encodes poses identically
for c in ("intrinsic_query", "extrinsic_query", "intrinsic_key", "extrinsic_key"):
  _save(f"raw_{c}", torch.tensor(ld(c), dtype=torch.float32))
print(f"[dump] DONE -> {OUT}")
