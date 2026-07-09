# GATE: is the Wan2.1 VAE temporally causal at latent frame 0?
#
# If yes, VAE(video)[:, 0] == VAE(first frame alone)[:, 0], which means the
# Fun-Camera `y` conditioning (= VAE of the first frame) can be taken straight
# from the concat records' `latents[:, 0]` — no `latent_condition` field needed.
#
# Uses a REAL HM-World cond.mp4 first frame at the training resolution.
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/spu9/DiffSynth-Studio")
from diffsynth.models.wan_video_vae import WanVideoVAE  # noqa: E402
from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter  # noqa: E402

VAE_PTH = "/data2/spu9/wan21_fun_camera/ckpt/Wan2.1_VAE.pth"
SID = "00001_City_loc10_Complex_Follow_1"
VIDEO = f"/data-2u-2/spu/HM-World/{SID}/cond.mp4"
H, W = 480, 832
N_FRAMES = 13  # 4k+1


def read_frames(path, n):
  import av
  from PIL import Image
  out = []
  with av.open(path) as c:
    for f in c.decode(video=0):
      out.append(f.to_image().resize((W, H), Image.BICUBIC))
      if len(out) == n:
        break
  return out


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  vae = WanVideoVAE()
  sd = torch.load(VAE_PTH, map_location="cpu", weights_only=True)
  sd = WanVideoVAEStateDictConverter(sd)  # prefixes 'model.'
  missing, unexpected = vae.load_state_dict(sd, strict=False)
  print(f"load: {len(missing)} missing, {len(unexpected)} unexpected")
  assert len(unexpected) == 0, unexpected[:5]
  vae = vae.to(device).eval()

  frames = read_frames(VIDEO, N_FRAMES)
  arr = np.stack([np.asarray(f, np.float32) / 127.5 - 1.0 for f in frames])  # [F,H,W,C]
  video = torch.from_numpy(arr).permute(3, 0, 1, 2)  # [C,F,H,W]

  with torch.no_grad():
    lat_full = vae.encode([video.to(device)], device=device)[0]          # [16, F_lat, h, w]
    lat_first = vae.encode([video[:, :1].to(device)], device=device)[0]  # [16, 1, h, w]

  a = lat_full[:, :1].float().cpu().numpy()
  b = lat_first.float().cpu().numpy()
  print("full latents:", tuple(lat_full.shape), " first-frame latents:", tuple(lat_first.shape))
  diff = np.abs(a - b)
  denom = max(np.abs(b).mean(), 1e-8)
  print(f"mean|first-frame latent| = {denom:.6f}")
  print(f"max abs diff  = {diff.max():.3e}")
  print(f"mean abs diff = {diff.mean():.3e}")
  print(f"rel = {diff.mean() / denom:.3e}")
  # Contrast: latent frame 1 obviously must NOT match (sanity that the test can fail).
  if lat_full.shape[1] > 1:
    c = lat_full[:, 1:2].float().cpu().numpy()
    print(f"control (latent frame 1 vs first-frame) rel = {np.abs(c - b).mean() / denom:.3e}")
  print("CAUSAL" if diff.mean() / denom < 1e-3 else "NOT CAUSAL")


if __name__ == "__main__":
  main()
