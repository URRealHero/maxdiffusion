"""VAE round-trip tail test (Phase H noisy-tail isolation).

Encode a real 121-frame video with our VAE, decode it back, and save the reconstruction.
If the RECONSTRUCTION's last frames are noisy (while the original isn't), our VAE *decode*
corrupts the tail -> that's the noisy-tail bug (memory-independent, matches the symptom).
If the reconstruction is clean to the end, the decode is fine and the noise is in the
denoising/sampler.

Single-host on 8-debug. Loads the demo video, takes NUM_FRAMES frames at HEIGHT x WIDTH.
Saves original + reconstruction mp4s; also prints per-frame Laplacian-variance (sharpness).
"""
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

from absl import app
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.utils import export_to_video

VIDEO = os.environ.get("RT_VIDEO", "/tmp/cs_mem_assets/input_video.mp4")
OUT = os.environ.get("RT_OUT", "/tmp")


def _load_video(path, n, h, w):
  import imageio.v2 as imageio
  rd = imageio.get_reader(path)
  frames = []
  for i, fr in enumerate(rd):
    frames.append(fr)
    if len(frames) >= n:
      break
  rd.close()
  import cv2
  frames = [cv2.resize(f, (w, h)) for f in frames]
  while len(frames) < n:  # pad by repeating last
    frames.append(frames[-1])
  v = np.stack(frames[:n], 0).astype(np.float32) / 127.5 - 1.0  # [n,h,w,3] in [-1,1]
  v = v.transpose(3, 0, 1, 2)[None]  # [1,3,n,h,w]
  return v


def _lap_var_per_frame(video):  # video: [T,H,W,3] uint8/float
  import cv2
  out = []
  for fr in video:
    g = cv2.cvtColor(np.asarray(fr).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    out.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
  return out


def run(config):
  dtype = jnp.float32
  pipeline, _, _ = WanCheckpointer2_2_FunCamera(config=config).load_checkpoint()
  n, h, w = config.num_frames, config.height, config.width
  video = _load_video(VIDEO, n, h, w)
  max_logging.log(f"[vae-rt] loaded video {video.shape} from {VIDEO}")
  vc = jnp.asarray(video, dtype=getattr(pipeline.vae, "dtype", jnp.float32))

  from flax.linen import partitioning as nn_partitioning
  with pipeline.vae_mesh, nn_partitioning.axis_rules(pipeline.vae_logical_axis_rules):
    enc = pipeline.vae.encode(vc, pipeline.vae_cache)[0]
    latents = enc.mode() if hasattr(enc, "mode") else enc
    latents.block_until_ready()
  max_logging.log(f"[vae-rt] latents {latents.shape} nan={bool(jnp.isnan(latents).any())} "
                  f"std={float(jnp.nan_to_num(latents).std()):.4f}")
  recon = pipeline._decode_latents_to_video(latents)  # [B,T,H,W,3] postprocessed
  recon = np.asarray(recon)[0]
  orig = ((video[0].transpose(1, 2, 3, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # [T,H,W,3]
  max_logging.log(f"[vae-rt] recon {recon.shape}  orig {orig.shape}")

  lv_o = _lap_var_per_frame(orig)
  lv_r = _lap_var_per_frame((recon * 255).astype(np.uint8) if recon.max() <= 1.01 else recon.astype(np.uint8))
  T = len(lv_r)
  max_logging.log("[vae-rt] per-frame sharpness (Laplacian var) orig vs recon (tail = bug if recon collapses):")
  for t in list(range(0, T, max(1, T // 10))) + [T - 4, T - 3, T - 2, T - 1]:
    if 0 <= t < T:
      max_logging.log(f"    frame {t:3d}: orig={lv_o[t]:9.1f}  recon={lv_r[t]:9.1f}  ratio={lv_r[t]/(lv_o[t]+1e-6):.3f}")
  head = np.mean(lv_r[: T // 2]); tail = np.mean(lv_r[-8:])
  max_logging.log(f"[vae-rt] recon sharpness head(mean first half)={head:.1f} tail(mean last 8)={tail:.1f} tail/head={tail/(head+1e-6):.3f}")

  export_to_video([(recon * 255).astype(np.uint8) if recon.max() <= 1.01 else recon.astype(np.uint8)][0] if False else recon,
                  os.path.join(OUT, "vae_rt_recon.mp4"), fps=config.fps)
  export_to_video(orig, os.path.join(OUT, "vae_rt_orig.mp4"), fps=config.fps)
  max_logging.log(f"[vae-rt] saved {OUT}/vae_rt_recon.mp4 + vae_rt_orig.mp4")


def main(argv):
  pyconfig.initialize(argv, validate_training=False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
