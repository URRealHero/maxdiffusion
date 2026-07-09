# GATE: does the POSITIONALLY-recovered sample_id actually match the record's contents?
#
# merge_clip_sidecar.py assigns sample_id from metadata line i -> record i%128.
# If that mapping were off (by a record, a shard, or a host), the records would
# silently train against the wrong camera/caption/CLIP. Comparing clip_feature
# proves nothing (we attached it BY that sid). The independent check is:
#
#     record.latents[:, 0]  ==  VAE(first frame of HM-World/<sid>/cond.mp4)
#
# because `latents` came from the ORIGINAL June encode and knows nothing about
# our sid assignment. A negative control (a different sid's video) must fail.
import sys

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

tf.config.set_visible_devices([], "GPU")
sys.path.insert(0, "/home/spu9/DiffSynth-Studio")
from diffsynth.models.wan_video_vae import WanVideoVAE  # noqa: E402
from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter  # noqa: E402

SHARD = "/data2/spu9/wan21_fun_camera/merged_shard0.tfrec"
VAE_PTH = "/data2/spu9/wan21_fun_camera/ckpt/Wan2.1_VAE.pth"
ROOT = "/data-2u-2/spu/HM-World"
H, W = 480, 832
CHECK_IDX = [0, 1, 63, 127]  # first, second, middle, last record of the shard


def first_frame_latent(vae, sid, device):
  import av
  with av.open(f"{ROOT}/{sid}/cond.mp4") as c:
    for f in c.decode(video=0):
      img = f.to_image().resize((W, H), Image.BICUBIC)
      break
  arr = np.asarray(img, np.float32) / 127.5 - 1.0
  v = torch.from_numpy(arr).permute(2, 0, 1)[:, None]  # [C,1,H,W]
  with torch.no_grad():
    return vae.encode([v.to(device)], device=device)[0][:, 0].float().cpu().numpy()


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  vae = WanVideoVAE()
  vae.load_state_dict(WanVideoVAEStateDictConverter(torch.load(VAE_PTH, map_location="cpu", weights_only=True)),
                      strict=False)
  vae = vae.to(device).eval()

  recs = []
  for i, raw in enumerate(tf.data.TFRecordDataset(SHARD)):
    if i in CHECK_IDX:
      ex = tf.train.Example.FromString(raw.numpy())
      sid = ex.features.feature["sample_id"].bytes_list.value[0].decode()
      lat = tf.io.parse_tensor(ex.features.feature["latents"].bytes_list.value[0], tf.float32).numpy()
      recs.append((i, sid, lat[:, 0]))
    if i > max(CHECK_IDX):
      break

  print(f"{'idx':>4} {'sample_id':44s} {'rel_err(match)':>14} {'rel_err(control)':>17}  verdict")
  ok = True
  for j, (i, sid, lat0) in enumerate(recs):
    ref = first_frame_latent(vae, sid, device)
    rel = np.abs(lat0 - ref).mean() / max(np.abs(ref).mean(), 1e-8)
    # negative control: a DIFFERENT record's sid
    other_sid = recs[(j + 1) % len(recs)][1]
    ref_o = first_frame_latent(vae, other_sid, device)
    rel_o = np.abs(lat0 - ref_o).mean() / max(np.abs(ref_o).mean(), 1e-8)
    good = rel < 0.05 and rel_o > 0.3
    ok &= good
    print(f"{i:>4} {sid:44s} {rel:>14.3e} {rel_o:>17.3e}  {'OK' if good else 'MISMATCH'}")
  print("SID ALIGNMENT", "VERIFIED" if ok else "BROKEN")


if __name__ == "__main__":
  main()
