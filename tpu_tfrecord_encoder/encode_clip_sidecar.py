# Encode CLIP image features for Wan2.1-Fun-Control-Camera training into
# sidecar TFRecords keyed by sample_id.
#
# For each HM-World sample, the conditioning image is the CONCAT video's first
# frame = cond.mp4 frame 0 (the y-ref of encode_concat_camera.py records).
# Faithful to the official DiffSynth path: frame -> PIL resize (W,H) bicubic ->
# [-1,1] tensor -> WanImageEncoder.encode_image (bicubic to CLIP 224 + CLIP
# normalize + ViT-H use_31_block) -> [257,1280].
#
# Output records: {sample_id: bytes, clip_feature_fp16: serialized fp16 tensor}
# (fp16 halves the upload; the merge step casts back to fp32 for the trainer).
#
# Env: needs torch+CUDA, tensorflow (writer only), av, PIL, and DiffSynth-Studio
# importable (--diffsynth-path). Multi-GPU: run one process per GPU with
# --worker-id/--worker-count and CUDA_VISIBLE_DEVICES.
#
# Example (one of 4 workers):
#   CUDA_VISIBLE_DEVICES=0 python encode_clip_sidecar.py \
#     --metadata-dir /data2/spu9/wan21_fun_camera/main_metadata \
#     --hmworld-root /data-2u-2/spu/HM-World \
#     --clip-pth /data2/spu9/wan21_fun_camera/ckpt/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#     --output-dir /data2/spu9/wan21_fun_camera/clip_sidecar \
#     --worker-id 0 --worker-count 4

import argparse
import glob
import json
import os
import sys

import numpy as np


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--metadata-dir", required=True,
                 help="dir of metadata_host_*.jsonl from the main encode (source of sample_ids)")
  p.add_argument("--hmworld-root", default="/data-2u-2/spu/HM-World")
  p.add_argument("--clip-pth", required=True)
  p.add_argument("--output-dir", required=True)
  p.add_argument("--diffsynth-path", default="/home/spu9/DiffSynth-Studio")
  p.add_argument("--width", type=int, default=832)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--batch-size", type=int, default=32)
  p.add_argument("--records-per-shard", type=int, default=512)
  p.add_argument("--worker-id", type=int, default=0)
  p.add_argument("--worker-count", type=int, default=1)
  p.add_argument("--limit", type=int, default=0)
  return p.parse_args()


def load_sample_ids(metadata_dir):
  sids = set()
  for path in sorted(glob.glob(os.path.join(metadata_dir, "metadata_host_*.jsonl"))):
    with open(path) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        sid = json.loads(line).get("sample_id")
        if sid is not None:
          sids.add(str(sid))
  return sorted(sids)


def read_first_frame(path):
  import av
  from PIL import Image
  with av.open(path) as container:
    for frame in container.decode(video=0):
      return frame.to_image()
  raise ValueError(f"no frames decoded from {path}")


def main():
  args = parse_args()
  sys.path.insert(0, args.diffsynth_path)
  import torch
  import tensorflow as tf

  tf.config.set_visible_devices([], "GPU")  # tf is only the tfrecord writer here
  from diffsynth.models.wan_video_image_encoder import WanImageEncoder
  from PIL import Image

  os.makedirs(args.output_dir, exist_ok=True)
  manifest_path = os.path.join(args.output_dir, f"manifest_w{args.worker_id:02d}.jsonl")
  done = set()
  if os.path.exists(manifest_path):
    with open(manifest_path) as f:
      done = {json.loads(l)["sample_id"] for l in f if l.strip()}
    print(f"[w{args.worker_id}] resume: {len(done)} already done")

  sids = load_sample_ids(args.metadata_dir)
  print(f"[w{args.worker_id}] {len(sids)} sample_ids total")
  sids = sids[args.worker_id::args.worker_count]
  sids = [s for s in sids if s not in done]
  if args.limit:
    sids = sids[: args.limit]
  print(f"[w{args.worker_id}] {len(sids)} to encode")
  if not sids:
    return

  device = "cuda" if torch.cuda.is_available() else "cpu"
  encoder = WanImageEncoder()
  sd = torch.load(args.clip_pth, map_location="cpu", weights_only=True)
  sd = {("model." + k): v for k, v in sd.items() if not k.startswith("textual.")}
  missing, unexpected = encoder.load_state_dict(sd, strict=False)
  assert not unexpected, unexpected[:5]
  missing_real = [m for m in missing if "textual" not in m]
  assert not missing_real, missing_real[:5]
  encoder = encoder.to(device).eval()

  def to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [H,W,C] in [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]

  shard_idx, written_in_shard, writer = 0, 0, None

  def next_writer():
    nonlocal shard_idx, written_in_shard, writer
    if writer is not None:
      writer.close()
    path = os.path.join(args.output_dir, f"clip_w{args.worker_id:02d}_file_{shard_idx:05d}.tfrec")
    writer = tf.io.TFRecordWriter(path)
    shard_idx += 1
    written_in_shard = 0

  # start after any existing shards of this worker (never overwrite)
  existing = glob.glob(os.path.join(args.output_dir, f"clip_w{args.worker_id:02d}_file_*.tfrec"))
  shard_idx = len(existing)
  next_writer()

  manifest = open(manifest_path, "a")
  n_done, n_err = 0, 0
  batch_sids, batch_imgs = [], []

  def flush():
    nonlocal n_done, written_in_shard, batch_sids, batch_imgs
    if not batch_sids:
      return
    with torch.no_grad():
      feats = encoder.encode_image([t.to(device) for t in batch_imgs])  # [B,257,1280]
    feats = feats.detach().float().cpu().numpy().astype(np.float16)
    for sid, feat in zip(batch_sids, feats):
      feature = {
          "sample_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sid.encode()])),
          "clip_feature_fp16": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(feat).numpy()])
          ),
      }
      writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
      manifest.write(json.dumps({"sample_id": sid}) + "\n")
      n_done += 1
      written_in_shard += 1
      if written_in_shard >= args.records_per_shard:
        next_writer()
    manifest.flush()
    batch_sids, batch_imgs = [], []

  for i, sid in enumerate(sids):
    path = os.path.join(args.hmworld_root, sid, "cond.mp4")
    try:
      img = read_first_frame(path).resize((args.width, args.height), Image.BICUBIC)
    except Exception as e:  # noqa: BLE001 - log and continue; missing clips reported at end
      print(f"[w{args.worker_id}] ERROR {sid}: {e}")
      n_err += 1
      continue
    batch_sids.append(sid)
    batch_imgs.append(to_tensor(img))
    if len(batch_sids) >= args.batch_size:
      flush()
    if (i + 1) % 500 == 0:
      print(f"[w{args.worker_id}] {i + 1}/{len(sids)} (written {n_done}, errors {n_err})", flush=True)
  flush()
  writer.close()
  manifest.close()
  print(f"[w{args.worker_id}] DONE: wrote {n_done}, errors {n_err}")
  if n_err:
    print(f"[w{args.worker_id}] WARNING: {n_err} samples failed — rerun to retry or investigate")


if __name__ == "__main__":
  main()
