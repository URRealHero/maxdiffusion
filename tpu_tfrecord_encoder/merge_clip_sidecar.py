# Merge CLIP sidecar TFRecords (encode_clip_sidecar.py) into the main
# concat+camera records, producing the *_full_clip dataset the
# Wan2.1-Fun-camera trainer reads.
#
# For every record in every main shard: look up its sample_id in the sidecar
# set, append `clip_feature` (fp32 serialized tensor, cast from the sidecar's
# fp16), and write the record to an identically-named shard in --output-dir.
# Shard-level resume: an output shard whose record count matches its input is
# skipped. Any record whose sample_id has no sidecar entry aborts the merge
# (unless --allow-missing, which drops the record and logs it).
#
# IO: paths may be local or gs:// (tf.io.gfile). Run where GCS bandwidth is
# good (a TPU pod host CPU is ideal: read+write stay inside GCP; needs only
# the standard maxdiffusion venv, no TPU devices touched).
#
# Example:
#   python merge_clip_sidecar.py \
#     --main-dir gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full \
#     --sidecar-dir /data2/spu9/wan21_fun_camera/clip_sidecar \
#     --output-dir gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full_clip

import argparse
import os

import numpy as np
import tensorflow as tf


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--main-dir", required=True)
  p.add_argument("--sidecar-dir", required=True)
  p.add_argument("--output-dir", required=True)
  p.add_argument("--allow-missing", action="store_true")
  p.add_argument("--shard-start", type=int, default=0)
  p.add_argument("--shard-end", type=int, default=0, help="0 = all")
  return p.parse_args()


def load_sidecars(sidecar_dir):
  """sample_id -> raw fp16 serialized tensor bytes (decoded lazily at write)."""
  shards = sorted(tf.io.gfile.glob(os.path.join(sidecar_dir, "clip_*_file_*.tfrec")))
  assert shards, f"no sidecar shards under {sidecar_dir}"
  table = {}
  for i, shard in enumerate(shards):
    for raw in tf.data.TFRecordDataset(shard):
      ex = tf.train.Example.FromString(raw.numpy())
      sid = ex.features.feature["sample_id"].bytes_list.value[0].decode()
      table[sid] = ex.features.feature["clip_feature_fp16"].bytes_list.value[0]
    if (i + 1) % 20 == 0:
      print(f"sidecars: {i + 1}/{len(shards)} shards, {len(table)} features", flush=True)
  print(f"sidecars loaded: {len(table)} features from {len(shards)} shards")
  return table


def count_records(path):
  n = 0
  for _ in tf.data.TFRecordDataset(path):
    n += 1
  return n


def main():
  args = parse_args()
  table = load_sidecars(args.sidecar_dir)

  main_shards = sorted(
      s for s in tf.io.gfile.glob(os.path.join(args.main_dir, "*.tfrec"))
  )
  assert main_shards, f"no main shards under {args.main_dir}"
  if args.shard_end:
    main_shards = main_shards[args.shard_start:args.shard_end]
  else:
    main_shards = main_shards[args.shard_start:]
  print(f"{len(main_shards)} main shards to merge")
  tf.io.gfile.makedirs(args.output_dir)

  missing_total = []
  for si, shard in enumerate(main_shards):
    name = os.path.basename(shard)
    out_path = os.path.join(args.output_dir, name)
    if tf.io.gfile.exists(out_path):
      n_in, n_out = count_records(shard), count_records(out_path)
      if n_in == n_out:
        print(f"[{si + 1}/{len(main_shards)}] {name}: exists with {n_out} records, skip")
        continue
      print(f"[{si + 1}/{len(main_shards)}] {name}: exists but {n_out}!={n_in}, rewriting")

    tmp_path = out_path + ".tmp"
    n, missing = 0, []
    with tf.io.TFRecordWriter(tmp_path) as writer:
      for raw in tf.data.TFRecordDataset(shard):
        ex = tf.train.Example.FromString(raw.numpy())
        sid = ex.features.feature["sample_id"].bytes_list.value[0].decode()
        blob = table.get(sid)
        if blob is None:
          missing.append(sid)
          if not args.allow_missing:
            raise KeyError(f"{name}: sample_id {sid} has no CLIP sidecar entry")
          continue
        feat = tf.io.parse_tensor(blob, out_type=tf.float16).numpy().astype(np.float32)
        assert feat.shape == (257, 1280), (sid, feat.shape)
        ex.features.feature["clip_feature"].bytes_list.value.append(
            tf.io.serialize_tensor(feat).numpy()
        )
        writer.write(ex.SerializeToString())
        n += 1
    tf.io.gfile.rename(tmp_path, out_path, overwrite=True)
    missing_total.extend(missing)
    print(f"[{si + 1}/{len(main_shards)}] {name}: {n} records merged"
          + (f", {len(missing)} MISSING dropped" if missing else ""), flush=True)

  print(f"MERGE DONE. total missing: {len(missing_total)}")
  if missing_total:
    print("first missing:", missing_total[:10])


if __name__ == "__main__":
  main()
