# Merge CLIP sidecar TFRecords (encode_clip_sidecar.py) into the main
# concat+camera records, producing the *_full_clip dataset the
# Wan2.1-Fun-camera trainer reads.
#
# ⚠ The June-2026 `concat_camera_encoded_full` records carry ONLY
# {latents, encoder_hidden_states, camera_extrinsic, camera_intrinsic} — no
# sample_id (that encoder predates the sample_id field). So this merge does TWO
# things per record:
#   1. recover `sample_id` POSITIONALLY from the per-host metadata jsonl:
#      line i of metadata_host_HHH_run_RRR.jsonl == record i%128 of shard
#      host_HHH_run_RRR_file_{i//128}.tfrec  (verified: every non-null
#      `tfrec_path` in the metadata agrees with i//128, 0 mismatches)
#   2. attach `clip_feature` (fp32) looked up by that sample_id.
# `latent_condition` is NOT needed: y = VAE(first frame) == latents[:, 0]
# because the Wan VAE is temporally causal (gated by check_vae_causality.py).
#
# The output records therefore gain `sample_id` too, which is what makes the
# held-out exclusion (`exclude_sample_ids_path`) possible for this dataset.
#
# IO: local or gs:// (tf.io.gfile). Run where GCS bandwidth is good (a TPU host
# CPU is ideal — no TPU devices are touched). Memory: the sidecar table is
# ~37 GB resident (55,961 x 257 x 1280 fp16); a v6e host has ~1.4 TiB.
#
# Example (one of N parallel shard ranges):
#   python merge_clip_sidecar.py \
#     --main-dir gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full \
#     --sidecar-dir gs://data_us_central1_a/hmworld_data/wan_2_1/clip_sidecar \
#     --metadata-dir gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full \
#     --output-dir gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full_clip \
#     --shard-start 0 --shard-end 56

import argparse
import json
import os
import re

import numpy as np
import tensorflow as tf

# Run IDs from the launchers are timestamped strings such as
# "tgtcam-20260710-120000", not only decimal process IDs. Anchor the capture
# between the stable run/file delimiters so hyphens and underscores are safe.
SHARD_RE = re.compile(r"host_(\d+)_run_(.+)_file_(\d+)\.tfrec$")
META_RE = re.compile(r"metadata_host_(\d+)_run_(.+)\.jsonl$")


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--main-dir", required=True)
  p.add_argument("--sidecar-dir", required=True)
  p.add_argument("--metadata-dir", required=True, help="dir holding metadata_host_*_run_*.jsonl")
  p.add_argument("--output-dir", required=True)
  p.add_argument("--records-per-shard", type=int, default=128)
  p.add_argument("--shard-start", type=int, default=0)
  p.add_argument("--shard-end", type=int, default=0, help="0 = through the end")
  return p.parse_args()


def load_metadata_sids(metadata_dir):
  """(host, run) -> [sample_id] in write order."""
  files = sorted(tf.io.gfile.glob(os.path.join(metadata_dir, "metadata_host_*_run_*.jsonl")))
  assert files, f"no metadata jsonl under {metadata_dir}"
  table = {}
  for path in files:
    m = META_RE.search(os.path.basename(path))
    assert m, path
    key = (m.group(1), m.group(2))
    with tf.io.gfile.GFile(path, "r") as f:
      sids = [json.loads(line)["sample_id"] for line in f if line.strip()]
    table[key] = sids
  total = sum(len(v) for v in table.values())
  print(f"metadata: {len(table)} hosts, {total} sample_ids", flush=True)
  return table


def load_sidecars(sidecar_dir):
  """sample_id -> raw serialized fp16 tensor bytes."""
  shards = sorted(tf.io.gfile.glob(os.path.join(sidecar_dir, "clip_*_file_*.tfrec")))
  assert shards, f"no sidecar shards under {sidecar_dir}"
  table = {}
  for i, shard in enumerate(shards):
    for raw in tf.data.TFRecordDataset(shard):
      ex = tf.train.Example.FromString(raw.numpy())
      sid = ex.features.feature["sample_id"].bytes_list.value[0].decode()
      table[sid] = ex.features.feature["clip_feature_fp16"].bytes_list.value[0]
    if (i + 1) % 25 == 0:
      print(f"sidecars: {i + 1}/{len(shards)} shards, {len(table)} features", flush=True)
  print(f"sidecars loaded: {len(table)} features", flush=True)
  return table


def count_records(path):
  return sum(1 for _ in tf.data.TFRecordDataset(path))


def main():
  args = parse_args()
  meta = load_metadata_sids(args.metadata_dir)
  clip = load_sidecars(args.sidecar_dir)

  shards = sorted(tf.io.gfile.glob(os.path.join(args.main_dir, "*.tfrec")))
  assert shards, f"no main shards under {args.main_dir}"
  end = args.shard_end or len(shards)
  shards = shards[args.shard_start:end]
  print(f"{len(shards)} main shards in range [{args.shard_start}, {end})", flush=True)
  tf.io.gfile.makedirs(args.output_dir)

  for si, shard in enumerate(shards):
    name = os.path.basename(shard)
    m = SHARD_RE.search(name)
    assert m, f"unexpected shard name {name}"
    host, run, fileno = m.group(1), m.group(2), int(m.group(3))
    sids_all = meta.get((host, run))
    assert sids_all is not None, f"no metadata for host {host} run {run}"
    base = fileno * args.records_per_shard
    expected = sids_all[base: base + args.records_per_shard]

    out_path = os.path.join(args.output_dir, name)
    if tf.io.gfile.exists(out_path):
      if count_records(out_path) == len(expected):
        print(f"[{si + 1}/{len(shards)}] {name}: complete, skip", flush=True)
        continue
      print(f"[{si + 1}/{len(shards)}] {name}: incomplete, rewriting", flush=True)

    tmp_path = out_path + ".tmp"
    n = 0
    with tf.io.TFRecordWriter(tmp_path) as writer:
      for raw in tf.data.TFRecordDataset(shard):
        assert n < len(expected), f"{name}: more records than metadata sids ({len(expected)})"
        sid = expected[n]
        ex = tf.train.Example.FromString(raw.numpy())
        assert "sample_id" not in ex.features.feature, f"{name}: record already has sample_id"
        blob = clip.get(sid)
        assert blob is not None, f"{name}#{n}: no CLIP feature for sample_id {sid}"
        feat = tf.io.parse_tensor(blob, out_type=tf.float16).numpy().astype(np.float32)
        assert feat.shape == (257, 1280), (sid, feat.shape)
        ex.features.feature["sample_id"].bytes_list.value.append(sid.encode())
        ex.features.feature["clip_feature"].bytes_list.value.append(tf.io.serialize_tensor(feat).numpy())
        writer.write(ex.SerializeToString())
        n += 1
    assert n == len(expected), f"{name}: {n} records but {len(expected)} metadata sids"
    tf.io.gfile.rename(tmp_path, out_path, overwrite=True)
    print(f"[{si + 1}/{len(shards)}] {name}: {n} records merged", flush=True)

  print("MERGE RANGE DONE")


if __name__ == "__main__":
  main()
