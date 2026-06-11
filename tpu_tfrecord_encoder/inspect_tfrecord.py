#!/usr/bin/env python3
"""Inspect encoded TV2V TFRecords: print shapes/stats of latents, cond_latents,
encoder_hidden_states for the first few records, and validate against expected
latent geometry.

Usage:
  python inspect_tfrecord.py gs://data_us_central1_a/hmworld_data/tv2v_ti2v5b_encoded_full \
      --num-records 3 --height 480 --width 832 --num-frames 81

  # or point directly at a single .tfrec file:
  python inspect_tfrecord.py gs://.../host_000_run_..._file_000000.tfrec --num-records 3
"""
import argparse
import numpy as np
import tensorflow as tf


def expected_latent_shape(height, width, num_frames, z_dim=48,
                          spatial=16, temporal=4):
  f_lat = (num_frames - 1) // temporal + 1
  return (z_dim, f_lat, height // spatial, width // spatial)


def list_tfrecords(path):
  if path.endswith(".tfrec") or path.endswith(".tfrecord"):
    return [path]
  patterns = [f"{path.rstrip('/')}/*.tfrec", f"{path.rstrip('/')}/*.tfrecord"]
  files = []
  for pat in patterns:
    files.extend(tf.io.gfile.glob(pat))
  return sorted(files)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path", help="TFRecord file or directory (local or gs://)")
  ap.add_argument("--num-records", type=int, default=3)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--width", type=int, default=832)
  ap.add_argument("--num-frames", type=int, default=81,
                  help="The num_frames used at ENCODING time (not training).")
  ap.add_argument("--cond-field", default="cond_latents")
  args = ap.parse_args()

  files = list_tfrecords(args.path)
  if not files:
    raise SystemExit(f"No .tfrec/.tfrecord files found under: {args.path}")
  print(f"Found {len(files)} TFRecord file(s). Reading first: {files[0]}\n")

  exp = expected_latent_shape(args.height, args.width, args.num_frames)
  print(f"Expected per-sample latent shape [C,F,H,W] = {exp}")
  print(f"  (height={args.height} width={args.width} num_frames={args.num_frames} "
        f"-> F_lat=(num_frames-1)//4+1={exp[1]}, H//16={exp[2]}, W//16={exp[3]})\n")

  feature_description = {
      "latents": tf.io.FixedLenFeature([], tf.string),
      args.cond_field: tf.io.FixedLenFeature([], tf.string),
      "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
  }

  ds = tf.data.TFRecordDataset(files[0])
  for i, raw in enumerate(ds.take(args.num_records)):
    parsed = tf.io.parse_single_example(raw, feature_description)
    lat = tf.io.parse_tensor(parsed["latents"], out_type=tf.float32).numpy()
    cond = tf.io.parse_tensor(parsed[args.cond_field], out_type=tf.float32).numpy()
    ehs = tf.io.parse_tensor(parsed["encoder_hidden_states"], out_type=tf.float32).numpy()

    def stats(name, a, expect=None):
      ok = "" if expect is None else ("  OK" if tuple(a.shape) == tuple(expect) else f"  !! EXPECTED {expect}")
      print(f"  {name:22s} shape={str(a.shape):24s} dtype={a.dtype} "
            f"min={np.nanmin(a):+.3f} max={np.nanmax(a):+.3f} "
            f"mean={np.nanmean(a):+.3f} std={np.nanstd(a):.3f} "
            f"finite={np.isfinite(a).mean():.4f}{ok}")

    print(f"--- record {i} ---")
    stats("latents (target)", lat, exp)
    stats(args.cond_field, cond, exp)
    stats("encoder_hidden_states", ehs)

    # Frozen-tail check: are the last temporal latent frames ~constant?
    # (padding repeats the last pixel frame, so the trailing latent frame(s)
    #  should look like a still if the source clip was shorter than num_frames.)
    if lat.ndim == 4 and lat.shape[1] >= 2:
      f = lat.shape[1]
      diffs = [float(np.mean(np.abs(lat[:, t] - lat[:, t - 1]))) for t in range(1, f)]
      print(f"    target per-frame Δ (mean abs diff vs prev frame), last 5: "
            f"{[round(d, 4) for d in diffs[-5:]]}")
      print(f"    (a near-zero trailing Δ = frozen/padded frames)")
    print()


if __name__ == "__main__":
  main()
