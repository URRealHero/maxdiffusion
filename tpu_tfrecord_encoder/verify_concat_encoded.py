#!/usr/bin/env python3
"""Verify a concat-camera encoded TFRecord dataset: COMPLETENESS + FINITENESS.

The concat-camera encoder writes per record:
  latents [48,F,H,W], encoder_hidden_states [512,4096],
  camera_extrinsic [F,3,4], camera_intrinsic [F,3,3]
and a per-host metadata sidecar `metadata_host_*.jsonl` (one line per encoded sample).

FAST by default: completeness comes from the metadata sidecars (no record reads), and
finiteness is sampled across a spread of shards with PARALLEL GCS reads.

Usage (run where GCS + tensorflow are available, e.g. the maxdiffusion venv on a TPU worker):
  python verify_concat_encoded.py gs://.../concat_camera_encoded_full --expected 55961
  python verify_concat_encoded.py gs://.../concat_camera_encoded_full --count   # + exact tfrecord count
  python verify_concat_encoded.py gs://.../concat_camera_encoded_full --full    # scan EVERY record (slow)
"""
import argparse

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
FEATURES = {
    "latents": tf.io.FixedLenFeature([], tf.string),
    "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    "camera_extrinsic": tf.io.FixedLenFeature([], tf.string),
    "camera_intrinsic": tf.io.FixedLenFeature([], tf.string),
}


def _parse(raw):
  p = tf.io.parse_single_example(raw, FEATURES)
  return {k: tf.io.parse_tensor(p[k], out_type=tf.float32).numpy() for k in FEATURES}


def _count_parallel(shards):
  """Exact record count, streamed in the TF runtime with parallel GCS reads (no parse)."""
  ds = tf.data.TFRecordDataset(shards, num_parallel_reads=AUTOTUNE)
  return int(ds.reduce(np.int64(0), lambda c, _: c + 1).numpy())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path", help="encoded dir (local or gs://)")
  ap.add_argument("--expected", type=int, default=0, help="expected sample count (e.g. manifest size)")
  ap.add_argument("--max-shards", type=int, default=40, help="shards to sample for finiteness (spread across the set)")
  ap.add_argument("--max-records", type=int, default=400, help="total records to stat in sample mode")
  ap.add_argument("--count", action="store_true", help="also compute the EXACT tfrecord record count (parallel)")
  ap.add_argument("--full", action="store_true", help="scan EVERY record for finiteness (slow)")
  args = ap.parse_args()

  base = args.path.rstrip("/")
  shards = sorted(tf.io.gfile.glob(f"{base}/*.tfrec")) or sorted(tf.io.gfile.glob(f"{base}/**/*.tfrec"))
  metas = sorted(tf.io.gfile.glob(f"{base}/metadata_host_*.jsonl"))
  print(f"dir: {base}")
  print(f"shards: {len(shards)}   metadata sidecars: {len(metas)}")
  if not shards:
    print("!! no .tfrec shards found — dataset is empty / wrong path / encode not started")
    return

  # --- completeness via metadata sidecars (fast: just count jsonl lines) ---
  meta_count = 0
  for m in metas:
    with tf.io.gfile.GFile(m, "r") as f:
      meta_count += sum(1 for line in f if line.strip())
  print(f"records (metadata count): {meta_count}", flush=True)
  if args.expected:
    tag = "OK" if meta_count == args.expected else f"(short by {args.expected - meta_count})"
    print(f"completeness (metadata): {meta_count}/{args.expected}  {tag}")

  # --- finiteness: parallel read of a sample (or all with --full) ---
  if args.full:
    scan = shards
    ds = tf.data.TFRecordDataset(scan, num_parallel_reads=AUTOTUNE)
  else:
    step = max(1, len(shards) // args.max_shards)
    scan = shards[::step][: args.max_shards]
    ds = tf.data.TFRecordDataset(scan, num_parallel_reads=AUTOTUNE).take(args.max_records)
  print(f"\nfiniteness scan: {'ALL records' if args.full else f'~{args.max_records} records across {len(scan)} shards'}", flush=True)

  agg = {k: {"finite": 0.0, "n": 0, "mn": np.inf, "mx": -np.inf} for k in FEATURES}
  first_bad = None
  scanned = 0
  for raw in ds:
    rec = _parse(raw)
    scanned += 1
    for k, a in rec.items():
      fin = float(np.isfinite(a).mean())
      agg[k]["finite"] += fin
      agg[k]["n"] += 1
      if np.isfinite(a).any():
        agg[k]["mn"] = min(agg[k]["mn"], float(np.nanmin(a)))
        agg[k]["mx"] = max(agg[k]["mx"], float(np.nanmax(a)))
      if fin < 1.0 and first_bad is None:
        first_bad = (k, fin, scanned)
    if args.full and scanned % 2000 == 0:
      print(f"  ...{scanned} records", flush=True)

  print(f"scanned {scanned} records")
  for k, s in agg.items():
    if s["n"]:
      print(f"  {k:22s} finite_frac={s['finite']/s['n']:.5f}  min={s['mn']:+.3f}  max={s['mx']:+.3f}  (n={s['n']})")
  if first_bad:
    k, fin, idx = first_bad
    print(f"\n!! NON-FINITE: field={k} finite_frac={fin:.4f} at scanned record #{idx}")
  else:
    print(f"\nOK: no non-finite values in {'ALL' if args.full else 'sampled'} records.")

  # --- optional exact record count ---
  if args.count:
    print("\ncounting all tfrecords (parallel)...", flush=True)
    total = _count_parallel(shards)
    tag = "OK" if not args.expected or total == args.expected else f"!! vs expected {args.expected}"
    print(f"records (tfrecord count): {total}  {tag}")


if __name__ == "__main__":
  main()
