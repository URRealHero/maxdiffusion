#!/usr/bin/env python3
"""Build a fixed seeded-random eval-set manifest for the Fun-camera comparison.

Deterministically selects N *valid* sample_ids from the full HM-World manifest
(seed-stable, independent of file order) and writes the matching rows from both
the main manifest and the caption manifest. Feeding these to
encode_concat_camera.py produces a small `fun_camera_eval_set/` TFRecord whose
records carry sample_id + caption (see make_example), so generated eval videos
are self-identifying.

Pure stdlib (json/random) so it runs anywhere; GCS download/upload is done by the
wrapper launcher (encode_fun_camera_eval.sh) via the gcloud CLI.
"""
import argparse
import json
import random


def load_jsonl(path):
  rows = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def is_valid(row):
  """Only pick clips whose required inputs exist (so the eval encode can't fail)."""
  return (
      not row.get("missing_required")
      and row.get("cond_video_exists", False)
      and row.get("tgt_video_exists", False)
      and row.get("camera_exists", False)
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True, help="full hm_world_manifest_gcs.jsonl (local path)")
  ap.add_argument("--caption-manifest", required=True, help="full hm_world_captions_concat.jsonl (local path)")
  ap.add_argument("--n", type=int, default=16)
  ap.add_argument("--seed", type=int, default=1234)
  ap.add_argument("--out-manifest", required=True)
  ap.add_argument("--out-caption-manifest", required=True)
  args = ap.parse_args()

  rows = load_jsonl(args.manifest)
  valid = [r for r in rows if is_valid(r)]
  # Sort by id first so selection depends ONLY on (seed, n), not manifest line order.
  valid.sort(key=lambda r: str(r["sample_id"]))
  rng = random.Random(args.seed)
  chosen = rng.sample(valid, min(args.n, len(valid)))

  caps = load_jsonl(args.caption_manifest)
  cap_by_id = {str(r["sample_id"]): r for r in caps}
  missing = [str(r["sample_id"]) for r in chosen if str(r["sample_id"]) not in cap_by_id]
  if missing:
    raise SystemExit(f"no caption manifest row for: {missing}")

  chosen_sorted = sorted(chosen, key=lambda r: str(r["sample_id"]))
  with open(args.out_manifest, "w") as f:
    for r in chosen_sorted:
      f.write(json.dumps(r) + "\n")
  with open(args.out_caption_manifest, "w") as f:
    for r in chosen_sorted:
      f.write(json.dumps(cap_by_id[str(r["sample_id"])]) + "\n")

  print(f"selected {len(chosen_sorted)} / {len(valid)} valid (of {len(rows)} total rows), seed={args.seed} n={args.n}")
  for r in chosen_sorted:
    print(f"  {r['sample_id']}")


if __name__ == "__main__":
  main()
