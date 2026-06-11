#!/usr/bin/env python3
"""Dump scalar metrics from a TensorBoard logdir (local or gs://) as plain text,
so they can be copy-pasted. Defaults to the first N steps of the debug/* and
learning/* tags written by the WAN trainers.

Usage:
  python dump_tb_scalars.py gs://data_us_east5_a/maxdiffusion/wan/ti2v5b-vace/vace-100k-bs64-lr-5e-5/tensorboard \
      --max-steps 12
"""
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("logdir", help="TensorBoard logdir (local or gs://). Use the path printed as "
                                 "'TensorBoard logs will be written to:' in the training log.")
  ap.add_argument("--max-steps", type=int, default=12, help="How many of the earliest steps to print.")
  ap.add_argument("--filter", default="debug/,learning/",
                  help="Comma-separated tag prefixes to include.")
  args = ap.parse_args()

  prefixes = tuple(p for p in args.filter.split(",") if p)
  acc = EventAccumulator(args.logdir, size_guidance={"scalars": 0})  # 0 = load all
  acc.Reload()

  tags = [t for t in acc.Tags().get("scalars", []) if t.startswith(prefixes)]
  if not tags:
    print(f"No scalar tags matching {prefixes} found. All scalar tags:")
    for t in acc.Tags().get("scalars", []):
      print(f"  {t}")
    return

  # Build {step: {tag: value}} for the earliest steps.
  by_step = {}
  for tag in tags:
    for ev in acc.Scalars(tag):
      by_step.setdefault(ev.step, {})[tag] = ev.value

  steps = sorted(by_step)[: args.max_steps]
  print(f"logdir: {args.logdir}")
  print(f"tags ({len(tags)}): {', '.join(sorted(tags))}\n")
  for s in steps:
    print(f"=== step {s} ===")
    for tag in sorted(by_step[s]):
      print(f"  {tag:48s} {by_step[s][tag]:.6g}")
    print()


if __name__ == "__main__":
  main()
