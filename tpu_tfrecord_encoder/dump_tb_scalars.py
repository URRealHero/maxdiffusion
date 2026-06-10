#!/usr/bin/env python3
"""Dump scalar metrics from a TensorBoard logdir as plain text.

Supports local paths and gs:// paths directly.

Example:
  python dump_tb_scalars.py \
    gs://data_us_east5_a/maxdiffusion/wan/ti2v5b-vace/vace-100k-bs64-lr-5e-5-zeroinit/tensorboard \
    --max-steps 30 \
    --filter "learning/,debug/"
"""

import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logdir",
        help="TensorBoard logdir, local or gs://.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="How many of the earliest steps to print.",
    )
    ap.add_argument(
        "--filter",
        default="debug/,learning/",
        help="Comma-separated tag prefixes to include.",
    )
    args = ap.parse_args()

    prefixes = tuple(p for p in args.filter.split(",") if p)

    acc = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    acc.Reload()

    all_scalar_tags = acc.Tags().get("scalars", [])
    tags = [t for t in all_scalar_tags if t.startswith(prefixes)]

    if not tags:
        print(f"No scalar tags matching {prefixes} found. All scalar tags:")
        for t in all_scalar_tags:
            print(f"  {t}")
        return

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