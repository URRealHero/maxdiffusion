"""Objective tail-degradation metric for generated videos (Phase H debugging).

Quantifies the "noise/corruption in the last few seconds" symptom without eyeballing:
  lap_t/h : ratio of tail vs mid-head Laplacian variance (detail retention).
            ~1.0 = sharp to the end; <<1 = detail collapse at the tail.
  df_t    : mean frame-to-frame abs diff over the tail (temporal flicker/instability).

Use it to compare OUR TPU output vs Captain-Safari's GPU validate_lora output on the
SAME sample: if CS's lap_t/h is also ~0.3, the long-horizon degradation is inherent to
the model/recipe and our pipeline reproduces it faithfully; if CS stays ~1.0, we diverge.

  python video_tail_degradation.py vid1.mp4 [vid2.mp4 ...]
"""
import sys
import cv2
import numpy as np


def analyze(path):
  cap = cv2.VideoCapture(path)
  lap, std, df = [], [], []
  prev = None
  while True:
    ok, f = cap.read()
    if not ok:
      break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap.append(cv2.Laplacian(g, cv2.CV_32F).var())
    std.append(float(g.std()))
    df.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
    prev = g
  cap.release()
  lap, std, df = map(np.array, (lap, std, df))
  n = len(lap)
  if n == 0:
    return None
  h = slice(int(n * 0.1), int(n * 0.5))   # mid-head reference
  t = slice(int(n * 0.85), n)             # tail
  return dict(n=n, lap_h=lap[h].mean(), lap_t=lap[t].mean(),
              lap_ratio=lap[t].mean() / (lap[h].mean() + 1e-6),
              std_h=std[h].mean(), std_t=std[t].mean(),
              df_h=df[h].mean(), df_t=df[t].mean())


if __name__ == "__main__":
  print(f"{'video':40s} {'F':>4s} {'lap_h':>7s} {'lap_t':>7s} {'lap_t/h':>8s} {'df_h':>5s} {'df_t':>5s}")
  for p in sys.argv[1:]:
    r = analyze(p)
    if r is None:
      print(f"{p:40s}  NO FRAMES")
      continue
    name = p.rsplit("/", 1)[-1]
    print(f"{name:40s} {r['n']:4d} {r['lap_h']:7.1f} {r['lap_t']:7.1f} {r['lap_ratio']:8.2f} "
          f"{r['df_h']:5.2f} {r['df_t']:5.2f}   (lap_t/h~1=sharp, <<1=tail collapse)")
