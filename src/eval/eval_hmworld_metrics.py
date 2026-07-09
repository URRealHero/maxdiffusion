#!/usr/bin/env python3
# Copyright 2026 Google LLC. Apache-2.0.
r"""Route A — HM-World (HyDRA-protocol) metric harness.

Reproduces HyDRA paper Table 2 metrics on OUR generated videos. Runs OFF-TPU, on the
local GPU box in the `captain_safari` conda env (PyTorch). Generation is done separately
on TPU (examples/eval_hmworld_generate.sh -> {sid}_gen.mp4 per clip); this tool scores them.

Metrics (HyDRA arXiv 2603.25716, Table 2) — NOT the repo's preference metrics:
  PSNR, SSIM, LPIPS  : pixel/perceptual reconstruction, generated TARGET vs GT target frames.
  Subj. Cons.        : VBench subject consistency (DINO frame-to-frame cosine), gen target.
  Bg.   Cons.        : VBench background consistency (CLIP frame-to-frame cosine), gen target.
  DSC_GT             : crop moving subject (YOLOv11 bbox) -> CLIP feat -> cosine(pred, GT).
  DSC_ctx            : same, cosine(pred, context) -> identity preserved across out-of-sight gap.

Frame layout: each clip = cond(77) + tgt(77) = 154 frames, sampled to NUM_FRAMES (153) by
compute_indices (mirrors tpu_tfrecord_encoder/encode_concat_camera.py). The cond/tgt boundary
in the sampled video = #(sampled indices < cond_len). PSNR/SSIM/LPIPS + DSC_GT compare the
TARGET window; DSC_ctx compares the target subject vs the CONTEXT (cond) subject.

GT comes from the RAW HM-World assets (not the VAE-decoded latents): tgt.mp4 / cond.mp4 /
tgt_mask.mp4 under raw_root/<sample_id>/. check.json gives the exit/entry event window.

Usage (captain_safari env, GPU):
  python eval_hmworld_metrics.py \
    --gen_dir   /local/hmworld_eval/lora-fix-95k \      # {sid}_gen.mp4 (synced from VIZ_OUT)
    --raw_root  /local/HM-World \                       # <sid>/{cond,tgt,cond_mask,tgt_mask}.mp4 + check.json
    --metrics   psnr,ssim,lpips,subj_cons,bg_cons,dsc \
    --out_csv   /local/hmworld_eval/lora-fix-95k.csv
  # repeat for the base run, then compare the two CSV aggregates.

Status: PSNR/SSIM/LPIPS + frame/boundary logic are implemented and runnable. VBench
(subj/bg) and DSC (YOLOv11 + CLIP) are wired with lazy deps + paper-faithful logic but need
the env packages installed and a final pass against a real check.json schema (see _event_window).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

# --- constants matching the encode/generation recipe ---
NUM_FRAMES = 153
COND_LEN = 77   # HM-World cond.mp4 frames (verified from manifest: cond(77)+tgt(77)=154)
TGT_LEN = 77
SAMPLE_MODE = "linspace"


# ======================= frame I/O + cond/tgt split =======================

def compute_indices(total, num_frames, sample_mode=SAMPLE_MODE):
  """Mirror of tpu_tfrecord_encoder.encode_concat_camera.compute_indices (kept inline so this
  tool is standalone). total>=num_frames: linspace (or first); else pad-repeat last."""
  if total >= num_frames:
    if sample_mode == "first":
      return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, total - 1, num_frames).round().astype(np.int64)
  return np.array(list(range(total)) + [total - 1] * (num_frames - total), dtype=np.int64)


def read_video(path):
  """RGB uint8 [F,H,W,3]. Tries decord -> imageio -> cv2 (whichever the env has)."""
  try:
    import decord
    vr = decord.VideoReader(path)
    return vr.get_batch(range(len(vr))).asnumpy().astype(np.uint8)
  except Exception:
    pass
  try:
    import imageio.v3 as iio
    return np.asarray(iio.imread(path, plugin="pyav")).astype(np.uint8)
  except Exception:
    pass
  import cv2
  cap = cv2.VideoCapture(path)
  out = []
  while True:
    ok, fr = cap.read()
    if not ok:
      break
    out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
  cap.release()
  return np.stack(out).astype(np.uint8)


def _resize(frames, h, w):
  import cv2
  return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def load_clip_assets(sid, gen_dir, raw_root, height, width):
  """Return dict with sampled gen/GT cond+tgt frames (uint8, HxW) and the tgt subject mask.

  gen video is the model's NUM_FRAMES output (cond+tgt concat). GT cond/tgt come from the raw
  mp4s sampled by the SAME concat indices, so pred frame i <-> GT frame i. boundary splits
  cond vs tgt. Returns None if assets are missing (caller skips + logs)."""
  gen_path = os.path.join(gen_dir, f"{sid}_gen.mp4")
  sdir = os.path.join(raw_root, sid)
  cond_p, tgt_p = os.path.join(sdir, "cond.mp4"), os.path.join(sdir, "tgt.mp4")
  if not (os.path.isfile(gen_path) and os.path.isfile(cond_p) and os.path.isfile(tgt_p)):
    return None

  gen = read_video(gen_path)
  cond_raw, tgt_raw = read_video(cond_p), read_video(tgt_p)
  cond_len, tgt_len = cond_raw.shape[0], tgt_raw.shape[0]

  # GT subject masks (white=subject) for BOTH cond (context) and tgt windows. With these we
  # crop the subject by mask bbox (DSC) and never need YOLO -> no ultralytics dependency.
  def _seg(name, n):
    p = os.path.join(sdir, name)
    if not os.path.isfile(p):
      return None
    mr = read_video(p)[:n]
    if mr.shape[0] < n:  # pad short masks by repeating last frame
      mr = np.concatenate([mr, np.repeat(mr[-1:], n - mr.shape[0], axis=0)], axis=0)
    return mr
  cm, tm = _seg("cond_mask.mp4", cond_len), _seg("tgt_mask.mp4", tgt_len)

  # ---- V2V layout: the generation is the TARGET clip ONLY (77 frames), frame-aligned 1:1
  # with raw tgt.mp4 (cond is a model input, not generated; no concat subsampling). Detected
  # by the gen frame count matching tgt_len instead of the fun-camera NUM_FRAMES concat. ----
  if abs(gen.shape[0] - tgt_len) <= 2 and gen.shape[0] < NUM_FRAMES:
    n = min(gen.shape[0], tgt_len)
    reentry_raw = _reentry_raw_frame(os.path.join(sdir, "check.json"))
    event = None if reentry_raw is None else (min(int(reentry_raw), n), n)
    return {
        "sid": sid, "boundary": 0,
        "gen_cond": gen[:0], "gen_tgt": _resize(gen[:n], height, width),
        "gt_cond": _resize(cond_raw, height, width), "gt_tgt": _resize(tgt_raw[:n], height, width),
        "gen_full": _resize(gen[:n], height, width), "gt_full": _resize(tgt_raw[:n], height, width),
        "cond_mask": None if cm is None else (_resize(cm, height, width)[..., 0] > 127),
        "tgt_mask": None if tm is None else (_resize(tm[:n], height, width)[..., 0] > 127),
        "event": event,
    }

  if gen.shape[0] != NUM_FRAMES:
    # generation should be NUM_FRAMES; tolerate small off-by and trim/pad to be safe
    gen = gen[:NUM_FRAMES]
  gen = _resize(gen, height, width)

  total = cond_len + tgt_len
  idx = compute_indices(total, NUM_FRAMES)
  boundary = int((idx < cond_len).sum())  # #sampled frames that fall in the cond segment

  concat_raw = np.concatenate([cond_raw, tgt_raw], axis=0)
  gt = _resize(concat_raw[idx], height, width)

  # masks sampled by the same concat indices and split at the boundary.
  full_mask = None
  if cm is not None and tm is not None:
    full_mask = (_resize(np.concatenate([cm, tm], axis=0)[idx], height, width)[..., 0] > 127)  # [F,H,W] bool

  return {
      "sid": sid, "boundary": boundary,
      "gen_cond": gen[:boundary], "gen_tgt": gen[boundary:],
      "gt_cond": gt[:boundary], "gt_tgt": gt[boundary:],
      "gen_full": gen, "gt_full": gt,   # full concat (cond+tgt), for --eval_region full (ti2v)
      "cond_mask": None if full_mask is None else full_mask[:boundary],
      "tgt_mask": None if full_mask is None else full_mask[boundary:],
      "event": _target_reentry_window(os.path.join(sdir, "check.json"), idx, cond_len, boundary),
  }


def _target_reentry_window(check_path, idx, cond_len, boundary):
  """(start, end) in TARGET-output-index space for the re-entry window: from the output frame
  where the subject re-enters (check.json tgt_check inFrame=true) to the end of the target.
  None if the subject never exits/re-enters -> DSC uses the whole target window. Maps the raw
  tgt re-entry frame through the concat sampling to the output-target index.

  check.json schema (verified): {"cond_check": {actor: {char: [{frame,inFrame}]}},
  "tgt_check": {...}} — inFrame=false during cond = exit; inFrame=true during tgt = re-entry."""
  reentry_raw = _reentry_raw_frame(check_path)
  tgt_out_len = len(idx) - boundary
  if reentry_raw is None:
    return None
  tgt_sampled_raw = idx[boundary:] - cond_len  # raw tgt-frame index per output-target position
  pos = np.where(tgt_sampled_raw >= reentry_raw)[0]
  start = int(pos[0]) if len(pos) else 0
  return (start, tgt_out_len)


def _reentry_raw_frame(check_path):
  """Earliest raw tgt-frame where any character re-enters (inFrame=true), from check.json
  tgt_check. None if the file is missing or has no re-entry event."""
  if not os.path.isfile(check_path):
    return None
  try:
    with open(check_path) as f:
      data = json.load(f)
  except Exception:
    return None
  frames = []
  for actor in data.get("tgt_check", {}).values():
    for char_events in actor.values():
      for ev in char_events:
        if ev.get("inFrame") is True and "frame" in ev:
          frames.append(int(ev["frame"]))
  return min(frames) if frames else None


# ======================= metrics =======================

EVAL_REGION = "target"   # "target" = target-window (v2v/default); "full" = full concat (ti2v/lora)
def _reg(a, which):      # which in ("gen","gt") -> full-concat array when EVAL_REGION=="full"
  k = f"{which}_full" if EVAL_REGION == "full" and f"{which}_full" in a else f"{which}_tgt"
  return a[k]

def m_psnr_ssim_lpips(a):
  """PSNR/SSIM/LPIPS per-frame mean; region = target window (default) or full concat (--eval_region full)."""
  from skimage.metrics import peak_signal_noise_ratio as psnr
  from skimage.metrics import structural_similarity as ssim
  g, t = _reg(a, "gen"), _reg(a, "gt")
  n = min(len(g), len(t))
  g, t = g[:n], t[:n]
  ps = float(np.mean([psnr(t[i], g[i], data_range=255) for i in range(n)]))
  ss = float(np.mean([ssim(t[i], g[i], channel_axis=-1, data_range=255) for i in range(n)]))
  lp = _lpips(g, t)
  return {"PSNR": ps, "SSIM": ss, "LPIPS": lp}


_LPIPS_NET = None
def _lpips(g, t):
  global _LPIPS_NET
  import torch, lpips
  if _LPIPS_NET is None:
    _LPIPS_NET = lpips.LPIPS(net="alex").to(_DEVICE).eval()
  def to_t(x):  # [F,H,W,3] uint8 -> [F,3,H,W] in [-1,1]
    return (torch.from_numpy(x).float().permute(0, 3, 1, 2).to(_DEVICE) / 127.5 - 1.0)
  with torch.no_grad():
    d = _LPIPS_NET(to_t(g), to_t(t))
  return float(d.mean().cpu())


def m_consistency(a):
  """VBench Subject Consistency (DINO) + Background Consistency (CLIP) on the generated target:
  mean cosine similarity of consecutive-frame features (and vs the first frame), per VBench.
  Requires the `vbench` package OR torch+dino+clip; lazily imported."""
  try:
    return _consistency_vbench(_reg(a, "gen"))
  except Exception as e:
    return {"Subj.Cons": float("nan"), "Bg.Cons": float("nan"), "_cons_err": str(e)[:80]}


def _consistency_vbench(frames):
  """Faithful-ish: Subj = DINO ViT feature frame-coherence; Bg = CLIP feature frame-coherence.
  VBench averages (1) consecutive-frame cosine and (2) cosine to the first frame."""
  import torch
  feats_dino = _embed_frames(frames, backbone="dino")
  feats_clip = _embed_frames(frames, backbone="clip")
  def coherence(F):
    F = torch.nn.functional.normalize(F, dim=-1)
    consec = (F[1:] * F[:-1]).sum(-1).mean()
    tofirst = (F[1:] * F[:1]).sum(-1).mean()
    return float(((consec + tofirst) / 2).cpu())
  return {"Subj.Cons": coherence(feats_dino), "Bg.Cons": coherence(feats_clip)}


def m_dsc(a):
  """Dynamic Subject Consistency: crop the moving subject (YOLOv11 bbox, or GT mask bbox as
  fallback) per frame, CLIP-embed, cosine sim of pred-vs-GT (DSC_GT) and pred-vs-context
  (DSC_ctx), over the re-entry window. Returns NaN with an error tag if deps/bbox missing."""
  try:
    import torch
    win = a["event"]  # (start,end) in target idx, or None -> whole target window
    gt_crops = _subject_crops(a["gt_tgt"], a.get("tgt_mask"), win)
    pr_crops = _subject_crops(a["gen_tgt"], a.get("tgt_mask"), win)   # GT-mask region on pred (recon-aligned)
    ctx_ref = _subject_crops(a["gt_cond"], a.get("cond_mask"), None)  # context subject (pre-occlusion)
    fp = _embed_crops(pr_crops); fg = _embed_crops(gt_crops); fc = _embed_crops(ctx_ref)
    def cos(x, y):
      x = torch.nn.functional.normalize(x.mean(0, keepdim=True), dim=-1)
      y = torch.nn.functional.normalize(y.mean(0, keepdim=True), dim=-1)
      return float((x * y).sum(-1).cpu())
    return {"DSC_GT": cos(fp, fg), "DSC_ctx": cos(fp, fc)}
  except Exception as e:
    return {"DSC_GT": float("nan"), "DSC_ctx": float("nan"), "_dsc_err": str(e)[:80]}


# --- embedding backends (lazy singletons) ---
_MODELS = {}
def _embed_frames(frames, backbone):
  """[F,H,W,3] uint8 -> [F,D] features via DINO or CLIP image encoder."""
  import torch
  key = f"img_{backbone}"
  if key not in _MODELS:
    if backbone == "dino":
      m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(_DEVICE).eval()
      _MODELS[key] = ("dino", m)
    else:
      import open_clip
      m, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
      _MODELS[key] = ("clip", m.to(_DEVICE).eval(), prep)
  import torch.nn.functional as Fnn
  t = torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(_DEVICE) / 255.0
  t = Fnn.interpolate(t, size=224, mode="bilinear", align_corners=False)
  with torch.no_grad():
    entry = _MODELS[key]
    if entry[0] == "dino":
      return entry[1](t)
    return entry[1].encode_image(t)


def _subject_crops(frames, mask, win):
  """List of subject-region crops (HxWx3 uint8). Uses YOLOv11 person/largest-box detection;
  falls back to the GT mask bbox when a mask is provided. `win`=(s,e) restricts frames."""
  if frames is None or len(frames) == 0:
    return []
  s, e = (win if win else (0, len(frames)))
  sel = frames[s:e]
  msel = None if mask is None else mask[s:e]
  crops = []
  if msel is not None:
    for fr, mk in zip(sel, msel):
      ys, xs = np.where(mk)
      if len(xs):
        crops.append(fr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    if crops:
      return crops
  # YOLOv11 fallback (faithful to the paper)
  from ultralytics import YOLO
  if "yolo" not in _MODELS:
    _MODELS["yolo"] = YOLO("yolo11n.pt")
  for fr in sel:
    r = _MODELS["yolo"].predict(fr[..., ::-1], verbose=False)[0]
    if len(r.boxes):
      x1, y1, x2, y2 = r.boxes.xyxy[r.boxes.conf.argmax()].cpu().numpy().astype(int)
      crops.append(fr[max(0, y1):y2, max(0, x1):x2])
  return crops


def _embed_crops(crops):
  import torch
  if not crops:
    raise ValueError("no subject crops")
  resized = np.stack([_resize_one(c, 224, 224) for c in crops])
  return _embed_frames(resized, backbone="clip")


def _resize_one(c, h, w):
  import cv2
  if c.size == 0:
    return np.zeros((h, w, 3), np.uint8)
  return cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)


# ======================= driver =======================

METRIC_FNS = {
    "psnr": m_psnr_ssim_lpips, "ssim": m_psnr_ssim_lpips, "lpips": m_psnr_ssim_lpips,
    "subj_cons": m_consistency, "bg_cons": m_consistency, "dsc": m_dsc,
}
_DEVICE = "cuda"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gen_dir", required=True, help="dir of {sid}_gen.mp4 (synced from VIZ_OUT)")
  ap.add_argument("--raw_root", required=True, help="HM-World raw root with <sid>/{cond,tgt,*_mask}.mp4 + check.json")
  ap.add_argument("--metrics", default="psnr,ssim,lpips,subj_cons,bg_cons,dsc")
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--width", type=int, default=832)
  ap.add_argument("--limit", type=int, default=0, help="0 = all sids found in gen_dir")
  ap.add_argument("--shard", default="0/1", help="i/N — process sids[i::N] for CPU-parallel sharding")
  ap.add_argument("--out_csv", required=True)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--eval_region", default="target", choices=["target", "full"],
                  help="target=target window (v2v); full=full cond+tgt concat (ti2v/lora)")
  args = ap.parse_args()
  global _DEVICE, EVAL_REGION
  _DEVICE = args.device
  EVAL_REGION = args.eval_region

  want = [m.strip() for m in args.metrics.split(",") if m.strip()]
  run_recon = any(m in want for m in ("psnr", "ssim", "lpips"))
  run_cons = any(m in want for m in ("subj_cons", "bg_cons"))
  run_dsc = "dsc" in want

  sids = sorted({f[:-len("_gen.mp4")] for f in os.listdir(args.gen_dir) if f.endswith("_gen.mp4")})
  _si, _sn = (int(x) for x in args.shard.split("/"))
  sids = sids[_si::_sn]
  if args.limit:
    sids = sids[: args.limit]
  print(f"[hmworld-metrics] {len(sids)} clips; metrics={want}", flush=True)

  rows, skipped = [], 0
  for i, sid in enumerate(sids):
    a = load_clip_assets(sid, args.gen_dir, args.raw_root, args.height, args.width)
    if a is None:
      skipped += 1
      continue
    row = {"sample_id": sid, "boundary": a["boundary"]}
    if run_recon:
      row.update(m_psnr_ssim_lpips(a))
    if run_cons:
      row.update(m_consistency(a))
    if run_dsc:
      row.update(m_dsc(a))
    rows.append(row)
    if (i + 1) % 25 == 0:
      print(f"  {i + 1}/{len(sids)} done", flush=True)

  if not rows:
    sys.exit(f"no scorable clips (skipped {skipped}); check gen_dir/raw_root paths")

  keys = [k for k in ("PSNR", "SSIM", "LPIPS", "Subj.Cons", "Bg.Cons", "DSC_GT", "DSC_ctx")
          if any(k in r for r in rows)]
  with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "boundary"] + keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

  print(f"\n[hmworld-metrics] {len(rows)} scored, {skipped} skipped -> {args.out_csv}")
  print("AGGREGATE (mean):")
  for k in keys:
    vals = [r[k] for r in rows if k in r and r[k] == r[k]]  # drop NaN
    if vals:
      print(f"  {k:10s} {np.mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
  main()
