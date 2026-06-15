# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

"""Visualize a checkpoint on a curated eval set: (eval_data_dir + checkpoint) ->
generate camera-control videos, named by sample_id, for manual inspection.

Single-host inference (like generate_wan_2_2_fun_camera.py): run on ONE worker.
Drives the certified Fun-camera pipeline from each eval record's own conditioning
(latent_condition -> y, camera matrices -> Plucker control, precomputed text
embeds), so no external image/trajectory files are needed. Optionally decodes the
record's ground-truth latents for side-by-side.

Checkpoint dispatch is whatever WanCheckpointer2_2_FunCamera.load_checkpoint does:
  - output_dir + step with a full-FT checkpoint   -> from_checkpoint
  - output_dir + step with a LoRA-only checkpoint -> PAI base + adapter overlay
  - empty/missing checkpoint dir                  -> PAI base (the pretrained model)

Usage (driven by examples/visualize_fun_camera.sh):
  python -m maxdiffusion.eval_fun_camera_visualize CONFIG \\
      eval_data_dir=gs://.../fun_camera_eval_set/encoded_seed1234_n16 \\
      output_dir=gs://.../viz run_name=base ...
"""

import os
import sys
import time
import json
from typing import Sequence

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.models.wan.camera_plucker import build_control_camera_latents
from maxdiffusion.trainers.wan_2_2_fun_camera_trainer import _build_first_frame_mask
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video


def _eval_feature_description():
  s = lambda: tf.io.FixedLenFeature([], tf.string)
  return {
      "latents": s(),
      "latent_condition": s(),
      "encoder_hidden_states": s(),
      "camera_extrinsic": s(),
      "camera_intrinsic": s(),
      "sample_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
      "caption": tf.io.FixedLenFeature([], tf.string, default_value=""),
  }


def read_eval_records(eval_dir, limit=0):
  """Read the curated eval TFRecords on host (small set), in file order."""
  files = sorted(tf.io.gfile.glob(os.path.join(eval_dir, "*.tfrec")))
  if not files:
    files = sorted(tf.io.gfile.glob(os.path.join(eval_dir, "*.tfrecord")))
  if not files:
    raise FileNotFoundError(f"No .tfrec/.tfrecord under eval_data_dir={eval_dir}")
  feat = _eval_feature_description()
  recs = []
  for raw in tf.data.TFRecordDataset(files):
    ex = tf.io.parse_single_example(raw, feat)
    p = lambda k: tf.io.parse_tensor(ex[k], out_type=tf.float32).numpy()
    sid = ex["sample_id"].numpy().decode("utf-8") or f"idx{len(recs):04d}"
    recs.append({
        "sample_id": sid,
        "caption": ex["caption"].numpy().decode("utf-8"),
        "latents": p("latents"),
        "latent_condition": p("latent_condition"),
        "encoder_hidden_states": p("encoder_hidden_states"),
        "camera_extrinsic": p("camera_extrinsic"),
        "camera_intrinsic": p("camera_intrinsic"),
    })
    if limit and len(recs) >= limit:
      break
  max_logging.log(f"[viz] read {len(recs)} eval record(s) from {eval_dir}")
  return recs


def _load_caption_manifest(path):
  """{sample_id: caption} from a caption manifest jsonl (raw text for re-encoding/display)."""
  lookup = {}
  with tf.io.gfile.GFile(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      r = json.loads(line)
      sid = str(r.get("sample_id", ""))
      if sid:
        lookup[sid] = r.get("caption", "")
  max_logging.log(f"[viz] loaded {len(lookup)} captions from {path}")
  return lookup


def _upload_file_to_gcs(gcs_dir, local_path):
  from google.cloud import storage

  path_without_scheme = gcs_dir.removeprefix("gs://")
  bucket_name, _, prefix = path_without_scheme.partition("/")
  blob = os.path.join(prefix, os.path.basename(local_path))
  storage.Client().bucket(bucket_name).blob(blob).upload_from_filename(local_path)


def run(config):
  load_start = time.perf_counter()
  step_sel = int(getattr(config, "eval_checkpoint_step", -1))
  step_sel = step_sel if step_sel >= 0 else None
  pipeline, _, step = WanCheckpointer2_2_FunCamera(config=config).load_checkpoint(step=step_sel)
  max_logging.log(f"[viz] checkpoint loaded (step={step}) in {time.perf_counter() - load_start:.1f}s")

  dtype = getattr(config, "activations_dtype", jnp.bfloat16)
  steps = int(getattr(config, "eval_num_inference_steps", 0)) or int(config.num_inference_steps)
  gs = float(getattr(config, "eval_guidance_scale", 1.0))
  save_gt = bool(getattr(config, "eval_save_gt_video", False))
  limit = int(getattr(config, "eval_num_generate_samples", 0))  # 0 => all
  tag = config.run_name
  # checkpoint_dir is forced to output_dir/run_name/checkpoints by pyconfig, so to
  # load another run's ckpt you set output_dir+run_name to THAT run. eval_output_dir
  # decouples where viz videos go from that.
  viz_out = str(getattr(config, "eval_output_dir", "") or "")
  if viz_out.startswith("gs://"):
    out_root = viz_out
  elif str(config.output_dir).startswith("gs://"):
    out_root = os.path.join(config.output_dir, tag)
  else:
    out_root = None
  local_tag = os.path.basename(viz_out.rstrip("/")) if viz_out else str(tag)
  local_dir = os.path.join("/tmp", "viz", local_tag)
  os.makedirs(local_dir, exist_ok=True)
  is_lead = jax.process_index() == 0

  records = read_eval_records(config.eval_data_dir, limit=limit)

  # Text conditioning mode:
  #   embedding (default) -> use the record's precomputed encoder_hidden_states (exact training cond)
  #   caption             -> re-encode the record's caption (or a manifest join) via umT5
  #   prompt              -> encode a single custom eval_prompt via umT5 (test text-following)
  text_mode = str(getattr(config, "eval_text_mode", "embedding")).lower()
  custom_prompt = str(getattr(config, "eval_prompt", "") or "")
  # Negative = the empty string by default (classic CFG unconditional), NOT WAN's
  # long default negative_prompt. Set eval_negative_prompt to override.
  neg_prompt = str(getattr(config, "eval_negative_prompt", "") or "")
  cap_path = str(getattr(config, "eval_caption_manifest", "") or "")
  cap_lookup = _load_caption_manifest(cap_path) if cap_path else {}
  if text_mode != "embedding":
    max_logging.log(f"[viz] text_mode={text_mode} (re-encoding raw text via umT5)")

  for rec in records:
    sid = rec["sample_id"]
    latent_condition = jnp.asarray(rec["latent_condition"], dtype=dtype)[None]  # [1,48,F,h,w]
    mask = _build_first_frame_mask(latent_condition)
    y_latents = jnp.concatenate([mask, latent_condition], axis=1).astype(dtype)
    control = build_control_camera_latents(
        jnp.asarray(rec["camera_extrinsic"])[None],
        jnp.asarray(rec["camera_intrinsic"])[None],
        height=config.height,
        width=config.width,
        moment_scale=float(getattr(config, "camera_moment_scale", 1.0)),
        dtype=dtype,
    )
    caption = rec["caption"] or cap_lookup.get(sid, "")
    text_kwargs = {}
    text_used = None
    if text_mode == "embedding":
      pe = jnp.asarray(rec["encoder_hidden_states"], dtype=dtype)[None]  # [1,512,4096]
      # Real umT5-encoded negative (not zeros) so CFG at guidance_scale>1 is correct
      # with the stored positive embedding. At gs<=1 the negative is simply unused.
      text_kwargs = {"prompt_embeds": pe, "negative_prompt": [neg_prompt]}
    else:
      text_used = custom_prompt if text_mode == "prompt" else caption
      if not text_used:
        raise ValueError(
            f"text_mode={text_mode} but no text for {sid} "
            f"(record caption empty; set eval_caption_manifest or eval_prompt)"
        )
      text_kwargs = {"prompt": [text_used], "negative_prompt": [neg_prompt]}  # umT5-encoded in-pipeline

    t0 = time.perf_counter()
    videos, trace = pipeline(
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        num_inference_steps=steps,
        guidance_scale=gs,
        y_latents=y_latents,
        control_camera_latents_input=control,
        use_kv_cache=config.use_kv_cache,
        **text_kwargs,
    )
    gen_s = time.perf_counter() - t0
    max_logging.log(f"[viz] {sid}: generated in {gen_s:.1f}s ({trace})")

    if not is_lead:
      continue
    gen_mp4 = os.path.join(local_dir, f"{sid}_gen.mp4")
    export_to_video(np.asarray(videos[0]), gen_mp4, fps=config.fps)
    written = [gen_mp4]

    if save_gt:
      gt = jnp.asarray(rec["latents"], dtype=dtype)[None]
      gt = pipeline._denormalize_latents(gt)
      gt_video = pipeline._decode_latents_to_video(gt)
      gt_mp4 = os.path.join(local_dir, f"{sid}_gt.mp4")
      export_to_video(np.asarray(gt_video[0]), gt_mp4, fps=config.fps)
      written.append(gt_mp4)

    ext = rec["camera_extrinsic"]
    meta = {
        "sample_id": sid,
        "caption": caption,
        "text_mode": text_mode,
        "text_used": text_used,  # null in embedding mode (the stored embedding was used)
        "checkpoint_tag": tag,
        "checkpoint_step": int(step) if step is not None else None,
        "height": int(config.height), "width": int(config.width), "num_frames": int(config.num_frames),
        "num_inference_steps": int(steps), "guidance_scale": float(gs),
        "gen_seconds": round(float(gen_s), 2),
        "camera_translation_delta": (ext[-1, :, 3] - ext[0, :, 3]).tolist(),
    }
    meta_path = os.path.join(local_dir, f"{sid}.json")
    with open(meta_path, "w") as f:
      json.dump(meta, f, indent=2)
    written.append(meta_path)

    if out_root is not None:
      for p in written:
        _upload_file_to_gcs(out_root, p)

  if is_lead:
    dest = out_root if out_root is not None else local_dir
    max_logging.log(f"[viz] done: {len(records)} sample(s) for tag '{tag}' -> {dest}")


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=False)
  config = pyconfig.config
  if not getattr(config, "eval_data_dir", ""):
    raise ValueError("Set eval_data_dir to the curated eval-set TFRecord directory.")
  with transformer_engine_context():
    run(config)


if __name__ == "__main__":
  app.run(main)
