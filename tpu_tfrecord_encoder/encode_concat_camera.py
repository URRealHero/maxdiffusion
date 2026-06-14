#!/usr/bin/env python3
"""Encode HM-World concat.mp4 + concat-caption + camera into MaxDiffusion WAN
TFRecords on a TPU VM.

Differs from encode_tv2v.py:
  * single video (concat.mp4) -> `latents` only (NO cond_latents)
  * adds per-frame camera control features:
      - camera_extrinsic [F, 3, 4]  (world->camera, local, frame 0 = identity, meters)
      - camera_intrinsic [F, 3, 3]  (pixels, from a fixed UE FOV)
  * `encoder_hidden_states` from the concat caption (unchanged path)

Camera conversion (HM-World camera.json -> Wan Fun camera controller format):
  cond_cam(77) + tgt_cam(77) c2w (Unreal world, cm) -> continuous 154 ->
  axis-permute [1,2,0,3] + flip Y + /100 (cm->m)  [HyDRA convention] ->
  re-base to global frame 0 -> store w2c [:3,:4].  Intrinsics from --camera-fov-deg
  (HM-World rendered with UE default => 90 deg horizontal).

Reuses the TPU/VAE/text/sharding machinery from encode_tv2v.py (same directory).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback

# Reuse the unchanged TPU machinery from the sibling encoder.
from encode_tv2v import (
    load_numpy,
    has_config_arg,
    infer_tpu_worker_index_from_hostname,
    prompt_clean,
    iter_jsonl,
    iter_manifest_records,
    copy_to_local_if_needed,
    read_video_frames,
    encode_prompts,
    encode_videos,
    bytes_feature,
    serialize_tensor,
    ShardedWriter,
    JsonlWriter,
    load_completed_sample_ids,
    wait_for_manual_file_barrier,
)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--config", required=True)
  p.add_argument("--config-arg", action="append", default=[])
  p.add_argument("--manifest", required=True, help="JSONL with sample_id + gs:// cond_video/tgt_video/camera (paths).")
  p.add_argument("--caption-manifest", required=True, help="JSONL with sample_id + caption (concat captions).")
  p.add_argument("--output-dir", required=True)
  p.add_argument("--caption-field", default="caption")
  # cond+tgt are read from GCS and concatenated in-memory (cond frames then tgt frames).
  p.add_argument("--cond-video-field", default="cond_video")
  p.add_argument("--tgt-video-field", default="tgt_video")
  p.add_argument("--camera-field", default="camera", help="Manifest field holding the camera.json path.")
  p.add_argument("--camera-fov-deg", type=float, default=90.0, help="Horizontal FOV (UE default 90).")
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--width", type=int, default=832)
  p.add_argument("--num-frames", type=int, default=153, help="Must be 4k+1 for the WAN VAE (154->153).")
  p.add_argument("--max-sequence-length", type=int, default=512)
  p.add_argument("--batch-size", type=int, default=1)
  p.add_argument("--records-per-shard", type=int, default=128)
  p.add_argument("--start", type=int, default=0)
  p.add_argument("--limit", type=int, default=None)
  p.add_argument("--sample-mode", choices=("uniform", "first"), default="uniform")
  p.add_argument("--video-decoder", choices=("cv2", "imageio"), default="cv2")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--host-index", type=int, default=None)
  p.add_argument("--host-count", type=int, default=None)
  p.add_argument("--print-tracebacks", action="store_true")
  p.add_argument("--run-id", default=None)
  p.add_argument("--final-file-barrier-timeout-seconds", type=int, default=7200)
  args = p.parse_args()
  if args.batch_size <= 0 or args.records_per_shard <= 0:
    p.error("--batch-size and --records-per-shard must be > 0")
  if (args.num_frames - 1) % 4 != 0:
    p.error(f"--num-frames must be 4k+1 for the WAN VAE, got {args.num_frames} (try 153)")
  if args.host_index is not None and args.host_count is None:
    p.error("--host-index requires --host-count")
  for item in args.config_arg:
    if "=" not in item:
      p.error(f"--config-arg must be key=value, got {item!r}")
  return args


def compute_indices(total: int, num_frames: int, sample_mode: str):
  np = load_numpy()
  if total >= num_frames:
    if sample_mode == "first":
      return np.arange(num_frames, dtype=np.int64)
    return np.linspace(0, total - 1, num_frames).round().astype(np.int64)
  idx = list(range(total)) + [total - 1] * (num_frames - total)
  return np.array(idx, dtype=np.int64)


def _read_frames_from_uri(uri, decoder, temp_dir):
  local_path = copy_to_local_if_needed(uri, temp_dir)
  try:
    return read_video_frames(local_path, decoder)
  finally:
    if uri.startswith("gs://"):
      try:
        os.remove(local_path)
      except FileNotFoundError:
        pass


def preprocess_concat(cond_uri, tgt_uri, height, width, num_frames, sample_mode, temp_dir, decoder):
  """Read cond + tgt, concatenate frames (cond then tgt), sample to num_frames.

  Returns ([1, C, F, H, W] float32, indices, total_frames). Avoids needing a
  physical concat.mp4 on GCS; the result is identical to encoding concat.mp4.
  """
  np = load_numpy()
  frames = _read_frames_from_uri(cond_uri, decoder, temp_dir) + _read_frames_from_uri(tgt_uri, decoder, temp_dir)
  total = len(frames)
  idx = compute_indices(total, num_frames, sample_mode)
  arrays = [np.asarray(frames[int(i)].resize((width, height)), dtype=np.float32) / 127.5 - 1.0 for i in idx]
  video = np.stack(arrays, axis=0)  # [F, H, W, C]
  return np.transpose(video, (3, 0, 1, 2))[None, ...], idx, total


def load_caption_map(caption_manifest, caption_field):
  caption_map = {}
  for record in iter_jsonl(caption_manifest):
    sample_id = record.get("sample_id")
    if sample_id is not None:
      caption_map[str(sample_id)] = record.get(caption_field)
  return caption_map


def _coordinate_transform(c2w):
  """HyDRA HM-World transform: Unreal world c2w (cm) -> camera-convention c2w (m)."""
  np = load_numpy()
  t = c2w[:, [1, 2, 0, 3]].copy()
  t[:3, 1] *= -1.0
  t[:3, 3] /= 100.0
  return t


def convert_camera(camera_json, fov_deg, width, height):
  """camera.json (cond_cam+tgt_cam c2w) -> (extrinsic[N,3,4] w2c local, intrinsic[N,3,3] px)."""
  np = load_numpy()
  cc, tc = camera_json["cond_cam"], camera_json["tgt_cam"]
  mats = [np.asarray(cc[str(i)], dtype=np.float64) for i in range(len(cc))]
  mats += [np.asarray(tc[str(i)], dtype=np.float64) for i in range(len(tc))]
  c2w = np.stack([_coordinate_transform(m) for m in mats], axis=0)  # [N,4,4]
  ref_w2c = np.linalg.inv(c2w[0])
  rel = ref_w2c[None] @ c2w          # relative c2w, frame 0 = identity
  w2c = np.linalg.inv(rel)           # local w2c
  extrinsic = w2c[:, :3, :4].astype(np.float32)
  fx = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
  K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  intrinsic = np.repeat(K[None], extrinsic.shape[0], axis=0)
  return extrinsic, intrinsic


def load_camera(camera_uri, fov_deg, width, height, indices, expected_total):
  np = load_numpy()
  import tensorflow as tf

  with tf.io.gfile.GFile(camera_uri, "r") as f:
    cam = json.load(f)
  ext_full, intr_full = convert_camera(cam, fov_deg, width, height)
  if ext_full.shape[0] != expected_total:
    raise ValueError(
        f"camera frames {ext_full.shape[0]} != video frames {expected_total} for {camera_uri}"
    )
  return ext_full[indices].astype(np.float32), intr_full[indices].astype(np.float32)


def make_example(latent, latent_condition, hidden_state, extrinsic, intrinsic,
                 sample_id=None, caption=None) -> bytes:
  import tensorflow as tf

  features = {
      "latents": bytes_feature(serialize_tensor(latent)),
      # Fun camera-control y-conditioning: VAE encode of [first_frame, zeros x (F-1)].
      # NOT derivable from `latents` (the VAE's temporal kernels mix the zero
      # padding differently than real frames). The 4ch mask is built in-trainer.
      "latent_condition": bytes_feature(serialize_tensor(latent_condition)),
      "encoder_hidden_states": bytes_feature(serialize_tensor(hidden_state)),
      "camera_extrinsic": bytes_feature(serialize_tensor(extrinsic)),
      "camera_intrinsic": bytes_feature(serialize_tensor(intrinsic)),
  }
  # Self-identifying records: embed sample_id (+ caption) so a curated eval set
  # can be named at generation time. Backward-compatible — the training parser
  # requests only the 5 tensors above and ignores these extra string features.
  if sample_id is not None:
    features["sample_id"] = bytes_feature(str(sample_id).encode("utf-8"))
  if caption is not None:
    features["caption"] = bytes_feature(str(caption).encode("utf-8"))
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def assert_finite(sample_id, *named):
  np = load_numpy()
  bad = [name for name, value in named if not np.isfinite(value).all()]
  if bad:
    raise ValueError(f"nonfinite encoded tensors for sample_id={sample_id}: {','.join(bad)}")


def main() -> int:
  args = parse_args()
  from maxdiffusion import pyconfig
  from maxdiffusion.pipelines.wan.wan_pipeline_2_2_dense import WanPipeline2_2_Dense
  import jax

  config_args = list(args.config_arg)
  manual_host_sharding = args.host_count is not None
  if manual_host_sharding and not has_config_arg(config_args, "skip_jax_distributed_system"):
    config_args.append("skip_jax_distributed_system=True")
  if manual_host_sharding:
    os.environ.setdefault("MAXDIFFUSION_FORCE_LOCAL_DEVICE_MESH", "1")

  pyconfig.initialize(["encode_concat_camera.py", args.config] + config_args)
  config = pyconfig.config

  height, width, num_frames = args.height, args.width, args.num_frames
  if height % 16 != 0 or width % 16 != 0:
    raise ValueError(f"height/width must be divisible by 16, got {height}x{width}")

  if manual_host_sharding:
    process_index = args.host_index if args.host_index is not None else infer_tpu_worker_index_from_hostname()
    process_count = args.host_count
  else:
    process_index, process_count = jax.process_index(), jax.process_count()

  print(f"Encoder {process_index}/{process_count}: {width}x{height} frames={num_frames} fov={args.camera_fov_deg}", flush=True)

  caption_map = load_caption_map(args.caption_manifest, args.caption_field)
  print(f"Process {process_index}: loaded {len(caption_map)} captions", flush=True)

  out_prefix = args.output_dir.rstrip("/")
  completed = load_completed_sample_ids(out_prefix, process_index) if args.resume else set()
  run_id = args.run_id or os.environ.get("TPU_TFRECORD_ENCODER_RUN_ID") or str(os.getpid())
  metadata_path = f"{out_prefix}/metadata_host_{process_index:03d}_run_{run_id}.jsonl"
  failures_path = f"{out_prefix}/failures_host_{process_index:03d}_run_{run_id}.jsonl"

  assigned = []
  for global_index, record in iter_manifest_records(args.manifest, args.start, args.limit):
    if (global_index - args.start) % process_count != process_index:
      continue
    sample_id = str(record.get("sample_id") or global_index)
    if sample_id in completed:
      continue
    assigned.append((global_index, sample_id, record))
  print(f"Process {process_index}: assigned {len(assigned)} records", flush=True)
  if args.dry_run:
    return 0

  pipeline = WanPipeline2_2_Dense.from_pretrained(config, load_transformer=False)
  if pipeline.text_encoder is not None:
    pipeline.text_encoder.eval()

  writer = ShardedWriter(args.output_dir, args.records_per_shard, process_index, run_id)
  metadata_writer = JsonlWriter(metadata_path)
  failures_writer = JsonlWriter(failures_path)

  with tempfile.TemporaryDirectory(prefix=f"wan_concat_cam_{process_index}_") as temp_dir:
    np = load_numpy()

    def mark_failed(records, error):
      for _, sample_id, record in records:
        failures_writer.write({"sample_id": sample_id, "error": str(error),
                               "process_index": process_index, "tfrec_path": writer.current_path})

    def flush_batch(records):
      if not records:
        return
      videos, prompts, cams = [], [], []
      for _, sample_id, record in records:
        cond_uri = record.get(args.cond_video_field)
        tgt_uri = record.get(args.tgt_video_field)
        camera_uri = record.get(args.camera_field)
        caption = caption_map.get(sample_id)
        if not cond_uri or not tgt_uri:
          raise ValueError(f"missing {args.cond_video_field}/{args.tgt_video_field}")
        if not camera_uri:
          raise ValueError(f"missing camera field {args.camera_field}")
        if not prompt_clean(str(caption or "")):
          raise ValueError(f"no caption for sample_id={sample_id} in caption manifest")
        video, idx, total = preprocess_concat(cond_uri, tgt_uri, height, width, num_frames, args.sample_mode, temp_dir, args.video_decoder)
        videos.append(video)
        cams.append(load_camera(camera_uri, args.camera_fov_deg, width, height, idx, total))
        prompts.append(str(caption))

      video_batch = np.concatenate(videos, axis=0)
      latents = encode_videos(pipeline, video_batch)
      # Fun camera-control conditioning: encode the MASKED video (first frame
      # kept, rest zeroed) through the exact same VAE + normalization path.
      masked_batch = np.concatenate(
          [video_batch[:, :, :1], np.zeros_like(video_batch[:, :, 1:])], axis=2)
      latent_conditions = encode_videos(pipeline, masked_batch)
      hidden_states = encode_prompts(pipeline, prompts, args.max_sequence_length)

      for (_, sample_id, record), latent, latent_condition, hidden, (ext, intr) in zip(
          records, latents, latent_conditions, hidden_states, cams):
        assert_finite(sample_id, ("latents", latent), ("latent_condition", latent_condition),
                      ("encoder_hidden_states", hidden),
                      ("camera_extrinsic", ext), ("camera_intrinsic", intr))
        writer.write(make_example(latent, latent_condition, hidden, ext, intr,
                                   sample_id=sample_id, caption=caption_map.get(sample_id)))
        metadata_writer.write({
            "sample_id": sample_id,
            "caption": caption_map.get(sample_id),
            "latent_shape": list(latent.shape),
            "latent_condition_shape": list(latent_condition.shape),
            "encoder_hidden_states_shape": list(hidden.shape),
            "camera_extrinsic_shape": list(ext.shape),
            "camera_intrinsic_shape": list(intr.shape),
            "height": height, "width": width, "num_frames": num_frames,
            "camera_fov_deg": args.camera_fov_deg,
            "process_index": process_index, "tfrec_path": writer.current_path,
        })
        print(f"encoded sample_id={sample_id} latent={latent.shape} ext={ext.shape}", flush=True)

    def flush_or_split(records, label):
      try:
        flush_batch(records)
      except Exception as exc:
        if len(records) == 1:
          mark_failed(records, exc)
          print(f"failed sample_id={records[0][1]}: {exc}", flush=True)
          if args.print_tracebacks:
            traceback.print_exc()
          return
        print(f"failed {label} ({len(records)}); retrying one by one: {exc}", flush=True)
        for r in records:
          flush_or_split([r], "single")

    batch = []
    for item in assigned:
      batch.append(item)
      if len(batch) >= args.batch_size:
        flush_or_split(batch, "batch")
        batch = []
    if batch:
      flush_or_split(batch, "final batch")

  writer.close()
  metadata_writer.close()
  failures_writer.close()

  if manual_host_sharding:
    print(f"Process {process_index}: wrote {writer.total_records}; manual file barrier", flush=True)
    wait_for_manual_file_barrier(args.output_dir, run_id, process_index, process_count, args.final_file_barrier_timeout_seconds)
  else:
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices("tpu_concat_camera_encoder_done")
    print(f"Process {process_index}: final barrier complete", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
