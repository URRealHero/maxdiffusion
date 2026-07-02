#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import traceback
from typing import Iterable

np = None


def load_numpy():
  global np
  if np is None:
    import numpy as np_module
    np = np_module
  return np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Encode raw TV2V videos + captions into MaxDiffusion WAN TFRecords on a TPU VM."
  )
  parser.add_argument("--config", required=True, help="MaxDiffusion WAN config YAML, e.g. base_wan_ti2v_5b.yml")
  parser.add_argument(
      "--config-arg",
      action="append",
      default=[],
      help="Extra MaxDiffusion config override in key=value form. Can be repeated.",
  )
  parser.add_argument("--manifest", required=True, help="JSONL manifest path, local or gs://")
  parser.add_argument("--output-dir", required=True, help="Output directory, local or gs://")
  parser.add_argument("--caption-field", default="caption")
  parser.add_argument("--condition-video-field", default="cond_video")
  parser.add_argument("--target-video-field", default="tgt_video")
  parser.add_argument("--condition-output-field", default="cond_latents")
  parser.add_argument(
      "--with-camera",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Also emit HyDRA-convention cam_emb_con/cam_emb_tgt [N_lat,12] from each sample's "
      "camera.json (relative to the TARGET frame-0 pose). Missing/bad camera.json is logged + "
      "skipped per-sample (the record is still encoded). Disable with --no-with-camera.",
  )
  parser.add_argument(
      "--camera-field",
      default="camera",
      help="Manifest field holding the camera.json path. If absent from a record, the path is "
      "derived as <target_video_dir>/camera.json.",
  )
  parser.add_argument("--height", type=int, default=None)
  parser.add_argument("--width", type=int, default=None)
  parser.add_argument("--num-frames", type=int, default=None)
  parser.add_argument("--max-sequence-length", type=int, default=512)
  parser.add_argument("--batch-size", type=int, default=1, help="Encoding microbatch per host.")
  parser.add_argument("--records-per-shard", type=int, default=128)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--sample-mode", choices=("uniform", "first"), default="uniform")
  parser.add_argument(
      "--video-decoder",
      choices=("cv2", "imageio"),
      default="cv2",
      help="Video decoder backend. cv2 avoids imageio/ffmpeg subprocess fork warnings after JAX starts.",
  )
  parser.add_argument("--resume", action="store_true", help="Skip sample_ids listed in this host's metadata sidecar.")
  parser.add_argument("--dry-run", action="store_true", help="Validate manifest sharding without loading models.")
  parser.add_argument(
      "--host-index",
      type=int,
      default=None,
      help="Manual shard index for non-JAX-distributed encoding. Defaults to parsing TPU worker id from hostname.",
  )
  parser.add_argument(
      "--host-count",
      type=int,
      default=None,
      help="Manual shard count for non-JAX-distributed encoding, e.g. 16 for v5p-128 or 64 for v6e-256.",
  )
  parser.add_argument(
      "--print-tracebacks",
      action="store_true",
      help="Print full exception tracebacks for failed records.",
  )
  parser.add_argument(
      "--run-id",
      default=None,
      help="Common run id shared by all hosts. Needed for the manual final file barrier.",
  )
  parser.add_argument(
      "--final-file-barrier-timeout-seconds",
      type=int,
      default=7200,
      help="When --host-count is set, wait this long for all hosts to write done markers. Use <=0 to disable.",
  )
  args = parser.parse_args()

  if args.batch_size <= 0:
    parser.error("--batch-size must be > 0")
  if args.records_per_shard <= 0:
    parser.error("--records-per-shard must be > 0")
  if args.condition_output_field in {"latents", "encoder_hidden_states"}:
    parser.error("--condition-output-field must not collide with latents or encoder_hidden_states")
  if args.host_index is not None and args.host_count is None:
    parser.error("--host-index requires --host-count")
  if args.host_count is not None:
    if args.host_count <= 0:
      parser.error("--host-count must be > 0")
    if args.host_index is not None and (args.host_index < 0 or args.host_index >= args.host_count):
      parser.error("--host-index must satisfy 0 <= host-index < host-count")
  for item in args.config_arg:
    if "=" not in item:
      parser.error(f"--config-arg must be key=value, got {item!r}")
  return args


def has_config_arg(config_args: list[str], key: str) -> bool:
  return any(item.split("=", 1)[0] == key for item in config_args)


def infer_tpu_worker_index_from_hostname() -> int:
  hostname = socket.gethostname()
  match = re.search(r"-w-(\d+)$", hostname)
  if match:
    return int(match.group(1))
  match = re.search(r"-w(\d+)$", hostname)
  if match:
    return int(match.group(1))
  raise ValueError(
      "Could not infer TPU worker index from hostname "
      f"{hostname!r}; pass --host-index explicitly."
  )


def prompt_clean(text: str) -> str:
  text = html.unescape(html.unescape(text or ""))
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def iter_jsonl(path: str) -> Iterable[dict]:
  import tensorflow as tf

  with tf.io.gfile.GFile(path, "r") as f:
    for line_number, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc


def iter_manifest_records(path: str, start: int, limit: int | None) -> Iterable[tuple[int, dict]]:
  yielded = 0
  for index, record in enumerate(iter_jsonl(path)):
    if index < start:
      continue
    if limit is not None and yielded >= limit:
      break
    yield index, record
    yielded += 1


def copy_to_local_if_needed(uri: str, temp_dir: str) -> str:
  if not uri.startswith("gs://"):
    return uri

  import tensorflow as tf

  suffix = Path(uri).suffix or ".mp4"
  local_path = os.path.join(temp_dir, f"video_{abs(hash(uri))}{suffix}")
  if not os.path.exists(local_path):
    tf.io.gfile.copy(uri, local_path, overwrite=True)
  return local_path


def read_video_frames(video_path: str, decoder: str):
  from PIL import Image

  frames = []
  if decoder == "cv2":
    try:
      import cv2
    except ImportError as exc:
      raise ImportError(
          "--video-decoder=cv2 requires opencv-python-headless in the TPU VM venv. "
          "Install it with: python -m pip install opencv-python-headless"
      ) from exc

    cap = cv2.VideoCapture(video_path)
    try:
      while True:
        ok, frame = cap.read()
        if not ok:
          break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
      cap.release()
  else:
    try:
      import imageio.v2 as imageio
    except ImportError as exc:
      raise ImportError(
          "--video-decoder=imageio requires imageio and pillow in the TPU VM environment."
      ) from exc

    with imageio.get_reader(video_path) as reader:
      for frame in reader:
        frames.append(Image.fromarray(frame).convert("RGB"))

  if not frames:
    raise ValueError(f"video has no decodable frames: {video_path}")
  return frames


def sample_frames(frames, num_frames: int, sample_mode: str):
  np = load_numpy()
  if len(frames) >= num_frames:
    if sample_mode == "first":
      indices = np.arange(num_frames)
    else:
      indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    return [frames[int(i)] for i in indices]

  padded = list(frames)
  padded.extend([frames[-1]] * (num_frames - len(frames)))
  return padded


def preprocess_video(
    video_uri: str,
    height: int,
    width: int,
    num_frames: int,
    sample_mode: str,
    temp_dir: str,
    decoder: str,
) -> np.ndarray:
  np = load_numpy()
  local_path = copy_to_local_if_needed(video_uri, temp_dir)
  try:
    frames = sample_frames(read_video_frames(local_path, decoder), num_frames, sample_mode)
  finally:
    if video_uri.startswith("gs://"):
      try:
        os.remove(local_path)
      except FileNotFoundError:
        pass

  arrays = []
  for frame in frames:
    frame = frame.resize((width, height))
    array = np.asarray(frame, dtype=np.float32) / 127.5 - 1.0
    arrays.append(array)

  video = np.stack(arrays, axis=0)  # [F, H, W, C]
  return np.transpose(video, (3, 0, 1, 2))[None, ...]  # [1, C, F, H, W]


def encode_prompts(pipeline, prompts: list[str], max_sequence_length: int) -> np.ndarray:
  embeds = pipeline._get_t5_prompt_embeds(
      prompt=[prompt_clean(prompt) for prompt in prompts],
      num_videos_per_prompt=1,
      max_sequence_length=max_sequence_length,
  )
  return embeds.detach().float().cpu().numpy().astype(np.float32)


def materialize_addressable_array(array) -> np.ndarray:
  """Copy a JAX array to NumPy using only this process's addressable shards."""
  np = load_numpy()
  import jax

  if not hasattr(array, "addressable_shards"):
    return np.asarray(jax.device_get(array), dtype=np.float32)

  shards = list(array.addressable_shards)
  if not shards:
    raise ValueError("JAX array has no addressable shards on this process")

  result = np.empty(array.shape, dtype=np.float32)
  for shard in shards:
    result[shard.index] = np.asarray(jax.device_get(shard.data), dtype=np.float32)
  return result


def _debug_jax_array(name: str, value) -> None:
  if os.environ.get("TPU_TFRECORD_ENCODER_DEBUG_VAE", "0").lower() not in {"1", "true", "yes"}:
    return

  import jax
  import jax.numpy as jnp

  finite = jnp.mean(jnp.isfinite(value).astype(jnp.float32))
  min_value = jnp.min(value)
  max_value = jnp.max(value)
  jax.debug.print(
      f"encoder debug {name}: shape={value.shape} finite={{finite}} min={{min}} max={{max}}",
      finite=finite,
      min=min_value,
      max=max_value,
  )


def _vae_redundant_axis_size(pipeline) -> int:
  mesh = getattr(pipeline, "vae_mesh", None)
  shape = getattr(mesh, "shape", {})
  try:
    return max(1, int(shape.get("redundant", 1)))
  except AttributeError:
    return 1


def _pad_videos_for_vae_mesh(pipeline, videos: np.ndarray) -> tuple[np.ndarray, int]:
  """Pad VAE encode batches so the batch axis satisfies VAE mesh sharding."""
  original_batch = int(videos.shape[0])
  redundant = _vae_redundant_axis_size(pipeline)
  if original_batch <= 0 or redundant <= 1 or original_batch % redundant == 0:
    return videos, original_batch

  np = load_numpy()
  pad = redundant - (original_batch % redundant)
  padded = np.concatenate([videos, np.repeat(videos[-1:], pad, axis=0)], axis=0)
  if os.environ.get("TPU_TFRECORD_ENCODER_DEBUG_VAE", "0").lower() in {"1", "true", "yes"}:
    print(
        f"encoder debug padded VAE batch {original_batch}->{padded.shape[0]} "
        f"for vae_mesh redundant={redundant}",
        flush=True,
    )
  return padded, original_batch


def encode_videos(pipeline, videos: np.ndarray) -> np.ndarray:
  import jax.numpy as jnp
  from flax.linen import partitioning as nn_partitioning

  videos, original_batch = _pad_videos_for_vae_mesh(pipeline, videos)
  video = jnp.asarray(videos, dtype=getattr(pipeline.vae, "dtype", jnp.float32))
  _debug_jax_array("input_video", video)

  with pipeline.vae_mesh, nn_partitioning.axis_rules(pipeline.vae_logical_axis_rules):
    encoded = pipeline.vae.encode(video, pipeline.vae_cache)[0].mode()

  _debug_jax_array("encoded_mode_before_normalize", encoded)
  latents_mean = jnp.array(pipeline.vae.latents_mean).reshape(1, 1, 1, 1, pipeline.vae.z_dim)
  latents_std = jnp.array(pipeline.vae.latents_std).reshape(1, 1, 1, 1, pipeline.vae.z_dim)
  _debug_jax_array("latents_mean", latents_mean)
  _debug_jax_array("latents_std", latents_std)
  latents = (encoded - latents_mean) / latents_std  # [B, F, H, W, C]
  _debug_jax_array("latents_after_normalize_channel_last", latents)
  latents = jnp.transpose(latents, (0, 4, 1, 2, 3))  # [B, C, F, H, W]
  latents = latents.astype(jnp.float32)
  _debug_jax_array("latents_after_transpose", latents)
  latents.block_until_ready()
  materialized = materialize_addressable_array(latents)
  materialized = materialized[:original_batch]
  if os.environ.get("TPU_TFRECORD_ENCODER_DEBUG_VAE", "0").lower() in {"1", "true", "yes"}:
    np = load_numpy()
    print(
        "encoder debug materialized_latents: "
        f"shape={materialized.shape} finite={np.isfinite(materialized).mean()} "
        f"min={np.nanmin(materialized)} max={np.nanmax(materialized)}",
        flush=True,
    )
  return materialized


def bytes_feature(value: bytes):
  import tensorflow as tf

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_tensor(array: np.ndarray) -> bytes:
  import tensorflow as tf

  return tf.io.serialize_tensor(tf.convert_to_tensor(array, dtype=tf.float32)).numpy()


def _apply_coordinate_transform(c2w: np.ndarray) -> np.ndarray:
  """HyDRA _apply_coordinate_transform (infer_hydra.py): Unreal-world c2w (cm) ->
  HyDRA camera-convention c2w (m). Permute columns [1,2,0,3], flip Y, cm->m."""
  np = load_numpy()
  t = np.array(c2w, dtype=np.float64)
  t = t[:, [1, 2, 0, 3]]
  t[:3, 1] *= -1.0
  t[:3, 3] /= 100.0
  return t


def _cam_dict_to_pose_list(cam_dict: dict, name: str, num_frames: int) -> list:
  """Read {"0":4x4, ..., "N-1":4x4} into an ordered list of 4x4 c2w arrays, padded to
  num_frames by repeating the last available pose (mirrors last-frame video padding in
  sample_frames)."""
  np = load_numpy()
  n_avail = len(cam_dict)
  mats = []
  for i in range(n_avail):
    key = str(i)
    if key not in cam_dict:
      raise ValueError(f"{name} missing key {key!r}")
    m = np.asarray(cam_dict[key], dtype=np.float64)
    if m.shape != (4, 4):
      raise ValueError(f"{name}[{key}] must be [4,4], got {tuple(m.shape)}")
    mats.append(m)
  if not mats:
    raise ValueError(f"{name} is empty")
  if num_frames > n_avail:
    mats.extend([mats[-1]] * (num_frames - n_avail))
  return mats


def load_hydra_cam_emb(camera_json_path: str, num_frames: int):
  """Compute HyDRA-convention relative camera embeddings from a HM-World camera.json.

  Replicates infer_hydra.py _apply_coordinate_transform + _compute_relative EXACTLY:
    cam_idx = list(range(num_frames))[::4]              # N_lat indices (77 -> 20, 81 -> 21)
    ref_c2w = transform(tgt_cam[0]); ref_w2c = inv(ref_c2w)
    for each stream (cond, tgt), each idx: rel = ref_w2c @ transform(pose[idx]);
                                           append rel[:3,:4].reshape(-1)  # (12,)
  BOTH streams are made relative to the TARGET frame-0 pose. Returns
  (cam_emb_con[N_lat,12], cam_emb_tgt[N_lat,12]) float32.
  """
  np = load_numpy()
  import tensorflow as tf

  with tf.io.gfile.GFile(camera_json_path, "r") as f:
    data = json.load(f)
  if "cond_cam" not in data or "tgt_cam" not in data:
    raise ValueError(f"camera.json missing cond_cam/tgt_cam: {camera_json_path}")

  cond_poses = _cam_dict_to_pose_list(data["cond_cam"], "cond_cam", num_frames)
  tgt_poses = _cam_dict_to_pose_list(data["tgt_cam"], "tgt_cam", num_frames)

  cam_idx = list(range(num_frames))[::4]  # HyDRA range(77)[::4] -> 20
  ref_c2w = _apply_coordinate_transform(tgt_poses[0])
  ref_w2c = np.linalg.inv(ref_c2w)

  def to_rel(poses):
    rel_list = []
    for idx in cam_idx:
      c2w = _apply_coordinate_transform(poses[idx])
      rel = ref_w2c @ c2w
      rel_list.append(rel[:3, :4].reshape(-1))
    return np.stack(rel_list, axis=0).astype(np.float32)  # [N_lat, 12]

  cam_emb_con = to_rel(cond_poses)
  cam_emb_tgt = to_rel(tgt_poses)
  if not (np.isfinite(cam_emb_con).all() and np.isfinite(cam_emb_tgt).all()):
    raise ValueError(f"nonfinite cam_emb from {camera_json_path}")
  return cam_emb_con, cam_emb_tgt


def resolve_camera_uri(record: dict, camera_field: str, target_video_field: str, condition_video_field: str) -> str | None:
  """camera.json path from the manifest camera_field if present, else derived from the
  target (or condition) video's directory as <video_dir>/camera.json."""
  cam = record.get(camera_field)
  if cam:
    return str(cam)
  video = record.get(target_video_field) or record.get(condition_video_field)
  if not video:
    return None
  return str(video).rsplit("/", 1)[0] + "/camera.json"


def make_example(
    latent: np.ndarray,
    cond_latent: np.ndarray,
    hidden_state: np.ndarray,
    condition_output_field: str,
    cam_emb_con: np.ndarray | None = None,
    cam_emb_tgt: np.ndarray | None = None,
    sample_id: str | None = None,
) -> bytes:
  load_numpy()
  import tensorflow as tf

  features = {
      "latents": bytes_feature(serialize_tensor(latent)),
      condition_output_field: bytes_feature(serialize_tensor(cond_latent)),
      "encoder_hidden_states": bytes_feature(serialize_tensor(hidden_state)),
  }
  if cam_emb_con is not None and cam_emb_tgt is not None:
    features["cam_emb_con"] = bytes_feature(serialize_tensor(cam_emb_con))
    features["cam_emb_tgt"] = bytes_feature(serialize_tensor(cam_emb_tgt))
  # Self-identifying records: embed sample_id so train/eval splits can be selected
  # by sid at load time (exclude the held-out set for training, keep it for eval)
  # without re-encoding. Mirrors encode_concat_camera.make_example.
  if sample_id is not None:
    features["sample_id"] = bytes_feature(str(sample_id).encode("utf-8"))
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def assert_finite_encoded_record(sample_id: str, latent: np.ndarray, cond_latent: np.ndarray, hidden_state: np.ndarray) -> None:
  np = load_numpy()
  checks = {
      "latents": latent,
      "cond_latents": cond_latent,
      "encoder_hidden_states": hidden_state,
  }
  bad = [name for name, value in checks.items() if not np.isfinite(value).all()]
  if bad:
    raise ValueError(f"nonfinite encoded tensors for sample_id={sample_id}: {','.join(bad)}")


class ShardedWriter:
  def __init__(self, output_dir: str, records_per_shard: int, host_index: int, run_id: str):
    import tensorflow as tf

    self.tf = tf
    self.output_dir = output_dir.rstrip("/")
    self.records_per_shard = records_per_shard
    self.host_index = host_index
    self.run_id = run_id
    self.shard_index = 0
    self.records_in_shard = 0
    self.total_records = 0
    self.writer = None
    self.current_path = None
    self.tf.io.gfile.makedirs(self.output_dir)

  def _open(self):
    path = f"{self.output_dir}/host_{self.host_index:03d}_run_{self.run_id}_file_{self.shard_index:06d}.tfrec"
    self.current_path = path
    print(f"Writing shard: {path}", flush=True)
    self.writer = self.tf.io.TFRecordWriter(path)

  def write(self, example: bytes):
    if self.writer is None:
      self._open()
    self.writer.write(example)
    self.records_in_shard += 1
    self.total_records += 1
    if self.records_in_shard >= self.records_per_shard:
      self.close_current()
      self.shard_index += 1

  def close_current(self):
    if self.writer is not None:
      self.writer.close()
      self.writer = None
      self.current_path = None
      self.records_in_shard = 0

  def close(self):
    self.close_current()


def load_completed_sample_ids(output_dir: str, host_index: int) -> set[str]:
  import tensorflow as tf

  pattern = f"{output_dir.rstrip('/')}/metadata_host_{host_index:03d}*.jsonl"
  completed = set()
  for path in tf.io.gfile.glob(pattern):
    for record in iter_jsonl(path):
      sample_id = record.get("sample_id")
      if sample_id:
        completed.add(str(sample_id))
  return completed


class JsonlWriter:
  def __init__(self, path: str):
    import tensorflow as tf

    self.path = path
    self.file = tf.io.gfile.GFile(path, "w")

  def write(self, record: dict) -> None:
    self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
    self.file.flush()

  def close(self) -> None:
    self.file.close()


def wait_for_manual_file_barrier(output_dir: str, run_id: str, host_index: int, host_count: int, timeout_seconds: int) -> None:
  if timeout_seconds <= 0:
    return

  import time
  import tensorflow as tf

  output_dir = output_dir.rstrip("/")
  done_path = f"{output_dir}/_done_{run_id}_host_{host_index:03d}.json"
  with tf.io.gfile.GFile(done_path, "w") as f:
    f.write(json.dumps({"run_id": run_id, "host_index": host_index, "host_count": host_count}) + "\n")

  expected = [f"{output_dir}/_done_{run_id}_host_{i:03d}.json" for i in range(host_count)]
  deadline = time.time() + timeout_seconds
  last_seen = -1
  while True:
    seen = sum(1 for path in expected if tf.io.gfile.exists(path))
    if seen == host_count:
      print(f"Process {host_index}: manual file barrier complete for run_id={run_id}", flush=True)
      return
    if seen != last_seen:
      print(f"Process {host_index}: manual file barrier waiting {seen}/{host_count} for run_id={run_id}", flush=True)
      last_seen = seen
    if time.time() > deadline:
      raise TimeoutError(
          f"Timed out waiting for manual file barrier run_id={run_id}: saw {seen}/{host_count} done files"
      )
    time.sleep(10)


def main() -> int:
  args = parse_args()

  from maxdiffusion import pyconfig
  from maxdiffusion.pipelines.wan.wan_pipeline_2_2_dense import WanPipeline2_2_Dense
  import jax
  import tensorflow as tf

  config_args = list(args.config_arg)
  manual_host_sharding = args.host_count is not None
  if manual_host_sharding and not has_config_arg(config_args, "skip_jax_distributed_system"):
    config_args.append("skip_jax_distributed_system=True")
  if manual_host_sharding:
    os.environ.setdefault("MAXDIFFUSION_FORCE_LOCAL_DEVICE_MESH", "1")

  pyconfig.initialize(["encode_tv2v.py", args.config] + config_args)
  config = pyconfig.config

  height = args.height if args.height is not None else config.height
  width = args.width if args.width is not None else config.width
  num_frames = args.num_frames if args.num_frames is not None else config.num_frames

  if height % 16 != 0 or width % 16 != 0:
    raise ValueError(f"height/width should be divisible by 16 for WAN VAE, got {height}x{width}")

  if manual_host_sharding:
    process_index = args.host_index
    process_count = args.host_count
    if process_index is None:
      process_index = infer_tpu_worker_index_from_hostname()
  else:
    process_index = jax.process_index()
    process_count = jax.process_count()

  print(
      f"Encoder process {process_index}/{process_count}: height={height} width={width} frames={num_frames} "
      f"manual_host_sharding={manual_host_sharding}",
      flush=True,
  )

  metadata_read_prefix = args.output_dir.rstrip("/")
  completed_sample_ids = load_completed_sample_ids(metadata_read_prefix, process_index) if args.resume else set()
  run_id = args.run_id or os.environ.get("TPU_TFRECORD_ENCODER_RUN_ID") or str(os.getpid())
  metadata_path = f"{metadata_read_prefix}/metadata_host_{process_index:03d}_run_{run_id}.jsonl"
  failures_path = f"{metadata_read_prefix}/failures_host_{process_index:03d}_run_{run_id}.jsonl"

  assigned = []
  for global_index, record in iter_manifest_records(args.manifest, args.start, args.limit):
    if (global_index - args.start) % process_count != process_index:
      continue
    sample_id = str(record.get("sample_id") or global_index)
    if sample_id in completed_sample_ids:
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

  with tempfile.TemporaryDirectory(prefix=f"wan_tv2v_encode_{process_index}_") as temp_dir:
    batch = []

    def mark_failed(records, error: Exception | str) -> None:
      for _, sample_id, record in records:
        failures_writer.write(
            {
                "sample_id": sample_id,
                "target_video": record.get(args.target_video_field),
                "condition_video": record.get(args.condition_video_field),
                "error": str(error),
                "process_index": process_index,
                "tfrec_path": writer.current_path,
            },
        )

    def flush_batch(records):
      if not records:
        return

      np = load_numpy()
      target_videos = []
      condition_videos = []
      prompts = []
      cam_embs = []
      for _, sample_id, record in records:
        target_uri = record.get(args.target_video_field)
        condition_uri = record.get(args.condition_video_field)
        caption = record.get(args.caption_field)
        if not target_uri:
          raise ValueError(f"missing target field {args.target_video_field}")
        if not condition_uri:
          raise ValueError(f"missing condition field {args.condition_video_field}")
        if not prompt_clean(str(caption or "")):
          raise ValueError(f"missing/empty caption field {args.caption_field}")

        target_videos.append(
            preprocess_video(str(target_uri), height, width, num_frames, args.sample_mode, temp_dir, args.video_decoder)
        )
        condition_videos.append(
            preprocess_video(str(condition_uri), height, width, num_frames, args.sample_mode, temp_dir, args.video_decoder)
        )
        prompts.append(str(caption))

        # HyDRA camera embeddings are OPTIONAL: a missing/bad camera.json is logged and
        # skipped for this sample only, so the record is still encoded (no crash).
        cam_emb_con = cam_emb_tgt = None
        if args.with_camera:
          camera_uri = resolve_camera_uri(
              record, args.camera_field, args.target_video_field, args.condition_video_field
          )
          if not camera_uri:
            print(f"camera skipped for sample_id={sample_id}: no camera path resolvable", flush=True)
          else:
            try:
              cam_emb_con, cam_emb_tgt = load_hydra_cam_emb(camera_uri, num_frames)
            except Exception as cam_exc:  # noqa: BLE001 - never fail the record on camera
              cam_emb_con = cam_emb_tgt = None
              print(f"camera skipped for sample_id={sample_id}: {cam_exc}", flush=True)
        cam_embs.append((cam_emb_con, cam_emb_tgt))

      target_array = np.concatenate(target_videos, axis=0)
      condition_array = np.concatenate(condition_videos, axis=0)
      latents = encode_videos(pipeline, target_array)
      cond_latents = encode_videos(pipeline, condition_array)
      hidden_states = encode_prompts(pipeline, prompts, args.max_sequence_length)

      encoded_records = list(zip(records, latents, cond_latents, hidden_states, cam_embs))
      for (_, sample_id, _), latent, cond_latent, hidden_state, _ in encoded_records:
        assert_finite_encoded_record(sample_id, latent, cond_latent, hidden_state)

      for (_, sample_id, record), latent, cond_latent, hidden_state, (cam_emb_con, cam_emb_tgt) in encoded_records:
        writer.write(
            make_example(
                latent, cond_latent, hidden_state, args.condition_output_field, cam_emb_con, cam_emb_tgt,
                sample_id=sample_id,
            )
        )
        meta = {
            "sample_id": sample_id,
            "caption": record.get(args.caption_field),
            "target_video": record.get(args.target_video_field),
            "condition_video": record.get(args.condition_video_field),
            "latent_shape": list(latent.shape),
            "condition_latent_shape": list(cond_latent.shape),
            "encoder_hidden_states_shape": list(hidden_state.shape),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "process_index": process_index,
            "tfrec_path": writer.current_path,
        }
        if cam_emb_con is not None and cam_emb_tgt is not None:
          meta["cam_emb_con_shape"] = list(cam_emb_con.shape)
          meta["cam_emb_tgt_shape"] = list(cam_emb_tgt.shape)
        metadata_writer.write(meta)
        cam_note = "" if cam_emb_con is None else f" cam_emb={cam_emb_con.shape}"
        print(f"encoded sample_id={sample_id} latent_shape={latent.shape}{cam_note}", flush=True)

    def flush_batch_or_split(records, label: str) -> None:
      try:
        flush_batch(records)
      except Exception as exc:
        if len(records) == 1:
          mark_failed(records, exc)
          _, sample_id, _ = records[0]
          print(f"failed sample_id={sample_id}: {exc}", flush=True)
          if args.print_tracebacks:
            traceback.print_exc()
          return
        print(f"failed {label} with {len(records)} records; retrying records one by one: {exc}", flush=True)
        for record in records:
          flush_batch_or_split([record], "single record")

    for item in assigned:
      batch.append(item)
      if len(batch) >= args.batch_size:
        flush_batch_or_split(batch, "batch")
        batch = []

    if batch:
      flush_batch_or_split(batch, "final batch")

  writer.close()
  metadata_writer.close()
  failures_writer.close()
  if manual_host_sharding:
    print(f"Process {process_index}: wrote {writer.total_records} records; waiting at manual file barrier", flush=True)
    wait_for_manual_file_barrier(
        args.output_dir,
        run_id,
        process_index,
        process_count,
        args.final_file_barrier_timeout_seconds,
    )
  else:
    print(f"Process {process_index}: wrote {writer.total_records} records; waiting at final multihost barrier", flush=True)

    # JAX distributed jobs are collective: if one host exits while others still run
    # VAE JAX work, the coordination service aborts the whole job. Keep fast hosts
    # alive until every host has closed its writers.
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices("tpu_tv2v_encoder_done")
    print(f"Process {process_index}: final barrier complete", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
