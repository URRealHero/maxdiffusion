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


def encode_videos(pipeline, videos: np.ndarray) -> np.ndarray:
  import jax.numpy as jnp
  from flax.linen import partitioning as nn_partitioning

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


def make_example(latent: np.ndarray, cond_latent: np.ndarray, hidden_state: np.ndarray, condition_output_field: str) -> bytes:
  load_numpy()
  import tensorflow as tf

  features = {
      "latents": bytes_feature(serialize_tensor(latent)),
      condition_output_field: bytes_feature(serialize_tensor(cond_latent)),
      "encoder_hidden_states": bytes_feature(serialize_tensor(hidden_state)),
  }
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
      for _, _, record in records:
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

      target_array = np.concatenate(target_videos, axis=0)
      condition_array = np.concatenate(condition_videos, axis=0)
      latents = encode_videos(pipeline, target_array)
      cond_latents = encode_videos(pipeline, condition_array)
      hidden_states = encode_prompts(pipeline, prompts, args.max_sequence_length)

      encoded_records = list(zip(records, latents, cond_latents, hidden_states))
      for (_, sample_id, _), latent, cond_latent, hidden_state in encoded_records:
        assert_finite_encoded_record(sample_id, latent, cond_latent, hidden_state)

      for (_, sample_id, record), latent, cond_latent, hidden_state in encoded_records:
        writer.write(make_example(latent, cond_latent, hidden_state, args.condition_output_field))
        metadata_writer.write(
            {
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
            },
        )
        print(f"encoded sample_id={sample_id} latent_shape={latent.shape}", flush=True)

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
