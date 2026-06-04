# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence
import jax
import time
import os
import subprocess
import numpy as np
from maxdiffusion import pyconfig, max_logging, max_utils
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion.checkpointing.wan_checkpointer_2_2_dense import WanCheckpointer2_2_Dense
from absl import app
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image
from google.cloud import storage
import flax
from maxdiffusion.common_types import WAN2_1, WAN2_2
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader


def upload_video_to_gcs(output_dir: str, video_path: str):
  """
  Uploads a local video file to a specified Google Cloud Storage bucket.
  """
  try:
    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    source_file_path = f"./{video_path}"
    destination_blob_name = os.path.join(folder_name, "videos", video_path)

    blob = bucket.blob(destination_blob_name)

    max_logging.log(f"Uploading {source_file_path} to {bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    max_logging.log(f"Upload complete {source_file_path}.")
    return f"gs://{bucket_name}/{destination_blob_name}"

  except Exception as e:
    max_logging.log(f"An error occurred: {e}")
    return None


def delete_file(file_path: str):
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def _prepare_video_for_tensorboard(video, max_frames):
  """Converts a generated video to tensorboardX add_video format: N,T,C,H,W."""
  video = np.asarray(video)
  if video.ndim == 3:
    video = video[None, ...]
  if video.ndim != 4:
    max_logging.log(f"Skipping TensorBoard video logging for unsupported shape: {video.shape}")
    return None

  # Accept either T,H,W,C or C,T,H,W.
  if video.shape[-1] in (1, 3, 4):
    video = video[..., :3]
  elif video.shape[0] in (1, 3, 4):
    video = np.transpose(video[:3], (1, 2, 3, 0))
  else:
    max_logging.log(f"Skipping TensorBoard video logging for ambiguous shape: {video.shape}")
    return None

  if max_frames > 0 and video.shape[0] > max_frames:
    frame_ids = np.linspace(0, video.shape[0] - 1, max_frames).round().astype(np.int32)
    video = video[frame_ids]

  if np.issubdtype(video.dtype, np.floating):
    video = np.clip(video, 0.0, 1.0) * 255.0
  else:
    video = np.clip(video, 0, 255)
  video = video.astype(np.uint8)
  return np.transpose(video, (0, 3, 1, 2))[None, ...]


def _video_to_frame_grid(tb_video, columns=8):
  # tb_video is N,T,C,H,W. Use first video and tile frames into an image strip/grid.
  frames = np.transpose(tb_video[0], (0, 2, 3, 1))
  rows = int(np.ceil(frames.shape[0] / columns))
  pad = rows * columns - frames.shape[0]
  if pad:
    frames = np.concatenate([frames, np.zeros((pad, *frames.shape[1:]), dtype=frames.dtype)], axis=0)
  row_images = []
  for row in range(rows):
    row_images.append(np.concatenate(frames[row * columns : (row + 1) * columns], axis=1))
  return np.concatenate(row_images, axis=0)


def _write_eval_video_to_tensorboard(writer, video, tag, step, fps, max_frames):
  if writer is None or jax.process_index() != 0:
    return
  tb_video = _prepare_video_for_tensorboard(video, max_frames)
  if tb_video is None:
    return
  try:
    writer.add_video(tag, tb_video, global_step=step, fps=fps)
  except Exception as e:  # Keep eval generation from failing because of logging.
    max_logging.log(f"Failed to write eval video to TensorBoard, writing frame grid fallback: {e}")
    try:
      writer.add_image(f"{tag}_frames", _video_to_frame_grid(tb_video), global_step=step, dataformats="HWC")
    except Exception as image_e:
      max_logging.log(f"Failed to write eval frame grid to TensorBoard: {image_e}")
  writer.flush()


def get_git_commit_hash():
  """Tries to get the current Git commit hash."""
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    return commit_hash
  except subprocess.CalledProcessError:
    max_logging.log("Warning: 'git rev-parse HEAD' failed. Not running in a git repo?")
    return None
  except FileNotFoundError:
    max_logging.log("Warning: 'git' command not found.")
    return None


jax.config.update("jax_use_shardy_partitioner", True)


def call_pipeline(config, pipeline, prompt, negative_prompt):
  model_key = config.model_name
  model_type = config.model_type
  if model_type == "I2V":
    image = load_image(config.image_url)
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
          use_kv_cache=config.use_kv_cache,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
          use_kv_cache=config.use_kv_cache,
      )
    else:
      raise ValueError(f"Unsupported model_name for I2V in config: {model_key}")
  elif model_type in ("TI2V", "TI2V-CC"):
    if model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_cfg_cache=config.use_cfg_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
          use_kv_cache=config.use_kv_cache,
      )
    else:
      raise ValueError(f"Unsupported model_name for TI2V in config: {model_key}")
  elif model_type == "T2V":
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_cfg_cache=config.use_cfg_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
          use_kv_cache=config.use_kv_cache,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
          use_kv_cache=config.use_kv_cache,
      )
    else:
      raise ValueError(f"Unsupported model_name for T2V in config: {model_key}")
  else:
    raise ValueError(f"Unsupported model_type in config: {model_type}")


def inference_generate_video(config, pipeline, filename_prefix="", writer=None, step=None):
  s0 = time.perf_counter()
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}, video: {filename_prefix}"
  )

  outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
  if isinstance(outputs, tuple):
    videos = outputs[0]
  else:
    videos = outputs

  max_logging.log(f"video {filename_prefix}, compile time: {(time.perf_counter() - s0)}")
  max_outputs = getattr(config, "tensorboard_eval_video_max_outputs", 1)
  max_frames = getattr(config, "tensorboard_eval_video_max_frames", 32)
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    uploaded_video_path = None
    if config.output_dir.startswith("gs://"):
      uploaded_video_path = upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
      # Delete local files to avoid storing too many videos.
      delete_file(f"./{video_path}")
    if i < max_outputs:
      tb_step = step if step is not None else 0
      _write_eval_video_to_tensorboard(
          writer,
          videos[i],
          tag=f"eval/generated_video_{i}",
          step=tb_step,
          fps=config.fps,
          max_frames=max_frames,
      )
      if writer is not None and jax.process_index() == 0 and uploaded_video_path:
        writer.add_text(f"eval/generated_video_{i}_path", uploaded_video_path, global_step=tb_step)
  return


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  model_key = config.model_name # WAN 2.1 / WAN 2.2
  writer = max_utils.initialize_summary_writer(config) # tensorboard logging 
  if jax.process_index() == 0 and writer: # process 0 writes.
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  if pipeline is None: # choosing ckpt (e.g., wan2.1, T2V)
    load_start = time.perf_counter()
    model_type = config.model_type
    if model_key == WAN2_1:
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_1(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_1(config=config)
    elif model_key == WAN2_2:
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_2(config=config)
      elif model_type == "TI2V":
        checkpoint_loader = WanCheckpointer2_2_Dense(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_2(config=config)
    else:
      raise ValueError(f"Unsupported model_name for checkpointer: {model_key}")
    pipeline, _, _ = checkpoint_loader.load_checkpoint() # loading
    load_time = time.perf_counter() - load_start
    max_logging.log(f"load_time: {load_time:.1f}s")
  else:
    load_time = 0.0

  # If LoRA is specified, inject layers and load weights.
  if (
      config.enable_lora
      and hasattr(config, "lora_config")
      and config.lora_config
      and config.lora_config["lora_model_name_or_path"]
  ):
    if model_key == WAN2_1:
      lora_loader = Wan2_1NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            transformer_weight_name=lora_config["weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

    if model_key == WAN2_2:
      lora_loader = Wan2_2NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            high_noise_weight_name=lora_config["high_noise_weight_name"][i],
            low_noise_weight_name=lora_config["low_noise_weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

  s0 = time.perf_counter()

  # Disable profiler for the first two runs to avoid duplicate uploads
  original_enable_profiler = config.enable_profiler if "enable_profiler" in config.get_keys() else False
  config.get_keys()["enable_profiler"] = False

  # Using global_batch_size_to_train_on so not to create more config variables
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )
  videos = call_pipeline(config, pipeline, prompt, negative_prompt)
  if isinstance(videos, tuple):
    videos = videos[0]

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  s0 = time.perf_counter()
  outputs = call_pipeline(config, pipeline, prompt, negative_prompt)
  if isinstance(outputs, tuple):
    videos, trace = outputs
  else:
    videos = outputs
    trace = {}
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_devices = jax.device_count()
    num_videos = num_devices * config.per_device_batch_size
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")
  summary = [
      f"\n{'=' * 50}",
      "  TIMING SUMMARY",
      f"{'=' * 50}",
      f"  Load (checkpoint):   {load_time:>7.1f}s",
      f"  Compile:             {compile_time:>7.1f}s",
      f"  Inference:           {generation_time:>7.1f}s",
  ]
  if trace:
    summary.extend([
        f"  {'─' * 40}",
        f"  Conditioning:        {trace.get('conditioning', 0.0):>7.1f}s",
        f"  Denoise Total:       {trace.get('denoise_total', 0.0):>7.1f}s",
        f"  VAE Decode:          {trace.get('vae_decode', 0.0):>7.1f}s",
    ])
  summary.append(f"{'=' * 50}")
  max_logging.log("\n".join(summary))

  s0 = time.perf_counter()
  # Restore original profiler setting for the profiling run
  config.get_keys()["enable_profiler"] = original_enable_profiler
  if max_utils.profiler_enabled(config):
    # Injecting user requested XLA tracing flags
    xla_flags = os.environ.get("XLA_FLAGS", "")
    new_flags = "--xla_enable_mxu_trace=true --xla_jf_dump_llo_html=true --xla_tpu_enable_llo_profiling=true"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {new_flags}"
    max_logging.log(f"Injected XLA_FLAGS for profiling: {new_flags}")

    videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    if isinstance(videos, tuple):
      videos = videos[0]
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash() # tracing
  pyconfig.initialize(argv) # configuration
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  max_utils.ensure_machinelearning_job_runs(pyconfig.config) # cloud job bookkeeping
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
