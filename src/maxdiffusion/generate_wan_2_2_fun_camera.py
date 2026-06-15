# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

import os
import sys
import time
from typing import Sequence

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

import jax
import jax.numpy as jnp

from absl import app
from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.common_types import WAN2_2
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image
import numpy as np

from maxdiffusion.pipelines.wan.wan_fun_camera_utils import (
    build_fun_camera_latents,
    build_fun_camera_latents_from_matrices,
)


def build_fun_camera_latents_from_config(config):
  ext_path = getattr(config, "extrinsic_clip_path", "") or ""
  int_path = getattr(config, "intrinsic_clip_path", "") or ""
  if ext_path and int_path:
    # Explicit-trajectory path: extrinsic [N,3,4] (w2c), intrinsic [N,3,3] (pixels).
    max_logging.log(f"Using explicit camera trajectory: {ext_path} / {int_path}")
    return build_fun_camera_latents_from_matrices(
        extrinsic=np.load(ext_path),
        intrinsic=np.load(int_path),
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        fps_in=float(getattr(config, "camera_clip_fps_in", 4.0)),
        fps_out=float(getattr(config, "camera_clip_fps_out", 24.0)),
    )
  return build_fun_camera_latents(
      direction=config.camera_control_direction,
      num_frames=config.num_frames,
      height=config.height,
      width=config.width,
      speed=float(config.camera_control_speed),
      origin=getattr(config, "camera_control_origin", None),
  )


def build_memory_inputs_from_config(config, dtype):
  """Load Captain-Safari 3D memory + pose tokens for memory-conditioned generation.

  Returns (memory[1,K*4*782,1024], target_pose_token[1,1,9], key_pose_token[1,K,9])
  or (None, None, None) when use_memory is off. Memory features are the first K
  keyframes flattened; pose tokens are encoded host-side from the query/key camera
  matrices (memory_pose.build_memory_pose_tokens), matching Captain-Safari exactly."""
  if not bool(getattr(config, "use_memory", False)):
    return None, None, None
  from maxdiffusion.models.wan.memory_pose import build_memory_pose_tokens

  mem_raw = np.load(config.memory_path, allow_pickle=True).astype(np.float32)  # [K,4,782,1024]
  # CS uses ALL keyframes the data provides (T = key_pose_token.shape[1]); the retriever's
  # "previous 4 frames" comments are stale. memory_n_key<=0 -> use all K (default); >0 -> first N.
  n_key_cfg = int(getattr(config, "memory_n_key", 0))
  n_key = mem_raw.shape[0] if n_key_cfg <= 0 else min(n_key_cfg, mem_raw.shape[0])
  memory = jnp.asarray(mem_raw[:n_key].reshape(1, -1, 1024), dtype=dtype)
  max_logging.log(f"Loaded memory: raw {mem_raw.shape} -> {memory.shape} ({n_key} keyframes)")

  target_np, key_np = build_memory_pose_tokens(
      extr_key=np.load(config.extrinsic_key_path),
      intr_key=np.load(config.intrinsic_key_path),
      extr_query=np.load(config.extrinsic_query_path),
      intr_query=np.load(config.intrinsic_query_path),
      n_key=n_key,
  )
  target_pose_token = jnp.asarray(target_np, dtype=dtype)  # [1,1,9]
  key_pose_token = jnp.asarray(key_np, dtype=dtype)        # [1,K,9]
  max_logging.log(f"Built pose tokens: target {target_pose_token.shape} key {key_pose_token.shape}")
  return memory, target_pose_token, key_pose_token


def run(config):
  if config.model_name != WAN2_2:
    raise ValueError("generate_wan_2_2_fun_camera.py only supports model_name=wan2.2")
  if config.model_type != "TI2V-CC":
    raise ValueError("Fun camera-control inference expects model_type=TI2V-CC")
  if not config.input_image_path:
    raise ValueError("Set input_image_path to the first/reference frame image.")
  has_clip = bool(getattr(config, "extrinsic_clip_path", "")) and bool(getattr(config, "intrinsic_clip_path", ""))
  if not config.camera_control_direction and not has_clip:
    raise ValueError(
        "Set camera_control_direction (e.g. Left) OR both extrinsic_clip_path + intrinsic_clip_path."
    )

  load_start = time.perf_counter()
  pipeline, _, _ = WanCheckpointer2_2_FunCamera(config=config).load_checkpoint()
  max_logging.log(f"load_time: {time.perf_counter() - load_start:.1f}s")

  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on
  dtype = getattr(config, "activations_dtype", jnp.bfloat16)

  image = load_image(config.input_image_path)
  y_latents = pipeline.prepare_fun_camera_y_latents(
      image,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      batch_size=len(prompt),
      dtype=dtype,
  )
  max_logging.log(f"Prepared y_latents shape: {y_latents.shape}")

  camera_np = build_fun_camera_latents_from_config(config)
  control_camera_latents_input = jnp.asarray(camera_np, dtype=dtype)
  if control_camera_latents_input.shape[0] == 1 and len(prompt) > 1:
    control_camera_latents_input = jnp.concatenate([control_camera_latents_input] * len(prompt), axis=0)
  max_logging.log(f"Prepared control_camera_latents_input shape: {control_camera_latents_input.shape}")

  memory, target_pose_token, key_pose_token = build_memory_inputs_from_config(config, dtype)

  videos, trace = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=config.guidance_scale,
      use_kv_cache=config.use_kv_cache,
      y_latents=y_latents,
      control_camera_latents_input=control_camera_latents_input,
      memory=memory,
      memory_pose_token=target_pose_token,
      memory_key_pose_token=key_pose_token,
  )
  max_logging.log(f"Inference trace: {trace}")

  output_name = getattr(config, "output_video_name", "wan22_fun_camera_output.mp4")
  for i in range(len(videos)):
    video_path = output_name if len(videos) == 1 else output_name.replace(".mp4", f"_{i}.mp4")
    export_to_video(videos[i], video_path, fps=config.fps)
    max_logging.log(f"Saved video: {video_path}")


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=False)
  config = pyconfig.config
  with transformer_engine_context():
    run(config)


if __name__ == "__main__":
  app.run(main)
