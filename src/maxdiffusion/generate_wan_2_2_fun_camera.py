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
from maxdiffusion.pipelines.wan.wan_fun_camera_utils import build_fun_camera_latents



def build_fun_camera_latents_from_config(config):
  return build_fun_camera_latents(
      direction=config.camera_control_direction,
      num_frames=config.num_frames,
      height=config.height,
      width=config.width,
      speed=float(config.camera_control_speed),
      origin=getattr(config, "camera_control_origin", None),
  )


def run(config):
  if config.model_name != WAN2_2:
    raise ValueError("generate_wan_2_2_fun_camera.py only supports model_name=wan2.2")
  if config.model_type != "TI2V-CC":
    raise ValueError("Fun camera-control inference expects model_type=TI2V-CC")
  if not config.input_image_path:
    raise ValueError("Set input_image_path to the first/reference frame image.")
  if not config.camera_control_direction:
    raise ValueError("Set camera_control_direction, e.g. Left, Right, Up, Down, In, Out.")

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
