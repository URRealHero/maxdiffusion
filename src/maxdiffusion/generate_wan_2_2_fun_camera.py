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
import numpy as np

from absl import app
from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.common_types import WAN2_2
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image


def _parse_camera_origin(origin):
  if origin in (None, "", "None", "none"):
    return None
  if isinstance(origin, str):
    return tuple(float(x.strip()) for x in origin.split(",") if x.strip())
  return origin


def build_fun_camera_latents_from_reference(direction, num_frames, height, width, speed, origin, reference_path):
  """Build packed Plucker latents exactly like Captain-Safari/DiffSynth."""
  if reference_path and reference_path not in sys.path:
    sys.path.insert(0, reference_path)
  try:
    import torch
    from diffsynth.models.wan_video_camera_controller import SimpleAdapter
  except Exception as exc:
    raise ImportError(
        "Could not import Captain-Safari/DiffSynth camera helper. Set "
        "camera_control_reference_path to a directory containing diffsynth/."
    ) from exc

  adapter = SimpleAdapter(24, 1, kernel_size=(2, 2), stride=(2, 2), downscale_factor=16)
  plucker = adapter.process_camera_coordinates(
      direction,
      num_frames,
      height,
      width,
      speed,
      _parse_camera_origin(origin),
  )
  control_camera_video = plucker[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
  control_camera_latents = torch.concat(
      [
          torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
          control_camera_video[:, :, 1:],
      ],
      dim=2,
  ).transpose(1, 2)
  b, f, c, h, w = control_camera_latents.shape
  control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
  control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
  return control_camera_latents.cpu().float().numpy()


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

  camera_np = build_fun_camera_latents_from_reference(
      direction=config.camera_control_direction,
      num_frames=config.num_frames,
      height=config.height,
      width=config.width,
      speed=float(config.camera_control_speed),
      origin=getattr(config, "camera_control_origin", None),
      reference_path=config.camera_control_reference_path,
  )
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
