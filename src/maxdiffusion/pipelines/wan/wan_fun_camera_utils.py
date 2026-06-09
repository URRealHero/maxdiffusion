# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

"""Utilities for PAI Wan2.2-Fun camera-control conditioning.

This ports the camera trajectory -> Plucker embedding packing used by
DiffSynth/Captain-Safari without importing DiffSynth at runtime.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

_DEFAULT_CAMERA_ORIGIN = (
    0,
    0.532139961,
    0.946026558,
    0.5,
    0.5,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
)


def parse_camera_origin(origin: Optional[object]) -> Tuple[float, ...]:
  if origin in (None, "", "None", "none"):
    return tuple(float(x) for x in _DEFAULT_CAMERA_ORIGIN)
  if isinstance(origin, str):
    values = tuple(float(x.strip()) for x in origin.split(",") if x.strip())
  else:
    values = tuple(float(x) for x in origin)
  if len(values) != len(_DEFAULT_CAMERA_ORIGIN):
    raise ValueError(f"camera origin must contain {len(_DEFAULT_CAMERA_ORIGIN)} values, got {len(values)}")
  return values


def generate_camera_coordinates(direction: str, length: int, speed: float, origin: Optional[object] = None):
  coordinates = [list(parse_camera_origin(origin))]
  while len(coordinates) < length:
    coor = coordinates[-1].copy()
    if "Left" in direction:
      coor[9] += speed
    if "Right" in direction:
      coor[9] -= speed
    if "Up" in direction:
      coor[13] += speed
    if "Down" in direction:
      coor[13] -= speed
    if "In" in direction:
      coor[18] -= speed
    if "Out" in direction:
      coor[18] += speed
    coordinates.append(coor)
  return coordinates


class _Camera:
  def __init__(self, entry: Sequence[float]):
    self.fx, self.fy, self.cx, self.cy = [float(x) for x in entry[1:5]]
    w2c = np.asarray(entry[7:], dtype=np.float32).reshape(3, 4)
    w2c_4x4 = np.eye(4, dtype=np.float32)
    w2c_4x4[:3, :] = w2c
    self.w2c_mat = w2c_4x4
    self.c2w_mat = np.linalg.inv(w2c_4x4).astype(np.float32)


def _get_relative_pose(cam_params):
  abs_w2cs = [cam.w2c_mat for cam in cam_params]
  abs_c2ws = [cam.c2w_mat for cam in cam_params]
  target_cam_c2w = np.array(
      [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      dtype=np.float32,
  )
  abs2rel = target_cam_c2w @ abs_w2cs[0]
  return np.asarray([target_cam_c2w] + [abs2rel @ c2w for c2w in abs_c2ws[1:]], dtype=np.float32)


def _ray_condition(intrinsics: np.ndarray, c2w: np.ndarray, height: int, width: int) -> np.ndarray:
  # intrinsics: [1, frames, 4], c2w: [1, frames, 4, 4]
  batch = intrinsics.shape[0]
  yy, xx = np.meshgrid(
      np.linspace(0, height - 1, height, dtype=np.float32),
      np.linspace(0, width - 1, width, dtype=np.float32),
      indexing="ij",
  )
  xx = xx.reshape(1, 1, height * width) + 0.5
  yy = yy.reshape(1, 1, height * width) + 0.5
  xx = np.broadcast_to(xx, (batch, intrinsics.shape[1], height * width))
  yy = np.broadcast_to(yy, (batch, intrinsics.shape[1], height * width))

  fx, fy, cx, cy = np.split(intrinsics, 4, axis=-1)
  z = np.ones_like(xx)
  x = (xx - cx) / fx * z
  y = (yy - cy) / fy * z
  dirs = np.stack((x, y, z), axis=-1)
  dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

  rays_d = dirs @ np.swapaxes(c2w[..., :3, :3], -1, -2)
  rays_o = np.broadcast_to(c2w[..., :3, 3][:, :, None, :], rays_d.shape)
  rays_dxo = np.cross(rays_o, rays_d)
  plucker = np.concatenate([rays_dxo, rays_d], axis=-1)
  return plucker.reshape(batch, c2w.shape[1], height, width, 6).astype(np.float32)


def process_camera_coordinates_to_plucker(
    direction: str,
    length: int,
    height: int,
    width: int,
    speed: float,
    origin: Optional[object] = None,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
) -> np.ndarray:
  cam_params = [_Camera(entry) for entry in generate_camera_coordinates(direction, length, speed, origin)]

  sample_wh_ratio = width / height
  pose_wh_ratio = original_pose_width / original_pose_height
  if pose_wh_ratio > sample_wh_ratio:
    resized_ori_w = height * pose_wh_ratio
    for cam in cam_params:
      cam.fx = resized_ori_w * cam.fx / width
  else:
    resized_ori_h = width / pose_wh_ratio
    for cam in cam_params:
      cam.fy = resized_ori_h * cam.fy / height

  intrinsic = np.asarray(
      [[cam.fx * width, cam.fy * height, cam.cx * width, cam.cy * height] for cam in cam_params],
      dtype=np.float32,
  )[None]
  c2ws = _get_relative_pose(cam_params)[None]
  plucker = _ray_condition(intrinsic, c2ws, height, width)[0]
  return plucker.astype(np.float32)  # [frames, height, width, 6]


def build_fun_camera_latents(
    direction: str,
    num_frames: int,
    height: int,
    width: int,
    speed: float,
    origin: Optional[object] = None,
) -> np.ndarray:
  """Return packed camera-control latents [1, 24, latent_frames, H, W]."""
  plucker = process_camera_coordinates_to_plucker(direction, num_frames, height, width, speed, origin)
  control_camera_video = np.transpose(plucker[:num_frames], (3, 0, 1, 2))[None]
  control_camera_latents = np.concatenate(
      [np.repeat(control_camera_video[:, :, 0:1], repeats=4, axis=2), control_camera_video[:, :, 1:]],
      axis=2,
  )
  control_camera_latents = np.transpose(control_camera_latents, (0, 2, 1, 3, 4))
  batch, frames, channels, h, w = control_camera_latents.shape
  if frames % 4 != 0:
    raise ValueError(f"packed camera frame count must be divisible by 4, got {frames}")
  control_camera_latents = control_camera_latents.reshape(batch, frames // 4, 4, channels, h, w)
  control_camera_latents = np.transpose(control_camera_latents, (0, 1, 3, 2, 4, 5))
  control_camera_latents = control_camera_latents.reshape(batch, frames // 4, channels * 4, h, w)
  return np.transpose(control_camera_latents, (0, 2, 1, 3, 4)).astype(np.float32)
