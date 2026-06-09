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


def _pack_plucker_to_camera_latents(plucker: np.ndarray, num_frames: int) -> np.ndarray:
  """Pack a Plucker embedding [F, H, W, 6] -> control latents [1, 24, F//4, H, W].

  Identical to DiffSynth's WanVideoUnit_FunCameraControl packing: repeat the first
  frame x4, then fold every 4 frames into the channel dim. Shared by both the
  direction-preset and explicit-trajectory paths.
  """
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


def build_fun_camera_latents(
    direction: str,
    num_frames: int,
    height: int,
    width: int,
    speed: float,
    origin: Optional[object] = None,
) -> np.ndarray:
  """Return packed camera-control latents [1, 24, latent_frames, H, W] (direction preset)."""
  plucker = process_camera_coordinates_to_plucker(direction, num_frames, height, width, speed, origin)
  return _pack_plucker_to_camera_latents(plucker, num_frames)


# ---------------------------------------------------------------------------
# Explicit-trajectory (extrinsic/intrinsic) path.
# Ports DiffSynth's interpolate_and_generate_plucker_cuda: SE(3) interpolation of
# 20 keyframes -> `length` frames, then Plucker generation, then the shared pack.
# extrinsic is expected in w2c convention (DiffSynth's process_vggt path uses the
# w2c interpolator), intrinsics in pixel units, used as-is (no rescale).
# ---------------------------------------------------------------------------
def _skew(w: np.ndarray) -> np.ndarray:
  wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
  O = np.zeros_like(wx)
  return np.stack(
      [
          np.stack([O, -wz, wy], axis=-1),
          np.stack([wz, O, -wx], axis=-1),
          np.stack([-wy, wx, O], axis=-1),
      ],
      axis=-2,
  )


def _so3_log(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
  tr = np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0), -2.0, 2.0) * 0.5
  tr = np.clip(tr, -1.0, 1.0)
  theta = np.arccos(tr)
  small = theta < eps
  denom = 2.0 * np.sin(theta)
  denom = np.where(small, np.ones_like(denom), denom)
  W_hat = (R - np.swapaxes(R, -1, -2)) / denom[..., None, None]
  w_gen = theta[..., None] * np.stack([W_hat[..., 2, 1], W_hat[..., 0, 2], W_hat[..., 1, 0]], axis=-1)
  w_small = np.stack(
      [
          (R[..., 2, 1] - R[..., 1, 2]) * 0.5,
          (R[..., 0, 2] - R[..., 2, 0]) * 0.5,
          (R[..., 1, 0] - R[..., 0, 1]) * 0.5,
      ],
      axis=-1,
  )
  return np.where(small[..., None], w_small, w_gen)


def _so3_exp(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
  theta = np.linalg.norm(w, axis=-1) + eps
  W = _skew(w)
  I = np.broadcast_to(np.eye(3, dtype=w.dtype), W.shape)
  a = (np.sin(theta) / theta)[..., None, None]
  b = ((1.0 - np.cos(theta)) / (theta * theta))[..., None, None]
  return I + a * W + b * (W @ W)


def _se3_exp_left(omega: np.ndarray, v_body: np.ndarray, tau: np.ndarray, eps: float = 1e-8):
  R = _so3_exp(omega * tau[..., None], eps=eps)
  theta = np.linalg.norm(omega, axis=-1) + eps
  tht = theta * tau
  W = _skew(omega)
  I = np.broadcast_to(np.eye(3, dtype=omega.dtype), W.shape)
  A = ((1.0 - np.cos(tht)) / (theta * theta))[..., None, None]
  B = ((tht - np.sin(tht)) / (theta * theta * theta))[..., None, None]
  V = I * tau[..., None, None] + A * W + B * (W @ W)
  p = (V @ v_body[..., None])[..., 0]
  return R, p


def _build_time_axes(n_in: int, fps_in: float, length: int, fps_out: float):
  t_in = np.arange(n_in, dtype=np.float32) / float(fps_in)
  t_out = np.arange(length, dtype=np.float32) / float(fps_out)
  return t_in, t_out


def _interp_with_extrap_1d(t_in: np.ndarray, y: np.ndarray, t_out: np.ndarray) -> np.ndarray:
  y = np.asarray(y, dtype=np.float32)
  if y.ndim == 1:
    y = y[None]
  lead = y.shape[:-1]
  N = y.shape[-1]
  y = y.reshape(-1, N)
  m0 = (y[:, 1] - y[:, 0]) / (t_in[1] - t_in[0])
  m1 = (y[:, -1] - y[:, -2]) / (t_in[-1] - t_in[-2])
  idx = np.searchsorted(t_in, t_out, side="right") - 1
  idx = np.clip(idx, 0, t_in.shape[0] - 2)
  t0 = t_in[idx]
  t1 = t_in[idx + 1]
  denom = (t1 - t0).astype(np.float32)
  denom[denom == 0] = 1.0
  alpha = (t_out - t0) / denom
  out = (1.0 - alpha) * y[:, idx] + alpha * y[:, idx + 1]
  head = t_out <= t_in[0]
  tail = t_out >= t_in[-1]
  if head.any():
    out[:, head] = y[:, :1] + m0[:, None] * (t_out[head] - t_in[0])
  if tail.any():
    out[:, tail] = y[:, -1:] + m1[:, None] * (t_out[tail] - t_in[-1])
  return out.reshape(*lead, t_out.shape[0])


def _interpolate_intrinsics(K: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
  K = np.asarray(K, dtype=np.float32)
  if K.ndim == 3:
    K = K[None]
  B, L = K.shape[0], t_out.shape[0]
  # constant extrapolation (clamp) for intrinsics, then linear interp
  t_clamped = np.clip(t_out, t_in[0], t_in[-1])
  fx = _interp_with_extrap_1d(t_in, K[..., 0, 0], t_clamped)
  fy = _interp_with_extrap_1d(t_in, K[..., 1, 1], t_clamped)
  cx = _interp_with_extrap_1d(t_in, K[..., 0, 2], t_clamped)
  cy = _interp_with_extrap_1d(t_in, K[..., 1, 2], t_clamped)
  K_out = np.zeros((B, L, 3, 3), dtype=np.float32)
  K_out[..., 0, 0] = fx
  K_out[..., 1, 1] = fy
  K_out[..., 0, 2] = cx
  K_out[..., 1, 2] = cy
  K_out[..., 2, 2] = 1.0
  return K_out


def _interpolate_extrinsics_w2c(E: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
  E = np.asarray(E, dtype=np.float32)
  if E.ndim == 4 and E.shape[1] == 1:
    E = E[:, 0]
  if E.ndim == 3:
    E = E[None]
  if E.shape[-2:] == (3, 4):
    E4 = np.tile(np.eye(4, dtype=np.float32), E.shape[:2] + (1, 1))
    E4[:, :, :3, :4] = E
    E = E4
  B, N = E.shape[:2]
  Rw2c = E[..., :3, :3]
  tw2c = E[..., :3, 3]
  Rc2w = np.swapaxes(Rw2c, -1, -2)
  tc2w = -(Rc2w @ tw2c[..., None])[..., 0]
  dt = (t_in[1:] - t_in[:-1]).reshape(1, -1, 1)
  R0 = Rc2w[:, :-1]
  dR = np.swapaxes(R0, -1, -2) @ Rc2w[:, 1:]
  omega = _so3_log(dR) / dt
  v_body = (np.swapaxes(R0, -1, -2) @ (tc2w[:, 1:] - tc2w[:, :-1])[..., None])[..., 0] / dt
  idx = np.searchsorted(t_in, t_out, side="right") - 1
  idx = np.clip(idx, 0, N - 2)
  tau = (t_out - t_in[idx]).astype(np.float32)
  R0_q = R0[:, idx]
  t0_q = tc2w[:, :-1][:, idx]
  R_delta, p_delta = _se3_exp_left(omega[:, idx], v_body[:, idx], tau)
  Rc = R0_q @ R_delta
  tc = t0_q + (R0_q @ p_delta[..., None])[..., 0]
  Rw = np.swapaxes(Rc, -1, -2)
  tw = -(Rw @ tc[..., None])[..., 0]
  E_out = np.tile(np.eye(4, dtype=np.float32), (B, t_out.shape[0], 1, 1))
  E_out[..., :3, :3] = Rw
  E_out[..., :3, 3] = tw
  return E_out


def _generate_plucker_from_matrices(extrinsic: np.ndarray, intrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
  """Plucker from w2c extrinsics + pixel intrinsics -> [B, L, H, W, 6]."""
  ext = np.asarray(extrinsic, dtype=np.float32)
  K = np.asarray(intrinsic, dtype=np.float32)
  if ext.ndim == 3:
    ext = ext[None]
  if K.ndim == 3:
    K = K[None]
  B, L = ext.shape[:2]
  y = np.linspace(0, height - 1, height, dtype=np.float32) + 0.5
  x = np.linspace(0, width - 1, width, dtype=np.float32) + 0.5
  v, u = np.meshgrid(y, x, indexing="ij")
  homog = np.stack([u, v, np.ones_like(u)], axis=-1)  # [H, W, 3]
  frames = []
  for f in range(L):
    R_w2c = ext[:, f, :3, :3]
    t_w2c = ext[:, f, :3, 3]
    R_c2w = np.swapaxes(R_w2c, -1, -2)
    t_c2w = -(R_c2w @ t_w2c[..., None])[..., 0]
    K_inv = np.linalg.inv(K[:, f])
    homog_b = np.broadcast_to(homog, (B, height, width, 3))
    r_cam = np.einsum("bij,bhwj->bhwi", K_inv, homog_b)
    d = np.einsum("bij,bhwj->bhwi", R_c2w, r_cam)
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12, None)
    o_e = np.broadcast_to(t_c2w[:, None, None, :], (B, height, width, 3))
    m = np.cross(o_e, d, axis=-1)
    frames.append(np.concatenate([m, d], axis=-1))
  return np.stack(frames, axis=1)


def build_fun_camera_latents_from_matrices(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    num_frames: int,
    height: int,
    width: int,
    fps_in: float = 4.0,
    fps_out: float = 24.0,
) -> np.ndarray:
  """Return packed camera-control latents [1, 24, F//4, H, W] from an explicit
  camera trajectory (extrinsic [N,3,4] w2c + intrinsic [N,3,3] pixels).

  Mirrors DiffSynth's interpolate_and_generate_plucker_cuda: interpolate N
  keyframes (@fps_in) to `num_frames` (@fps_out) via SE(3), then Plucker + pack.
  """
  ext = np.asarray(extrinsic, dtype=np.float32)
  K = np.asarray(intrinsic, dtype=np.float32)
  if ext.ndim == 4 and ext.shape[1] == 1:
    ext = ext[:, 0]
  if K.ndim == 4 and K.shape[1] == 1:
    K = K[:, 0]
  if ext.ndim == 3:
    ext = ext[None]
  if K.ndim == 3:
    K = K[None]
  n_in = ext.shape[1]
  t_in, t_out = _build_time_axes(n_in, fps_in, num_frames, fps_out)
  E4 = _interpolate_extrinsics_w2c(ext, t_in, t_out)
  Kq = _interpolate_intrinsics(K, t_in, t_out)
  plucker = _generate_plucker_from_matrices(E4[..., :3, :4], Kq, height, width)[0]  # [F, H, W, 6]
  return _pack_plucker_to_camera_latents(plucker, num_frames)
