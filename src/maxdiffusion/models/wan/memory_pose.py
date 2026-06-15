"""Host-side camera pose encoding for Captain-Safari 3D memory (pure numpy).

Ports CS's model_fn_wan_video memory-pose preprocessing so the JAX pipeline can
build the memory retriever's pose tokens from raw camera matrices, exactly matching
Captain-Safari:
  convert_to_local_coordinates  (first key frame -> origin, w2c)
  extri_intri_to_pose_encoding  ("absT_quaR_FoV": [T(3), quat_xyzw(4), fov_h, fov_w])
  mat_to_quat                   (PyTorch3D, scalar-last XYZW, standardized real>=0)

These are tiny per-sample ops (a handful of 3x4 matrices) -> run on host, not TPU.
Validated against the CS torch reference: pose tokens match to ~1e-6.
"""
import numpy as np

POSE_HW = (256, 512)  # image_size_hw used by CS for FoV (note: (H, W))


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
  """Rotation matrices (...,3,3) -> quaternions (...,4), scalar-last XYZW, standardized."""
  batch = matrix.shape[:-2]
  m = matrix.reshape(batch + (9,))
  m00, m01, m02, m10, m11, m12, m20, m21, m22 = [m[..., i] for i in range(9)]

  q_abs = np.sqrt(np.maximum(0.0, np.stack([
      1.0 + m00 + m11 + m22,
      1.0 + m00 - m11 - m22,
      1.0 - m00 + m11 - m22,
      1.0 - m00 - m11 + m22,
  ], axis=-1)))

  quat_by_rijk = np.stack([
      np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
      np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
      np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
      np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
  ], axis=-2)

  flr = np.array(0.1, dtype=q_abs.dtype)
  quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

  best = np.argmax(q_abs, axis=-1)  # (...,)
  onehot = (np.arange(4) == best[..., None])  # (...,4) bool
  out = quat_candidates[onehot].reshape(batch + (4,))
  out = out[..., [1, 2, 3, 0]]  # rijk -> ijkr (scalar-last)
  out = np.where(out[..., 3:4] < 0, -out, out)  # standardize real>=0
  return out


def convert_to_local_coordinates(extrinsics: np.ndarray) -> np.ndarray:
  """(N,3,4) w2c extrinsics -> (N,3,4) with the first frame as the origin."""
  extrinsics = np.asarray(extrinsics, dtype=np.float64)
  N = extrinsics.shape[0]
  R0 = extrinsics[0, :, :3]
  t0 = extrinsics[0, :, 3]
  T0_c2w = np.eye(4)
  T0_c2w[:3, :3] = R0.T
  T0_c2w[:3, 3] = -(R0.T @ t0)
  T0_w2c = np.linalg.inv(T0_c2w)

  out = np.zeros_like(extrinsics)
  for i in range(N):
    Ti_w2c = np.eye(4)
    Ti_w2c[:3, :3] = extrinsics[i, :, :3]
    Ti_w2c[:3, 3] = extrinsics[i, :, 3]
    Ti_local_c2w = T0_w2c @ np.linalg.inv(Ti_w2c)
    out[i] = np.linalg.inv(Ti_local_c2w)[:3, :]
  return out


def extri_intri_to_pose_encoding(extrinsics: np.ndarray, intrinsics: np.ndarray,
                                 image_size_hw=POSE_HW) -> np.ndarray:
  """(B,S,3,4)+(B,S,3,3) -> (B,S,9) = [T(3), quat_xyzw(4), fov_h, fov_w]."""
  extrinsics = np.asarray(extrinsics, dtype=np.float64)
  intrinsics = np.asarray(intrinsics, dtype=np.float64)
  R = extrinsics[..., :3, :3]
  T = extrinsics[..., :3, 3]
  quat = mat_to_quat(R)
  H, W = image_size_hw
  fov_h = 2.0 * np.arctan((H / 2.0) / intrinsics[..., 1, 1])
  fov_w = 2.0 * np.arctan((W / 2.0) / intrinsics[..., 0, 0])
  return np.concatenate([T, quat, fov_h[..., None], fov_w[..., None]], axis=-1)


def build_memory_pose_tokens(extr_key, intr_key, extr_query, intr_query,
                             n_key=4, image_size_hw=POSE_HW):
  """Raw camera matrices -> (target_pose_token [1,1,9], key_pose_token [1,n_key,9]).

  Mirrors model_fn_wan_video: concat first n_key key extrinsics with the query, convert
  to local coords (first key frame = origin), then pose-encode query (S=1) and keys (S=n_key).
  extr_key (>=n_key,3,4), intr_key (>=n_key,3,3), extr_query (3,4), intr_query (3,3)."""
  extr_key = np.asarray(extr_key, dtype=np.float64)[:n_key]
  intr_key = np.asarray(intr_key, dtype=np.float64)[:n_key]
  extr_query = np.asarray(extr_query, dtype=np.float64)
  intr_query = np.asarray(intr_query, dtype=np.float64)

  combined = np.concatenate([extr_key, extr_query[None]], axis=0)  # (n_key+1,3,4)
  combined_local = convert_to_local_coordinates(combined)
  extr_key_local = combined_local[:n_key][None]       # (1,n_key,3,4)
  extr_query_local = combined_local[n_key:][None]     # (1,1,3,4)

  target = extri_intri_to_pose_encoding(extr_query_local, intr_query[None, None], image_size_hw)  # (1,1,9)
  key = extri_intri_to_pose_encoding(extr_key_local, intr_key[None], image_size_hw)               # (1,n_key,9)
  return target.astype(np.float32), key.astype(np.float32)
