"""JAX port of the Fun camera-control Plucker pipeline, for TRAINING.

The inference path (pipelines/wan/wan_fun_camera_utils.py, NumPy) was certified
against official PAI weights in E1; this module mirrors its math exactly so the
training-side conditioning lives in the same space the pretrained adapter
expects. Differences vs the inference path:

  * jit-able / batched (computed on-device inside the train step from the tiny
    per-frame matrices stored in each TFRecord — the packed rays are ~1.5GB per
    sample at pixel resolution and must never be stored or host-transferred);
  * NO SE(3) interpolation: training records store one (extrinsic, intrinsic)
    pair PER FRAME, already rebased to frame0 = identity.

Conventions (identical to _generate_plucker_from_matrices):
  extrinsic: [B, F, 3, 4]  world->camera (w2c), translations in meters
  intrinsic: [B, F, 3, 3]  pixel-unit K
  rays at pixel centers (+0.5), directions normalized with 1e-12 clip,
  plucker = [moment = o x d, direction d]  (6 channels)
Packing (identical to _pack_plucker_to_camera_latents): repeat frame 0 x4,
fold every 4 frames into channels -> [B, 24, (F+3)//4, H, W].
"""

import jax
import jax.numpy as jnp


def plucker_from_matrices(
    extrinsic: jax.Array,
    intrinsic: jax.Array,
    height: int,
    width: int,
    moment_scale: float = 1.0,
) -> jax.Array:
  """Plucker rays [B, F, H, W, 6] from per-frame w2c extrinsics + pixel K.

  moment_scale optionally rescales the moment (o x d) channels; the direction
  channels are unit vectors and are never scaled. Default 1.0 == the certified
  inference behavior (PAI's adapter was trained on unscaled meters-range
  moments, so only change this for from-scratch experiments).
  """
  ext = extrinsic.astype(jnp.float32)
  K = intrinsic.astype(jnp.float32)

  r_w2c = ext[..., :3, :3]                                   # [B, F, 3, 3]
  t_w2c = ext[..., :3, 3]                                    # [B, F, 3]
  r_c2w = jnp.swapaxes(r_w2c, -1, -2)
  t_c2w = -jnp.einsum("bfij,bfj->bfi", r_c2w, t_w2c)         # camera origin in world

  # Pixel-center grid, same +0.5 and (u, v, 1) layout as the NumPy reference.
  y = jnp.linspace(0, height - 1, height, dtype=jnp.float32) + 0.5
  x = jnp.linspace(0, width - 1, width, dtype=jnp.float32) + 0.5
  v, u = jnp.meshgrid(y, x, indexing="ij")
  homog = jnp.stack([u, v, jnp.ones_like(u)], axis=-1)        # [H, W, 3]

  k_inv = jnp.linalg.inv(K)                                   # [B, F, 3, 3]
  r_cam = jnp.einsum("bfij,hwj->bfhwi", k_inv, homog)         # [B, F, H, W, 3]
  d = jnp.einsum("bfij,bfhwj->bfhwi", r_c2w, r_cam)
  d = d / jnp.clip(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-12, None)

  o = jnp.broadcast_to(t_c2w[:, :, None, None, :], d.shape)
  m = jnp.cross(o, d) * moment_scale
  return jnp.concatenate([m, d], axis=-1)                     # [B, F, H, W, 6]


def pack_plucker_to_camera_latents(plucker: jax.Array) -> jax.Array:
  """[B, F, H, W, 6] -> packed control latents [B, 24, (F+3)//4, H, W].

  Identical to the NumPy reference / DiffSynth's WanVideoUnit_FunCameraControl:
  channel-first, repeat the first frame x4, fold every 4 frames into channels.
  """
  b, f, h, w, c = plucker.shape
  video = jnp.transpose(plucker, (0, 4, 1, 2, 3))             # [B, 6, F, H, W]
  video = jnp.concatenate(
      [jnp.repeat(video[:, :, 0:1], repeats=4, axis=2), video[:, :, 1:]], axis=2
  )                                                           # [B, 6, F+3, H, W]
  frames = f + 3
  if frames % 4 != 0:
    raise ValueError(f"packed camera frame count must be divisible by 4, got {frames}")
  lat = jnp.transpose(video, (0, 2, 1, 3, 4))                 # [B, F+3, 6, H, W]
  lat = lat.reshape(b, frames // 4, 4, c, h, w)
  lat = jnp.transpose(lat, (0, 1, 3, 2, 4, 5))                # [B, F_lat, 6, 4, H, W]
  lat = lat.reshape(b, frames // 4, c * 4, h, w)
  return jnp.transpose(lat, (0, 2, 1, 3, 4))                  # [B, 24, F_lat, H, W]


def build_control_camera_latents(
    extrinsic: jax.Array,
    intrinsic: jax.Array,
    height: int,
    width: int,
    moment_scale: float = 1.0,
    dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
  """TFRecord camera matrices -> control_camera_latents_input for WanModel.

  Wrapped in stop_gradient: cameras are pure inputs; no gradients flow back
  through the ray construction.
  """
  plucker = plucker_from_matrices(extrinsic, intrinsic, height, width, moment_scale)
  packed = pack_plucker_to_camera_latents(plucker)
  return jax.lax.stop_gradient(packed).astype(dtype)
