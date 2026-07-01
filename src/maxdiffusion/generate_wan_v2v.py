"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# V2V-1e: inference / generation for the v2v baseline (and hydra) model.
#
# A JAX port of HyDRA's inference (HyDRA/diffsynth/pipelines/HyDRA.py __call__,
# the concat/split + denoise loop). Given a conditioning clip (latents) + camera
# (cam_emb_con/tgt) + text, generate the target clip:
#
#   latents_input = concat([cond_latents, tgt], axis=2)   # [B,16,40,h,w]
#   for t in scheduler.timesteps (FlowMatch, N steps):
#       cond half is ALWAYS = cond_latents (clean)         # HyDRA.py:332
#       noise_pred = model(latents_input, t, text, cam_con, cam_tgt)
#       tgt = scheduler.step(noise_pred[:, :, tgt_len:], t, tgt)   # tgt half only, HyDRA.py:342
#   decode(tgt) -> target video.
#
# This is the exact inverse of the training loss
# (wan_v2v_concat_trainer.v2v_concat_loss): training seeds the cond half clean and
# supervises the tgt half; inference seeds the cond half clean and denoises the
# tgt half. The SAME script serves the baseline (hydra=False) and hydra
# (hydra=True) models -- the architecture is chosen by the `hydra` config flag at
# checkpoint-load time (create_sharded_logical_transformer threads it into
# WanModel), so pass hydra to match the checkpoint that was trained.
#
# The trained checkpoint is loaded through the training checkpointer
# (WanCheckpointerV2V.load_checkpoint), which handles BOTH a full-state save and a
# compact _v2v_only adapter-overlay save (save_lora_only=True) -- the same orbax
# path train_wan_v2v.py restores from.
#
# Smoke (single host, tpu-v6e-8-debug):
#   TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
#   MAXDIFFUSION_FORCE_LOCAL_DEVICE_MESH=1 \
#   python src/maxdiffusion/generate_wan_v2v.py \
#     src/maxdiffusion/configs/base_wan_2_1_v2v.yml \
#     run_name=v2v-gen-smoke hydra=True v2v_concat=True \
#     checkpoint_dir=gs://data_us_central1_a/maxdiffusion/wan/v2v-dbg/dbg-hydra-real/checkpoints \
#     v2v_record_tfrecord=<a -hydra-style-camera .tfrec> \
#     num_inference_steps=8 remat_policy=FULL guidance_scale=1.0 \
#     skip_jax_distributed_system=True per_device_batch_size=1 \
#     replicate_vae=True vae_spatial=4 \
#     output_dir=/home/spu9/maxdiffusion_outputs

import os
import sys
import time
from typing import Optional, Sequence

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.trainers.wan_v2v_concat_trainer import WanCheckpointerV2V
from maxdiffusion.utils import export_to_video


# ---------------------------------------------------------------------------
# Input record loading (mirrors the tv2v tfrecord schema read by
# WanV2VConcatTrainer.load_dataset: `latents`/`cond_latents` [16, N_lat, h, w],
# `encoder_hidden_states` [512, 4096], `cam_emb_con`/`cam_emb_tgt` [N_lat, 12]).
# ---------------------------------------------------------------------------
_FEATURE_DESCRIPTION = {
    "latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "latent": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "cond_latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "condition_latent": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    "cam_emb_con": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "cam_emb_tgt": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def _resolve_tfrecord_path(config) -> str:
  path = str(getattr(config, "v2v_record_tfrecord", "") or "").strip()
  if path:
    if tf.io.gfile.isdir(path):
      files = sorted(tf.io.gfile.glob(os.path.join(path, "*.tfrec")))
      if not files:
        files = sorted(tf.io.gfile.glob(os.path.join(path, "*")))
      if not files:
        raise ValueError(f"No tfrecord files under v2v_record_tfrecord dir: {path}")
      return files[0]
    return path
  # Fall back to the first .tfrec under train_data_dir.
  data_dir = str(getattr(config, "train_data_dir", "") or "").strip()
  if not data_dir:
    raise ValueError("Set v2v_record_tfrecord (a .tfrec file) or train_data_dir.")
  files = sorted(tf.io.gfile.glob(os.path.join(data_dir, "*.tfrec")))
  if not files:
    files = [f for f in sorted(tf.io.gfile.glob(os.path.join(data_dir, "*"))) if not f.endswith(".json")]
  if not files:
    raise ValueError(f"No tfrecord files found under train_data_dir: {data_dir}")
  return files[0]


def _pick(features, primary, alt):
  x = features[primary]
  return tf.where(tf.strings.length(x) > 0, x, features[alt])


def load_v2v_record(config):
  """Load ONE record and return numpy (cond_latents, tgt_latents, cam_emb_con,
  cam_emb_tgt, encoder_hidden_states), each WITHOUT a batch dim.

  tgt_latents (the ground-truth target) is returned only for reference/eval; the
  denoise loop does NOT use it.
  """
  path = _resolve_tfrecord_path(config)
  index = int(getattr(config, "v2v_record_index", 0))
  max_logging.log(f"V2V-1e: reading record #{index} from {path}")

  ds = tf.data.TFRecordDataset([path]).skip(index).take(1)
  raw = None
  for r in ds:
    raw = r
    break
  if raw is None:
    raise ValueError(f"Record index {index} out of range in {path}")

  f = tf.io.parse_single_example(raw, _FEATURE_DESCRIPTION)
  tgt_latents = tf.io.parse_tensor(_pick(f, "latents", "latent"), out_type=tf.float32).numpy()
  cond_latents = tf.io.parse_tensor(_pick(f, "cond_latents", "condition_latent"), out_type=tf.float32).numpy()
  encoder_hidden_states = tf.io.parse_tensor(f["encoder_hidden_states"], out_type=tf.float32).numpy()

  if tf.strings.length(f["cam_emb_con"]).numpy() > 0 and tf.strings.length(f["cam_emb_tgt"]).numpy() > 0:
    cam_emb_con = tf.io.parse_tensor(f["cam_emb_con"], out_type=tf.float32).numpy()
    cam_emb_tgt = tf.io.parse_tensor(f["cam_emb_tgt"], out_type=tf.float32).numpy()
  else:
    raise ValueError(
        f"Record {path}#{index} has no cam_emb_con/cam_emb_tgt. Use a "
        "--with-camera (-hydra-style-camera) tfrecord, or pass cameras explicitly."
    )
  return cond_latents, tgt_latents, cam_emb_con, cam_emb_tgt, encoder_hidden_states


# ---------------------------------------------------------------------------
# Denoise (mirror HyDRA.py __call__ + model_fn_wan_video).
# ---------------------------------------------------------------------------
@jax.jit
def _v2v_forward(graphdef, state, rest_of_state, hidden_states, timestep, encoder_hidden_states, cam_emb_con, cam_emb_tgt):
  """Velocity prediction over the FULL [B,16,40,h,w] concat (same forward the
  trainer's v2v_concat_loss calls, deterministic=True).

  graphdef is a regular (non-static) jit arg -- the proven maxdiffusion pattern
  (transformer_forward_pass_full_cfg): an nnx GraphDef is an unhashable pytree
  aux, so it must NOT be passed via static_argnums.
  """
  model = nnx.merge(graphdef, state, rest_of_state)
  return model(
      hidden_states=hidden_states,
      timestep=timestep,
      encoder_hidden_states=encoder_hidden_states,
      cam_emb_con=cam_emb_con,
      cam_emb_tgt=cam_emb_tgt,
      deterministic=True,
  )


def denoise_v2v(
    pipeline,
    config,
    cond_latents,
    cam_emb_con,
    cam_emb_tgt,
    encoder_hidden_states,
    encoder_hidden_states_neg=None,
):
  """HyDRA denoise loop. Returns (tgt_latents [B,16,N_lat,h,w], info dict)."""
  weights_dtype = config.weights_dtype
  num_inference_steps = int(config.num_inference_steps)
  guidance_scale = float(getattr(config, "guidance_scale", 1.0))
  do_cfg = guidance_scale != 1.0 and encoder_hidden_states_neg is not None

  cond_latents = jnp.asarray(cond_latents, dtype=weights_dtype)  # [B,16,N_lat,h,w]
  cam_emb_con = jnp.asarray(cam_emb_con, dtype=weights_dtype)  # [B,N_lat,12]
  cam_emb_tgt = jnp.asarray(cam_emb_tgt, dtype=weights_dtype)  # [B,N_lat,12]
  ehs = jnp.asarray(encoder_hidden_states, dtype=weights_dtype)  # [B,512,4096]
  ehs_neg = jnp.asarray(encoder_hidden_states_neg, dtype=weights_dtype) if do_cfg else None

  tgt_len = int(cond_latents.shape[2])  # N_lat -- the cond (first) half length
  bsz = int(cond_latents.shape[0])

  # FlowMatch scheduler (same class the trainer uses; NOT the pipeline's UniPC
  # inference scheduler). shift = flow_shift, full-strength denoise (start at pure
  # noise), mirroring HyDRA's set_timesteps(denoising_strength=1.0, shift=sigma_shift).
  scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
  sstate = scheduler.create_state()
  sstate = scheduler.set_timesteps(
      sstate,
      num_inference_steps=num_inference_steps,
      denoising_strength=1.0,
      shift=float(config.flow_shift),
  )
  timesteps = jnp.asarray(sstate.timesteps, dtype=jnp.float32)  # [N], ~[0, 1000]

  # Initial tgt = pure noise (HyDRA: latents = noise when input_video is None).
  noise_key = jax.random.key(int(config.seed))
  tgt = jax.random.normal(noise_key, cond_latents.shape, dtype=weights_dtype)

  graphdef, state, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

  step_times = []
  cond_clean_max_abs = None
  with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    for step in range(num_inference_steps):
      t = timesteps[step]
      # cond half is ALWAYS the clean input; only tgt is carried between steps.
      latents_input = jnp.concatenate([cond_latents, tgt], axis=2)  # [B,16,2*N_lat,h,w]

      t0 = time.perf_counter()
      if do_cfg:
        latents_doubled = jnp.concatenate([latents_input, latents_input], axis=0)
        ehs_combined = jnp.concatenate([ehs, ehs_neg], axis=0)
        cam_con_d = jnp.concatenate([cam_emb_con, cam_emb_con], axis=0)
        cam_tgt_d = jnp.concatenate([cam_emb_tgt, cam_emb_tgt], axis=0)
        t_b = jnp.broadcast_to(t, (bsz * 2,))
        pred = _v2v_forward(graphdef, state, rest_of_state, latents_doubled, t_b, ehs_combined, cam_con_d, cam_tgt_d)
        noise_pred = pred[bsz:] + guidance_scale * (pred[:bsz] - pred[bsz:])
      else:
        t_b = jnp.broadcast_to(t, (bsz,))
        noise_pred = _v2v_forward(graphdef, state, rest_of_state, latents_input, t_b, ehs, cam_emb_con, cam_emb_tgt)
      # Step the TGT (second) half ONLY (HyDRA.py:342).
      tgt_pred = noise_pred[:, :, tgt_len:]
      tgt, sstate = scheduler.step(sstate, tgt_pred, t, tgt, return_dict=False)
      tgt = tgt.astype(weights_dtype)
      tgt.block_until_ready()
      step_times.append(time.perf_counter() - t0)

      # GATE-4 probe (cheap): the cond half fed to the model must still equal the
      # clean input every step (== 0 by construction, but verify numerically).
      cond_diff = float(jnp.max(jnp.abs(latents_input[:, :, :tgt_len] - cond_latents)))
      cond_clean_max_abs = cond_diff if cond_clean_max_abs is None else max(cond_clean_max_abs, cond_diff)
      max_logging.log(
          f"  step {step + 1}/{num_inference_steps} t={float(t):.1f} "
          f"tgt[min={float(jnp.min(tgt)):.3f} max={float(jnp.max(tgt)):.3f} "
          f"finite={bool(jnp.all(jnp.isfinite(tgt)))}] ({step_times[-1]:.2f}s)"
      )

  info = {
      "tgt_finite": bool(jnp.all(jnp.isfinite(tgt))),
      "tgt_min": float(jnp.min(tgt)),
      "tgt_max": float(jnp.max(tgt)),
      "cond_clean_max_abs": cond_clean_max_abs,
      "step_times": step_times,
      "do_cfg": do_cfg,
      "tgt_len": tgt_len,
  }
  return tgt, info


def save_video(config, video_bhwc):
  """Save video[0] (a [F,H,W,3] uint8 clip) locally, and to GCS if output_dir is gs://."""
  name = str(getattr(config, "v2v_output_video_name", "v2v_output.mp4"))
  local_dir = str(getattr(config, "output_dir", ".") or ".")
  is_gcs = local_dir.startswith("gs://")
  local_path = name if is_gcs else os.path.join(local_dir, name)
  if not is_gcs:
    os.makedirs(local_dir, exist_ok=True)
  export_to_video(video_bhwc[0], local_path, fps=int(config.fps))
  max_logging.log(f"V2V-1e: saved local video {local_path} ({video_bhwc.shape[1]} frames)")
  if is_gcs:
    run = str(getattr(config, "run_name", "") or "")
    dst = "/".join([local_dir.rstrip("/"), run, "videos", name]) if run else "/".join([local_dir.rstrip("/"), "videos", name])
    tf.io.gfile.copy(local_path, dst, overwrite=True)
    max_logging.log(f"V2V-1e: uploaded video to {dst}")
    return dst
  return local_path


def run(config):
  # 1. Load the trained v2v checkpoint (base + adapters) via the training path.
  load_start = time.perf_counter()
  checkpointer = WanCheckpointerV2V(config=config)
  pipeline, _opt_state, step = checkpointer.load_checkpoint()
  if pipeline is None or getattr(pipeline, "transformer", None) is None:
    raise RuntimeError("Checkpoint load returned no pipeline/transformer.")
  n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(pipeline.transformer, nnx.Param))))
  max_logging.log(
      f"V2V-1e: loaded checkpoint step={step} in {time.perf_counter() - load_start:.1f}s "
      f"(transformer params={n_params:,}, hydra={getattr(config, 'hydra', False)}, "
      f"v2v_concat={getattr(config, 'v2v_concat', False)})"
  )

  # 2. Inputs from ONE tfrecord record (cond latents + camera + text encoded).
  cond_np, tgt_np, cam_con_np, cam_tgt_np, ehs_np = load_v2v_record(config)
  max_logging.log(
      f"V2V-1e: record shapes cond_latents={cond_np.shape} tgt_latents={tgt_np.shape} "
      f"cam_emb_con={cam_con_np.shape} cam_emb_tgt={cam_tgt_np.shape} enc_hidden={ehs_np.shape}"
  )
  cond_latents = cond_np[None]  # add batch dim -> [1,16,N_lat,h,w]
  cam_emb_con = cam_con_np[None]
  cam_emb_tgt = cam_tgt_np[None]
  ehs = ehs_np[None]

  # The flash-attention shard_map shards the batch axis over data/fsdp, so a
  # batch of 1 is not evenly divisible by an fsdp>1 mesh. Tile the single record
  # up to the device count (pure data-parallel replicas -> batch 1 per device,
  # same wall-clock) so the smoke runs on the whole slice; slice back to 1 for
  # decode. num_data_replicas = data * fsdp (the axes the batch shards over).
  mesh_shape = dict(zip(config.mesh_axes, pipeline.mesh.devices.shape))
  num_data_replicas = int(mesh_shape.get("data", 1)) * int(mesh_shape.get("fsdp", 1))
  if num_data_replicas > 1:
    cond_latents = np.repeat(cond_latents, num_data_replicas, axis=0)
    cam_emb_con = np.repeat(cam_emb_con, num_data_replicas, axis=0)
    cam_emb_tgt = np.repeat(cam_emb_tgt, num_data_replicas, axis=0)
    ehs = np.repeat(ehs, num_data_replicas, axis=0)
    max_logging.log(f"V2V-1e: tiled batch 1 -> {num_data_replicas} (data*fsdp) for even shard_map division")

  # 3. Denoise the tgt half (HyDRA loop).
  denoise_start = time.perf_counter()
  tgt_latents, info = denoise_v2v(pipeline, config, cond_latents, cam_emb_con, cam_emb_tgt, ehs)
  max_logging.log(f"V2V-1e: denoise ({config.num_inference_steps} steps) in {time.perf_counter() - denoise_start:.1f}s")

  # 4. VAE-decode the tgt half (denormalize -> decode), reusing the pipeline path.
  # Keep only the first replica (all are the same conditioning; different noise).
  tgt_latents = tgt_latents[:1]
  decode_start = time.perf_counter()
  den = pipeline._denormalize_latents(tgt_latents)
  video = pipeline._decode_latents_to_video(den)  # np [B,F,H,W,3] uint8
  max_logging.log(f"V2V-1e: decode in {time.perf_counter() - decode_start:.1f}s -> video {video.shape} {video.dtype}")

  saved = save_video(config, video)

  # --- Gates (plumbing verification) ---
  if bool(getattr(config, "v2v_gen_gate_checks", True)):
    st = info["step_times"]
    steady = st[1:] if len(st) > 1 else st
    video_finite = bool(np.all(np.isfinite(video.astype(np.float32))))
    vmin, vmax = float(video.min()), float(video.max())
    g1 = step is not None
    g2 = info["tgt_finite"]
    g3 = (video.ndim == 5) and (video.shape[-1] == 3) and video_finite and (0 <= vmin) and (vmax <= 255)
    g4 = (info["cond_clean_max_abs"] is not None) and (info["cond_clean_max_abs"] == 0.0)
    max_logging.log("=" * 60)
    max_logging.log("V2V-1e PLUMBING GATES")
    max_logging.log(f"  G1 checkpoint loads          : {'PASS' if g1 else 'FAIL'} (step={step}, params={n_params:,})")
    max_logging.log(
        f"  G2 denoise tgt-half finite    : {'PASS' if g2 else 'FAIL'} "
        f"(tgt min={info['tgt_min']:.3f} max={info['tgt_max']:.3f})"
    )
    max_logging.log(
        f"  G3 decode shape+range         : {'PASS' if g3 else 'FAIL'} "
        f"(shape={video.shape} dtype={video.dtype} range=[{vmin:.0f},{vmax:.0f}])"
    )
    max_logging.log(
        f"  G4 cond half stayed clean     : {'PASS' if g4 else 'FAIL'} "
        f"(max|cond_in - cond_latents|={info['cond_clean_max_abs']})"
    )
    if steady:
      max_logging.log(
          f"  step time: first={st[0]:.2f}s steady-mean={sum(steady)/len(steady):.2f}s "
          f"(cfg={info['do_cfg']})"
      )
    max_logging.log(f"  output: {saved}")
    max_logging.log("=" * 60)
  return saved


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=False)
  config = pyconfig.config
  max_utils.ensure_machinelearning_job_runs(config)
  with transformer_engine_context():
    run(config)


if __name__ == "__main__":
  app.run(main)
