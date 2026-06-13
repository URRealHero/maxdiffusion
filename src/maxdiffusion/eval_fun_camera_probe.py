"""E4b gate: camera-causality probe for the Fun camera-control trainer.

Computes the flow-match eval loss on real dataset records twice:
  (a) with each sample's TRUE camera trajectory
  (b) with cameras SHUFFLED across the batch (roll by 1)
at fixed timesteps. If the model exploits the camera signal, loss(true) must be
clearly below loss(shuffled). Works with the pretrained PAI checkpoint (no
training needed) and with our orbax checkpoints.

Run on ONE pod worker:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  MAXDIFFUSION_FORCE_LOCAL_DEVICE_MESH=1 \
  python src/maxdiffusion/eval_fun_camera_probe.py \
    src/maxdiffusion/configs/base_wan_2_2_fun_5b_camera.yml \
    run_name=e4b-probe output_dir=/tmp/e4b skip_jax_distributed_system=True \
    per_device_batch_size=1 replicate_vae=True vae_spatial=4 \
    height=480 width=832 num_frames=153
"""

from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.wan.camera_plucker import build_control_camera_latents
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.trainers.wan_2_2_fun_camera_trainer import (
    _build_first_frame_mask,
    _per_token_timesteps,
)

TFREC = "gs://data_us_central1_a/hmworld_data/fun_camera_encoded_full/host_001_run_1007319_file_000000.tfrec"
N_RECORDS = 8
FIXED_TIMESTEPS = (300.0, 700.0)  # mid/high noise; fixed for variance reduction
# batch=1 per forward: batch=2 trips an XLA fusion-emitter crash on the 4-chip
# single-host layout (the adapter's 6144ch pixel-unshuffle conv). The "shuffled"
# condition pairs each record with the NEXT record's cameras instead of an
# in-batch roll.


def read_records():
  feature_description = {
      k: tf.io.FixedLenFeature([], tf.string)
      for k in ("latents", "latent_condition", "encoder_hidden_states", "camera_extrinsic", "camera_intrinsic")
  }
  ds = tf.data.TFRecordDataset([TFREC]).take(N_RECORDS)
  records = []
  for raw in ds:
    f = tf.io.parse_single_example(raw, feature_description)
    records.append({k: tf.io.parse_tensor(f[k], tf.float32).numpy()[None] for k in feature_description})
  return records


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  config = pyconfig.config

  from maxdiffusion.pipelines.wan.wan_pipeline_2_2_fun_camera import WanPipeline2_2_FunCamera

  pipe = WanPipeline2_2_FunCamera.from_pretrained(config)
  max_logging.log("PROBE: pipeline loaded")
  model = pipe.transformer
  p_h, p_w = model.config.patch_size[1], model.config.patch_size[2]

  scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
  scheduler_state = scheduler.create_state()
  scheduler.set_timesteps(scheduler_state, num_inference_steps=1000, training=True)

  rng = jax.random.key(config.seed)

  def eval_loss(batch, cam_batch):
    latents = jnp.asarray(batch["latents"], config.weights_dtype)
    latent_condition = jnp.asarray(batch["latent_condition"], config.weights_dtype)
    text = jnp.asarray(batch["encoder_hidden_states"], config.weights_dtype)
    ext = jnp.asarray(cam_batch["camera_extrinsic"])
    intr = jnp.asarray(cam_batch["camera_intrinsic"])

    control = build_control_camera_latents(
        ext, intr, height=config.height, width=config.width, dtype=config.weights_dtype
    )
    f_lat, h_lat, w_lat = latents.shape[2], latents.shape[3], latents.shape[4]
    tokens_per_frame = (h_lat // p_h) * (w_lat // p_w)

    total = 0.0
    for t_val in FIXED_TIMESTEPS:
      timesteps = jnp.full((latents.shape[0],), t_val, dtype=jnp.float32)
      noise = jax.random.normal(jax.random.fold_in(rng, int(t_val)), latents.shape, latents.dtype)
      noisy, target, _ = scheduler.apply_flow_match(noise, latents, timesteps)
      noisy = noisy.at[:, :, 0:1].set(latents[:, :, 0:1])
      mask = _build_first_frame_mask(latents)
      hidden = jnp.concatenate([noisy, mask, latent_condition], axis=1)
      t_tok = _per_token_timesteps(timesteps, f_lat, tokens_per_frame)

      with pipe.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        pred = model(
            hidden_states=hidden,
            timestep=t_tok,
            encoder_hidden_states=text,
            deterministic=True,
            control_camera_latents_input=control,
        )
      total += float(jnp.mean((target[:, :, 1:] - pred[:, :, 1:]) ** 2))
    return total / len(FIXED_TIMESTEPS)

  records = read_records()
  max_logging.log(f"PROBE: {len(records)} records, batch=1 forwards")
  true_losses, shuf_losses = [], []
  for i, rec in enumerate(records):
    other = records[(i + 1) % len(records)]
    lt = eval_loss(rec, rec)
    ls = eval_loss(rec, other)
    true_losses.append(lt)
    shuf_losses.append(ls)
    max_logging.log(f"PROBE record {i}: true={lt:.5f}  shuffled={ls:.5f}")

  mt, ms = float(np.mean(true_losses)), float(np.mean(shuf_losses))
  margin = (ms - mt) / mt * 100
  max_logging.log(f"PROBE RESULT: true={mt:.5f}  shuffled={ms:.5f}  margin={margin:+.1f}%")
  max_logging.log("PROBE VERDICT: " + ("CAMERA SIGNAL USED" if ms > mt * 1.02 else "NO CLEAR CAMERA EFFECT"))


if __name__ == "__main__":
  app.run(main)
