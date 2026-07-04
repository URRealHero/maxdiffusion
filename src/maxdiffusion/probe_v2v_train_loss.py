"""Decisive training-objective probe: OUR v2v_concat_loss evaluated on
(a) the OFFICIAL HyDRA weights, (b) a trained fix-try-1 checkpoint, at IDENTICAL
fixed batches and timesteps.

Logic: official weights are (approximately) a minimum of the OFFICIAL objective.
If our loss implementation matches theirs, official weights must score clearly
LOWER than our quarter/half-trained checkpoints. If official weights score the
same or higher on OUR loss, our objective differs from the one that produced
them -> code issue in loss/noising/conditioning.

Run on debug-8 (single host, fsdp=8), once per model config:
  python src/maxdiffusion/probe_v2v_train_loss.py <yml> run_name=... [model args] \
      probe_tag=official|ours5000|...
Prints: per-timestep mean loss over N fixed records.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.trainers.wan_v2v_concat_trainer import WanCheckpointerV2V, v2v_concat_loss

TFREC = "gs://data_us_central1_a/hmworld_data/wan_2_1/tv2v_encoded_full-hydra-style-camera/host_001_run_sid-wan21-full-001_file_000000.tfrec"
N_RECORDS = 8
TIMESTEPS = [100.0, 300.0, 500.0, 700.0, 900.0]

FEAT = {
    "latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "cond_latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
    "cam_emb_con": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "cam_emb_tgt": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def load_batch():
  rows = {"cond_latents": [], "tgt_latents": [], "cam_emb_con": [], "cam_emb_tgt": [], "encoder_hidden_states": []}
  for raw in tf.data.TFRecordDataset([TFREC]).take(N_RECORDS):
    f = tf.io.parse_single_example(raw, FEAT)
    rows["tgt_latents"].append(tf.io.parse_tensor(f["latents"], tf.float32).numpy())
    rows["cond_latents"].append(tf.io.parse_tensor(f["cond_latents"], tf.float32).numpy())
    rows["encoder_hidden_states"].append(tf.io.parse_tensor(f["encoder_hidden_states"], tf.float32).numpy())
    rows["cam_emb_con"].append(tf.io.parse_tensor(f["cam_emb_con"], tf.float32).numpy())
    rows["cam_emb_tgt"].append(tf.io.parse_tensor(f["cam_emb_tgt"], tf.float32).numpy())
  return {k: jnp.asarray(np.stack(v)) for k, v in rows.items()}


def main(argv):
  pyconfig.initialize(argv, validate_training=False)
  config = pyconfig.config
  tag = str(getattr(config, "probe_tag", "model"))
  checkpointer = WanCheckpointerV2V(config=config)
  pipeline, _, step = checkpointer.load_checkpoint()
  max_logging.log(f"[{tag}] loaded (ckpt step={step})")

  # training scheduler exactly as the trainer builds it (train_flow_shift honored)
  from maxdiffusion.schedulers import FlaxFlowMatchScheduler
  shift = float(getattr(config, "train_flow_shift", -1.0) or -1.0)
  sched = FlaxFlowMatchScheduler(shift=shift if shift > 0 else 3.0, dtype=jnp.float32)
  st = sched.create_state()
  st = sched.set_timesteps(st, num_inference_steps=1000, training=True)

  batch = load_batch()
  model = pipeline.transformer
  rng = jax.random.key(0)

  with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    for t in TIMESTEPS:
      ts = jnp.full((N_RECORDS,), t, dtype=jnp.float32)
      loss = v2v_concat_loss(model, dict(batch), rng, rng, ts, sched, config)
      max_logging.log(f"[{tag}] t={t:5.0f}  loss={float(loss):.5f}")
  max_logging.log(f"[{tag}] PROBE DONE")


if __name__ == "__main__":
  main(sys.argv)
