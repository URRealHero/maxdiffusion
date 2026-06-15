"""TPU-side noise_pred parity vs the CS dump — the decisive stage-3 check.

Feeds CS's EXACT dumped tensors (latents, timestep, context, y, control, memory + raw poses)
into OUR loaded DiT and diffs the resulting noise_pred against CS's dumped noise_pred.

  noise_pred MATCHES  -> model+conditioning are correct; the gap is the sampler/decode.
  noise_pred DIFFERS  -> the DiT forward (base/LoRA/memory) is off; we localize from here.

Also reports the diff with memory OFF (memory_context=None) to quantify memory's contribution,
and prints simple stats so we see whether our output is garbage vs a close-but-shifted match.

Run via examples/tpu_noise_pred_parity.sh (single host on 8-debug). PARITY_DIR holds the
downloaded CS *.npy dumps. Reuses the normal checkpoint/config load path so the model is
identical to generation.
"""
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from absl import app

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_SRC not in sys.path:
  sys.path.insert(0, _REPO_SRC)

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.pipelines.wan.wan_pipeline import transformer_forward_pass
from maxdiffusion.models.wan.memory_pose import build_memory_pose_tokens

D = os.environ.get("PARITY_DIR", "/tmp/cs_parity")


def _load(name):
  return np.load(os.path.join(D, name), allow_pickle=True)


def _stats(name, a):
  a = np.asarray(a, dtype=np.float64)
  max_logging.log(f"  {name:14s} shape={tuple(a.shape)} mean={a.mean():.5f} std={a.std():.5f} "
                  f"min={a.min():.4f} max={a.max():.4f}")


def _rel(ours, ref):
  ours = np.asarray(ours, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
  d = np.abs(ours - ref)
  denom = np.abs(ref).max() + 1e-8
  # cosine similarity (scale/shift-robust signal of "same direction")
  cos = float((ours.flatten() @ ref.flatten()) / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-8))
  return d.max(), d.mean(), d.max() / denom, cos


def run(config):
  dtype = jnp.bfloat16
  max_logging.log(f"[parity] loading pipeline (use_memory={getattr(config,'use_memory',False)})")
  pipeline, _, _ = WanCheckpointer2_2_FunCamera(config=config).load_checkpoint()
  transformer = pipeline.transformer

  latents = jnp.asarray(_load("latents.npy"), dtype=dtype)         # (1,48,31,44,80)
  y = jnp.asarray(_load("y.npy"), dtype=dtype)                     # (1,52,31,44,80)
  context = jnp.asarray(_load("context.npy"), dtype=dtype)         # (1,512,4096)
  control = jnp.asarray(_load("control_camera_latents_input.npy"), dtype=dtype)  # (1,24,31,704,1280)
  timestep = jnp.asarray(_load("timestep.npy"), dtype=jnp.int32)   # (1,)
  cs_noise = _load("noise_pred.npy")                              # (1,48,31,44,80)
  mem_np = np.asarray(_load("memory.npy"), dtype=np.float32)       # (1,16,4,782,1024)

  max_logging.log("[parity] CS input stats:")
  for n, a in [("latents", latents), ("y", y), ("context", context), ("control", control),
               ("cs_noise_pred", cs_noise), ("memory", mem_np)]:
    _stats(n, a)

  # memory + pose tokens (ALL keyframes -> T from the dump)
  n_key = mem_np.shape[1]
  memory = jnp.asarray(mem_np.reshape(1, -1, 1024), dtype=dtype)
  tgt, key = build_memory_pose_tokens(
      _load("raw_extrinsic_key.npy"), _load("raw_intrinsic_key.npy"),
      _load("raw_extrinsic_query.npy"), _load("raw_intrinsic_query.npy"), n_key=n_key)
  memory_context = transformer.compute_memory_context(
      memory, jnp.asarray(tgt, dtype=dtype), jnp.asarray(key, dtype=dtype)).astype(dtype)
  _stats("memory_ctx", memory_context)

  # assemble DiT input exactly like run_inference: x = concat([latents, y], channel)
  x = jnp.concatenate([latents, y], axis=1)  # (1,100,31,44,80)
  rope_channels = latents.shape[1] + y.shape[1]
  dummy = jnp.zeros((latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4], rope_channels))
  rotary = transformer.rope(dummy)
  graphdef, state, rest = nnx.split(transformer, nnx.Param, ...)

  def fwd(mc):
    with pipeline.mesh:
      np_out, _ = transformer_forward_pass(
          graphdef, state, rest, x, timestep, context,
          do_classifier_free_guidance=False, guidance_scale=5.0,
          rotary_emb=rotary, control_camera_latents_input=control, memory_context=mc)
    return np.asarray(np_out, dtype=np.float64)

  max_logging.log("[parity] forward WITH memory ...")
  ours_mem = fwd(memory_context)
  max_logging.log("[parity] forward WITHOUT memory ...")
  ours_nomem = fwd(None)

  _stats("ours(mem)", ours_mem)
  _stats("ours(nomem)", ours_nomem)
  mm, me, mr, mc_ = _rel(ours_mem, cs_noise)
  nm, ne, nr, nc = _rel(ours_nomem, cs_noise)
  mem_effect = np.abs(ours_mem - ours_nomem).max()
  max_logging.log("================ NOISE_PRED PARITY ================")
  max_logging.log(f"  WITH memory   : max_abs={mm:.4e} mean_abs={me:.4e} rel={mr:.4e} cos={mc_:.5f}")
  max_logging.log(f"  WITHOUT memory: max_abs={nm:.4e} mean_abs={ne:.4e} rel={nr:.4e} cos={nc:.5f}")
  max_logging.log(f"  memory effect on our noise_pred (max|with-without|): {mem_effect:.4e}")
  max_logging.log("  VERDICT: cos~1 & small rel -> DiT matches CS (gap=sampler/decode);")
  max_logging.log("           cos<<1 / large rel -> DiT forward itself diverges (weights/conditioning).")
  np.save("/tmp/ours_noise_pred_mem.npy", ours_mem)


def main(argv):
  pyconfig.initialize(argv, validate_training=False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
