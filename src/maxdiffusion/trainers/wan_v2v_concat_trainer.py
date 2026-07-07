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

# V2V-1c: v2v concat TRAINER for the HyDRA-baseline reimplementation.
#
# Mirrors HyDRA/train_hydra.py training_step + _freeze_and_mark_trainables:
#   input latents = concat([cond(20 latent frames), tgt(20 latent frames)], axis=2)
#                   -> [B, 16, 40, h, w]  (cond first half, tgt second half)
#   noise         = randn_like(latents); noisy = apply_flow_match(noise, latents, t)
#                   over ALL 40 frames, then the cond (first) half is FORCED clean
#                   (noisy[:, :, :tgt_len] = latents[:, :, :tgt_len]).
#   camera        = cam_emb_con [B, 20, 12] + cam_emb_tgt [B, 20, 12], injected per
#                   latent-frame half INSIDE the blocks (V2V-1a); NO y-concat, NO
#                   control-adapter, NO extra input channels (Wan2.1 in_dim=out_dim=16).
#   loss          = flow-match MSE on the TGT (second) half only (frames tgt_len:).
#   trainable     = only params whose path contains cam_encoder_con / cam_encoder_tgt
#                   / projector / attn1 (self-attention); everything else FROZEN via
#                   optax.set_to_zero (no optimizer state for the frozen base).
#
# Everything else (optimizer build, schedule, logging, checkpoint cadence) is the
# verified WAN 2.1 training loop, unchanged.

import functools
import json
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
import tensorflow as tf
from flax import nnx

from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.wan_trainer import WanTrainer
from maxdiffusion import max_logging, max_utils
from maxdiffusion.pipelines.wan.wan_pipeline_2_1 import WanPipeline2_1

import orbax.checkpoint as ocp

from jax.sharding import PartitionSpec as P


# The 4 trainable-param substrings (HyDRA train_hydra.py:96, but maxdiffusion names
# the self-attention module `attn1`, not `self_attn`). Base params (patch_embedding,
# cross-attn attn2, ffn, condition_embedder, norms, proj_out) stay frozen.
V2V_DEFAULT_TRAINABLE_SUBSTRINGS = ("cam_encoder_con", "cam_encoder_tgt", "projector", "attn1")


def v2v_trainable_substrings(config):
  """The trainable-param substrings for the v2v partial fine-tune.

  Reads `v2v_trainable_param_substrings` (comma-separated); falls back to
  `lora_trainable_param_substrings`, then to the HyDRA-baseline default.
  """
  raw = str(getattr(config, "v2v_trainable_param_substrings", "") or "").strip()
  if not raw:
    raw = str(getattr(config, "lora_trainable_param_substrings", "") or "").strip()
  if not raw:
    return list(V2V_DEFAULT_TRAINABLE_SUBSTRINGS)
  return [s.strip() for s in raw.split(",") if s.strip()]


def _path_str(path):
  """Stable string for an nnx flat-state path tuple (save/overlay must agree)."""
  return "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path)


class WanCheckpointerV2V(WanCheckpointer2_1):
  """Wan2.1 base loader (with the v2v_concat scaffold filled by the load path) +
  the trainable-substring freezing optimizer (optax.multi_transform).

  Reuses WanCheckpointer2_1.load_checkpoint / load_diffusers_checkpoint, so the
  transformer is built through create_sharded_logical_transformer, which threads
  config.v2v_concat -> WanModel and fresh-inits cam_encoder_{con,tgt}/projector
  (V2V-1b). Only the optimizer construction is specialized here.
  """

  def _create_optimizer(self, model, config, learning_rate):
    """Freeze everything except the v2v trainable substrings (mirror
    WanCheckpointer2_2_FunCamera._create_optimizer): only params whose path
    contains one of the substrings are trained; the rest are frozen via
    optax.set_to_zero (no Adam state, so the frozen base costs no extra memory)."""
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    base_tx = max_utils.create_optimizer(config, learning_rate_scheduler)

    substrings = v2v_trainable_substrings(config)
    if not substrings:
      return base_tx, learning_rate_scheduler  # full fine-tune

    _, params, _ = nnx.split(model, nnx.Param, ...)

    def _label(path, _leaf):
      ps = jax.tree_util.keystr(path)
      return "train" if any(s in ps for s in substrings) else "freeze"

    labels = jax.tree_util.tree_map_with_path(_label, params)
    leaves = jax.tree_util.tree_leaves(labels)
    n_train = sum(1 for v in leaves if v == "train")
    max_logging.log(
        f"V2V partial fine-tune: training {n_train}/{len(leaves)} param tensors "
        f"(substrings={substrings}); base frozen via set_to_zero."
    )
    tx = optax.multi_transform({"train": base_tx, "freeze": optax.set_to_zero()}, labels)
    return tx, learning_rate_scheduler

  def load_diffusers_checkpoint(self):
    return WanPipeline2_1.from_pretrained(self.config)

  def _overlay_v2v_checkpoint(self, saved_flat):
    """Load the Wan2.1 base pipeline (with the fresh v2v scaffold) and overlay the
    saved trainable (adapter/attn1) params on top."""
    pipeline = self.load_diffusers_checkpoint()
    transformer = pipeline.transformer
    state = nnx.state(transformer, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    applied = 0
    for p, v in flat.items():
      sp = _path_str(p)
      if sp in saved_flat:
        target = v.value
        new_val = jnp.asarray(saved_flat[sp], dtype=target.dtype)
        sharding = getattr(target, "sharding", None)
        v.value = jax.device_put(new_val, sharding) if sharding is not None else new_val
        applied += 1
    if applied != len(saved_flat):
      missing = set(saved_flat) - {_path_str(p) for p in flat}
      raise ValueError(
          f"V2V overlay mismatch: applied {applied}/{len(saved_flat)} saved tensors; "
          f"unmatched saved keys: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
      )
    nnx.update(transformer, nnx.from_flat_state(flat))
    max_logging.log(f"V2V-only checkpoint: overlaid {applied} trainable tensors onto the Wan2.1 base.")
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_1, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      if restored_checkpoint.wan_config.get("_v2v_only", False):
        pipeline = self._overlay_v2v_checkpoint(restored_checkpoint.wan_state)
      else:
        pipeline = WanPipeline2_1.from_checkpoint(self.config, restored_checkpoint)
        if "opt_state" in restored_checkpoint.wan_state.keys():
          opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default Wan2.1 v2v pipeline.")
      pipeline = self.load_diffusers_checkpoint()
    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_1, train_states):
    """Full-state save, or (when save_lora_only=True) a compact save of only the
    v2v trainable params."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    wan_config = config_to_json(pipeline.transformer)

    substrings = v2v_trainable_substrings(self.config)
    save_v2v_only = bool(getattr(self.config, "save_lora_only", False)) and len(substrings) > 0
    if save_v2v_only:
      source = getattr(train_states, "params", train_states)
      flat = dict(nnx.to_flat_state(source))
      v2v_flat = {_path_str(p): v.value for p, v in flat.items() if any(s in _path_str(p) for s in substrings)}
      if len(v2v_flat) == 0:
        raise ValueError(
            f"save_lora_only=True but no params matched {substrings}; refusing to save an empty checkpoint."
        )
      wan_config["_v2v_only"] = True
      wan_config["_v2v_substrings"] = substrings
      state_save = ocp.args.StandardSave(v2v_flat)
      max_logging.log(f"  V2V-only checkpoint: {len(v2v_flat)} trainable tensors (base frozen, not saved).")
    else:
      state_save = ocp.args.StandardSave(train_states)

    items = {
        "wan_config": ocp.args.JsonSave(wan_config),
        "wan_state": state_save,
    }
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")


def v2v_concat_loss(model, batch, noise_rng, dropout_rng, timesteps, scheduler, config, _zero_cond_half_pred=False):
  """The V2V-1c forward + flow-match loss (HyDRA train_hydra.py:109-138).

  `batch` carries cond_latents/tgt_latents [B, 16, 20, h, w], cam_emb_con/tgt
  [B, 20, 12], encoder_hidden_states [B, 512, 4096].

  `_zero_cond_half_pred` is a GATE-C3 hook (default False, unused in training):
  when True, the cond (first) half of the model prediction is zeroed before the
  loss. Because the loss reads only the tgt (second) half, the result must be
  identical -- proving the loss is computed on the tgt half alone.
  """
  cond_latents = batch["cond_latents"].astype(config.weights_dtype)  # [B, 16, 20, h, w]
  tgt_latents = batch["tgt_latents"].astype(config.weights_dtype)  # [B, 16, 20, h, w]
  cam_emb_con = batch["cam_emb_con"].astype(config.weights_dtype)  # [B, 20, 12]
  cam_emb_tgt = batch["cam_emb_tgt"].astype(config.weights_dtype)  # [B, 20, 12]
  encoder_hidden_states = batch["encoder_hidden_states"].astype(config.weights_dtype)  # [B, 512, 4096]

  # V2V frame-concat along the latent-frame axis: cond FIRST, tgt SECOND.
  latents = jnp.concatenate([cond_latents, tgt_latents], axis=2)  # [B, 16, 40, h, w]
  tgt_len = latents.shape[2] // 2  # 20: [:tgt_len] = cond (kept clean), [tgt_len:] = tgt (loss).

  noise = jax.random.normal(key=noise_rng, shape=latents.shape, dtype=latents.dtype)
  noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)
  # FORCE the cond (first) half clean (HyDRA:114). The cond frames are the given
  # context; only the tgt frames are denoised.
  noisy_latents = noisy_latents.at[:, :, :tgt_len].set(latents[:, :, :tgt_len])

  with jax.named_scope("forward_pass"):
    model_pred = model(
        hidden_states=noisy_latents,  # 16-ch noisy latents only (NO y-concat / control-adapter).
        timestep=timesteps,  # scalar [B] timestep.
        encoder_hidden_states=encoder_hidden_states,
        cam_emb_con=cam_emb_con,
        cam_emb_tgt=cam_emb_tgt,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )

  if _zero_cond_half_pred:
    model_pred = model_pred.at[:, :, :tgt_len].set(0.0)

  with jax.named_scope("loss"):
    # Loss ONLY on the tgt (second) half (HyDRA:134-138).
    loss = (training_target[:, :, tgt_len:] - model_pred[:, :, tgt_len:]) ** 2
    if not config.disable_training_weights:
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight
    loss = jnp.mean(loss)

  return loss


# ---------------------------------------------------------------------------
# Camera-from-camera.json FALLBACK (used only when the tfrec has no cam_emb_*).
#
# Self-contained replica of tpu_tfrecord_encoder.encode_tv2v.load_hydra_cam_emb
# (verified bit-exact against the re-encoded tv2v cam_emb: max|diff|=0). Kept
# here so the trainer has no import dependency on the encoder package at train
# time. Reads camera.json via tf.io.gfile (works for local and gs:// paths).
# ---------------------------------------------------------------------------
# Where to find <sample_id>/camera.json when the fallback fires. Overridable via
# config keys v2v_camera_json_local_root / v2v_camera_json_gcs_root.
V2V_CAMERA_JSON_LOCAL_ROOT = "/data-2u-2/spu/HM-World"
V2V_CAMERA_JSON_GCS_ROOT = "gs://data_us_central1_a/hmworld_raw/HM-World/HM-World"


def _v2v_apply_coordinate_transform(c2w):
  """Unreal-world c2w (cm) -> HyDRA camera-convention c2w (m): permute columns
  [1,2,0,3], flip Y, cm->m (encode_tv2v._apply_coordinate_transform)."""
  t = np.array(c2w, dtype=np.float64)
  t = t[:, [1, 2, 0, 3]]
  t[:3, 1] *= -1.0
  t[:3, 3] /= 100.0
  return t


def _v2v_load_hydra_cam_emb(camera_json_path, num_frames):
  """Compute HyDRA-convention relative camera embeddings from a HM-World
  camera.json (both streams relative to the TARGET frame-0 pose). num_frames is
  the RAW frame count (e.g. 77); cam_idx = range(num_frames)[::4] -> N_lat rows.
  Returns (cam_emb_con[N_lat,12], cam_emb_tgt[N_lat,12]) float32."""
  with tf.io.gfile.GFile(camera_json_path, "r") as f:
    data = json.load(f)
  if "cond_cam" not in data or "tgt_cam" not in data:
    raise ValueError(f"camera.json missing cond_cam/tgt_cam: {camera_json_path}")

  def poses(key):
    d = data[key]
    n = len(d)
    mats = [np.asarray(d[str(i)], dtype=np.float64) for i in range(n)]
    if not mats:
      raise ValueError(f"{key} empty in {camera_json_path}")
    if num_frames > n:  # pad by repeating last pose (mirrors last-frame video pad)
      mats.extend([mats[-1]] * (num_frames - n))
    return mats

  cond_poses = poses("cond_cam")
  tgt_poses = poses("tgt_cam")
  cam_idx = list(range(num_frames))[::4]
  ref_w2c = np.linalg.inv(_v2v_apply_coordinate_transform(tgt_poses[0]))

  def to_rel(ps):
    return np.stack(
        [(ref_w2c @ _v2v_apply_coordinate_transform(ps[i]))[:3, :4].reshape(-1) for i in cam_idx],
        axis=0,
    ).astype(np.float32)

  con = to_rel(cond_poses)
  tgt = to_rel(tgt_poses)
  if not (np.isfinite(con).all() and np.isfinite(tgt).all()):
    raise ValueError(f"nonfinite cam_emb from {camera_json_path}")
  return con, tgt


def _v2v_cam_from_json(sample_id, n_lat, local_root, gcs_root):
  """tf.py_function body: locate <sample_id>/camera.json (local root first, then
  GCS) and compute cam_emb_con/tgt for N_lat latent frames. num_frames for the
  derivation = 4*(N_lat-1)+1 (20 latent -> 77 raw)."""
  sid = sample_id.numpy().decode("utf-8") if hasattr(sample_id, "numpy") else str(sample_id)
  if not sid:
    raise ValueError(
        "V2V camera fallback: record has NO cam_emb_con/tgt in the tfrec AND no "
        "`sample_id` tfrec feature, so camera.json cannot be located. Re-encode the "
        "data with `--with-camera`, or add a `sample_id` feature to the encoder."
    )
  n = int(n_lat.numpy()) if hasattr(n_lat, "numpy") else int(n_lat)
  num_frames = 4 * (n - 1) + 1
  local_root = local_root.numpy().decode("utf-8") if hasattr(local_root, "numpy") else str(local_root)
  gcs_root = gcs_root.numpy().decode("utf-8") if hasattr(gcs_root, "numpy") else str(gcs_root)
  local_path = f"{local_root.rstrip('/')}/{sid}/camera.json"
  gcs_path = f"{gcs_root.rstrip('/')}/{sid}/camera.json"
  path = local_path if tf.io.gfile.exists(local_path) else gcs_path
  con, tgt = _v2v_load_hydra_cam_emb(path, num_frames)
  return con, tgt


def _v2v_cam_missing_error(sample_id):
  """tf.py_function body for the fallback-disabled branch: fail loudly."""
  sid = sample_id.numpy().decode("utf-8") if hasattr(sample_id, "numpy") else str(sample_id)
  raise ValueError(
      f"V2V record (sample_id={sid!r}) has no cam_emb_con/tgt in the tfrec and "
      "v2v_camera_from_json_fallback=False. Enable the fallback or re-encode with --with-camera."
  )


# ---------------------------------------------------------------------------
# Train/eval split by sample_id (held-out set).
#
# The re-encoded tv2v tfrecs carry a `sample_id` bytes feature. We train on
# (full - held-out) and eval on the held-out set. The split is applied as a
# tf.data.filter over the PARSED record, using an in-graph tf.lookup.StaticHashTable
# (sid -> 1 for held-out, default 0) -- NOT a py_function -- so it stays efficient
# and multi-host safe (every host builds the identical table from the same file).
# ---------------------------------------------------------------------------
def _v2v_load_heldout_sids(path):
  """Read the held-out sample_id list (one sid per line) via tf.io.gfile.

  Blank lines and surrounding whitespace are stripped; works for local and gs://.
  """
  with tf.io.gfile.GFile(path, "r") as f:
    raw = f.read()
  return [s.strip() for s in raw.splitlines() if s.strip()]


def _v2v_make_split_filter(config):
  """Build the tf.data filter predicate for the sample_id train/eval split.

  Returns None (no filtering -> current behavior) when `v2v_split_mode` == 'all'.
  Otherwise reads `v2v_heldout_sids_file` into an in-graph StaticHashTable and
  returns a predicate over the PARSED feature dict (keyed on features['sample_id']):
    - 'train': keep records whose sample_id is NOT in the held-out set.
    - 'eval' : keep records whose sample_id IS in the held-out set.
  A record with no sample_id feature parses to an empty string (default_value ""),
  which is never in the held-out set -> kept by 'train', dropped by 'eval'.
  """
  mode = str(getattr(config, "v2v_split_mode", "all") or "all").strip().lower()
  if mode == "all":
    return None  # strict no-op: bit-identical to the pre-split behavior.
  if mode not in ("train", "eval"):
    raise ValueError(f"v2v_split_mode must be one of all/train/eval, got {mode!r}.")

  path = str(getattr(config, "v2v_heldout_sids_file", "") or "").strip()
  if not path:
    raise ValueError(
        f"v2v_split_mode={mode!r} requires v2v_heldout_sids_file to be set "
        "(path to the held-out sample_id list, one sid per line)."
    )
  sids = _v2v_load_heldout_sids(path)
  if not sids:
    raise ValueError(f"v2v_heldout_sids_file {path!r} is empty; no held-out sids to split on.")

  keys = tf.constant(sids, dtype=tf.string)
  values = tf.ones([len(sids)], dtype=tf.int32)
  table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=0)
  keep_heldout = mode == "eval"
  max_logging.log(
      f"V2V split filter: mode={mode} ("
      f"{'INCLUDE ONLY' if keep_heldout else 'EXCLUDE'} held-out); "
      f"{len(sids)} held-out sids from {path}."
  )

  def _split_filter(features):
    is_heldout = tf.equal(table.lookup(features["sample_id"]), 1)
    return is_heldout if keep_heldout else tf.logical_not(is_heldout)

  return _split_filter


class WanV2VConcatTrainer(WanTrainer):
  """Fine-tunes Wan2.1-T2V-1.3B with the V2V-1a per-block camera scaffold, on
  frame-concat (cond|tgt) latents, training only the camera/projector/self-attn
  params (HyDRA baseline)."""

  def _get_checkpointer(self):
    return WanCheckpointerV2V(config=self.config)

  def get_data_shardings(self, mesh):
    # Shard the input batch over ('data','fsdp') ONLY (the activation batch axes).
    # With per_device_batch_size<1 the loader delivers batch=num_devices and the
    # train step slices to global_batch_size_to_train_on IN-JIT; slicing a batch
    # dim sharded across the context axis silently corrupts the samples that must
    # migrate between context shards (repro'd 2026-07-06: half the sliced batch
    # comes back 100% NaN). ('data','fsdp') is divisible before AND after the
    # slice, and matches the model's activation batch sharding. bs>=1 runs are
    # unaffected (the slice is a no-op there).
    data_sharding = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))
    return {
        "cond_latents": data_sharding,
        "tgt_latents": data_sharding,
        "cam_emb_con": data_sharding,
        "cam_emb_tgt": data_sharding,
        "encoder_hidden_states": data_sharding,
    }

  def get_eval_data_shardings(self, mesh):
    # No eval-loss path (like the Fun camera trainer): the encoded dataset carries
    # no per-record `timesteps` feature. Returned only for API symmetry.
    return self.get_data_shardings(mesh)

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "V2V training only supports dataset_type=tfrecord with cache_latents_text_encoder_outputs=True"
      )

    # REAL WAN2.1 tv2v tfrecord schema (verified against a live record in
    # gs://data_us_central1_a/hmworld_data/wan_2_1/tv2v_encoded_full-hydra-style-camera):
    #   float feature `latents`               [16, N_lat, h, w]  -> internal tgt_latents
    #   float feature `cond_latents`          [16, N_lat, h, w]  -> internal cond_latents
    #   float feature `encoder_hidden_states` [512, 4096]
    #   float feature `cam_emb_con`/`cam_emb_tgt` [N_lat, 12]  (present ONLY in the
    #     --with-camera re-encode; ABSENT in older camera-less encodes)
    #   bytes feature `sample_id`             (present in the re-encode; used for the
    #     train/eval held-out split filter below and the camera.json fallback). Older
    #     encodes lack it -> parses to "" (default_value), which the split filter and
    #     fallback both handle gracefully.
    #
    # The OLD (pre-fix) loader read a feature named `tgt_latents`, which does NOT
    # exist (the encoder writes `latents`), and required cam_emb_* unconditionally.
    # Legacy singular aliases (`latent` / `condition_latent`) are also accepted so
    # the loader is robust to either encoder naming convention.
    cam_fallback = bool(getattr(config, "v2v_camera_from_json_fallback", True))
    cam_local_root = str(getattr(config, "v2v_camera_json_local_root", V2V_CAMERA_JSON_LOCAL_ROOT))
    cam_gcs_root = str(getattr(config, "v2v_camera_json_gcs_root", V2V_CAMERA_JSON_GCS_ROOT))

    feature_description = {
        # target latents: real key `latents`, legacy alias `latent`.
        "latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "latent": tf.io.FixedLenFeature([], tf.string, default_value=""),
        # condition latents: real key `cond_latents`, legacy alias `condition_latent`.
        "cond_latents": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "condition_latent": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        # camera + sample_id are OPTIONAL (absent in older/camera-less encodes).
        "cam_emb_con": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "cam_emb_tgt": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "sample_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }

    def _pick(features, primary, alt):
      """First non-empty serialized-tensor string among {primary, alt}."""
      x = features[primary]
      return tf.where(tf.strings.length(x) > 0, x, features[alt])

    def prepare_sample(features):
      tgt_latents = tf.io.parse_tensor(_pick(features, "latents", "latent"), out_type=tf.float32)
      cond_latents = tf.io.parse_tensor(_pick(features, "cond_latents", "condition_latent"), out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)

      # N_lat = the cond temporal dim (axis 1 of [16, N_lat, h, w]).
      n_lat = tf.shape(cond_latents)[1]
      has_cam = tf.logical_and(
          tf.strings.length(features["cam_emb_con"]) > 0,
          tf.strings.length(features["cam_emb_tgt"]) > 0,
      )

      def _cam_from_tfrec():
        return (
            tf.io.parse_tensor(features["cam_emb_con"], out_type=tf.float32),
            tf.io.parse_tensor(features["cam_emb_tgt"], out_type=tf.float32),
        )

      def _cam_from_json():
        con, tgt = tf.py_function(
            func=_v2v_cam_from_json,
            inp=[features["sample_id"], n_lat, cam_local_root, cam_gcs_root],
            Tout=[tf.float32, tf.float32],
        )
        con.set_shape([None, 12])
        tgt.set_shape([None, 12])
        return con, tgt

      def _cam_error():
        con, tgt = tf.py_function(func=_v2v_cam_missing_error, inp=[features["sample_id"]], Tout=[tf.float32, tf.float32])
        con.set_shape([None, 12])
        tgt.set_shape([None, 12])
        return con, tgt

      # If cam_emb is in the tfrec -> use it directly; else -> compute from
      # camera.json (fallback) or fail loudly when the fallback is disabled.
      false_branch = _cam_from_json if cam_fallback else _cam_error
      cam_emb_con, cam_emb_tgt = tf.cond(has_cam, _cam_from_tfrec, false_branch)

      return {
          "cond_latents": cond_latents,
          "tgt_latents": tgt_latents,
          "cam_emb_con": cam_emb_con,
          "cam_emb_tgt": cam_emb_tgt,
          "encoder_hidden_states": encoder_hidden_states,
      }

    # Train/eval split by sample_id: built here (once per host) so the StaticHashTable
    # is captured by the tf.data graph and applied on the PARSED record BEFORE batching
    # (inside _make_tfrecord_iterator). None when v2v_split_mode == 'all' -> no filter.
    split_filter_fn = _v2v_make_split_filter(config)

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample,
        is_training=is_training,
        filter_fn=split_filter_fn,
    )
    # Reshard each loaded batch to the activation batch sharding ('data','fsdp')
    # OUTSIDE jit. The multihost loader hardcodes batch -> ALL mesh axes
    # (multihost_dataloading.py:_build_global_shape_and_sharding); with
    # per_device_batch_size<1 the in-step slice of that layout silently corrupts
    # the samples that must migrate across the context axis (repro'd 2026-07-06:
    # half the sliced batch returns 100% NaN). jax.device_put here is the
    # guaranteed-correct reshard; must match get_data_shardings (jit in_shardings).
    batch_sharding = jax.sharding.NamedSharding(mesh, P(("data", "fsdp")))

    def _reshard_batches(inner):
      for batch in inner:
        yield {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

    return _reshard_batches(data_iterator)

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(train_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    # No eval loss (dataset has no per-record timesteps). None => base loop skips it.
    return None


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    bsz = data["cond_latents"].shape[0]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    return v2v_concat_loss(model, data, new_rng, dropout_rng, timesteps, scheduler, config)

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = jax.tree_util.tree_reduce(
      lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))),
      grads,
      initializer=-1.0,
  )

  # Trainable (v2v adapter/self-attn) gradient norm, separated from the frozen
  # base. Flat-zero here would mean the camera/projector path is disconnected.
  substrings = tuple(v2v_trainable_substrings(config))

  def _trainable_sq(path, arr):
    ps = jax.tree_util.keystr(path)
    is_trainable = any(s in ps for s in substrings)
    return jnp.sum(arr.astype(jnp.float32) ** 2) if is_trainable else jnp.float32(0.0)

  v2v_grad_norm = jnp.sqrt(
      jax.tree_util.tree_reduce(
          lambda a, b: a + b,
          jax.tree_util.tree_map_with_path(_trainable_sq, grads),
          initializer=jnp.float32(0.0),
      )
  )

  new_state = state.apply_gradients(grads=grads)

  # Weight-magnitude probes on the POST-update trainable params. The zero-init
  # camera encoders' absmax IS the conditioning-learning signal: it must climb
  # past 2^-8 (=0.0039) toward the official ~0.13. Flatlining at 0.0039 = the
  # bf16 update-underflow stall (see cast_with_exclusion fp32 exclusion). Watch
  # weights/cam_encoder_con_absmax in tensorboard from step 0.
  def _family_absmax(params, keyword):
    def _m(path, arr):
      ps = jax.tree_util.keystr(path)
      return jnp.max(jnp.abs(arr)).astype(jnp.float32) if keyword in ps else jnp.float32(0.0)
    return jax.tree_util.tree_reduce(
        jnp.maximum, jax.tree_util.tree_map_with_path(_m, params), jnp.float32(0.0)
    )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
          "learning/v2v_grad_norm": v2v_grad_norm,
          "weights/cam_encoder_con_absmax": _family_absmax(new_state.params, "cam_encoder_con"),
          "weights/cam_encoder_tgt_absmax": _family_absmax(new_state.params, "cam_encoder_tgt"),
          "weights/projector_absmax": _family_absmax(new_state.params, "projector"),
          "weights/tokenizer_absmax": _family_absmax(new_state.params, "tokenizer"),
      },
      "scalars": {},
  }

  return new_state, scheduler_state, metrics, new_rng
