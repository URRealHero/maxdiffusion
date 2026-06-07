"""
Copyright 2025 Google LLC

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

import functools

from flax import nnx
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
import jaxopt
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.base_wan_trainer import BaseWanTrainer
import tensorflow as tf


class WanFrameConcatTrainer(BaseWanTrainer):

  def _get_checkpointer(self):
    return WanCheckpointer2_1(config=self.config)

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {"latents": data_sharding, "encoder_hidden_states": data_sharding, "cond_latents": data_sharding}
    return data_sharding

  def get_eval_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "cond_latents": data_sharding,
        "timesteps": data_sharding,
    }
    return data_sharding

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    """
    Load dataset - supports both real tfrecord and synthetic data.

    Args:
        mesh: JAX mesh for sharding
        pipeline: Optional WAN pipeline to extract dimensions from (for synthetic data)
        is_training: Whether this is for training or evaluation

    Returns:
        Data iterator
    """
    # Stages of training as described in the Wan 2.1 paper - https://arxiv.org/pdf/2503.20314
    # Image pre-training - txt2img 256px
    # Image-video joint training - stage 1. 256 px images and 192px 5 sec videos at fps=16
    # Image-video joint training - stage 2. 480px images and 480px 5 sec videos at fps=16
    # Image-video joint training - stage final. 720px images and 720px 5 sec videos at fps=16
    # prompt embeds shape: (1, 512, 4096)
    # For now, we will pass the same latents over and over
    # TODO - create a dataset

    config = self.config

    if config.dataset_type == "synthetic":
      raise ValueError("WanFrameConcatTrainer requires TFRecords with latents, encoder_hidden_states, and cond_latents")

    config = self.config
    if config.dataset_type != "tfrecord" and not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Wan 2.1 training only supports config.dataset_type set to tfrecords and config.cache_latents_text_encoder_outputs set to True"
      )
    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        "cond_latents": tf.io.FixedLenFeature([], tf.string),
    }

    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

    def prepare_sample_train(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      cond_latents = tf.io.parse_tensor(features["cond_latents"], out_type=tf.float32)
      return {"latents": latents, "encoder_hidden_states": encoder_hidden_states, "cond_latents": cond_latents}

    def prepare_sample_eval(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      cond_latents = tf.io.parse_tensor(features["cond_latents"], out_type=tf.float32)
      timesteps = features["timesteps"]
      return {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "cond_latents": cond_latents,
          "timesteps": timesteps,
      }

    data_iterator = make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample_train if is_training else prepare_sample_eval,
        is_training=is_training,
    )
    return data_iterator

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(train_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    return jax.jit(
        functools.partial(eval_step, scheduler=pipeline.scheduler, config=self.config),
        in_shardings=(state_shardings, eval_data_shardings, None, None),
        out_shardings=(None, None),
    )


def train_step(state, data, rng, scheduler_state, scheduler, config):
  return step_optimizer(state, data, rng, scheduler_state, scheduler, config)


def _finite_fraction(tree):
  leaves = jax.tree_util.tree_leaves(tree)
  finite = jnp.array(0.0, dtype=jnp.float32)
  total = jnp.array(0.0, dtype=jnp.float32)

  for leaf in leaves:
    is_finite = jnp.isfinite(leaf)
    finite = finite + jnp.sum(is_finite.astype(jnp.float32))
    total = total + jnp.asarray(leaf.size, dtype=jnp.float32)

  return finite / jnp.maximum(total, jnp.array(1.0, dtype=jnp.float32))


def _tree_max_abs(tree):
  return jax.tree_util.tree_reduce(
      lambda max_val, arr: jnp.maximum(max_val, jnp.max(jnp.abs(arr))),
      tree,
      initializer=jnp.array(0.0, dtype=jnp.float32),
  )


def _as_bool(value):
  return str(value).lower() == "true"


def _csv_config(value):
  if value is None:
    return ()
  return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _tree_path_to_str(path):
  try:
    return jax.tree_util.keystr(path)
  except Exception:  # pragma: no cover - defensive for older JAX path entries.
    pieces = []
    for entry in path:
      pieces.append(str(getattr(entry, "key", getattr(entry, "name", entry))))
    return ".".join(pieces)

def _tree_all_finite(tree):
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    return jnp.array(True)

  all_finite = jnp.array(True)
  for leaf in leaves:
    all_finite = all_finite & jnp.all(jnp.isfinite(leaf))

  return all_finite

def _make_trainable_grad_mask(grads, config):
  # lora_trainable_param_substrings takes priority when set (e.g. "lora_").
  lora_substrings = _csv_config(getattr(config, "lora_trainable_param_substrings", ""))
  if lora_substrings:
    substrings = lora_substrings
  else:
    substrings = _csv_config(getattr(config, "frame_concat_trainable_param_substrings", ""))
  if not substrings:
    return None

  def make_mask(path, grad):
    path_str = _tree_path_to_str(path)
    is_trainable = any(substring in path_str for substring in substrings)
    return jnp.ones_like(grad, dtype=jnp.bool_) if is_trainable else jnp.zeros_like(grad, dtype=jnp.bool_)

  path_leaves, treedef = jax.tree_util.tree_flatten_with_path(grads)
  mask_leaves = [make_mask(path, grad) for path, grad in path_leaves]
  return jax.tree_util.tree_unflatten(treedef, mask_leaves)


def _apply_trainable_grad_mask(grads, trainable_mask):
  if trainable_mask is None:
    return grads
  return jax.tree_util.tree_map(lambda grad, mask: jnp.where(mask, grad, jnp.zeros_like(grad)), grads, trainable_mask)


def _restore_frozen_params(old_state, new_state, trainable_mask):
  if trainable_mask is None:
    return new_state
  restored_params = jax.tree_util.tree_map(
      lambda old_param, new_param, mask: jnp.where(mask, new_param, old_param),
      old_state.params,
      new_state.params,
      trainable_mask,
  )
  return new_state.replace(params=restored_params)


def _mask_fraction(trainable_mask):
  if trainable_mask is None:
    return jnp.array(1.0, dtype=jnp.float32)

  trainable = jnp.array(0.0, dtype=jnp.float32)
  total = jnp.array(0.0, dtype=jnp.float32)

  for leaf in jax.tree_util.tree_leaves(trainable_mask):
    trainable = trainable + jnp.sum(leaf.astype(jnp.float32))
    total = total + jnp.asarray(leaf.size, dtype=jnp.float32)

  return trainable / jnp.maximum(total, jnp.array(1.0, dtype=jnp.float32))


def _debug_print_trainable_paths(params, config):
  if not _as_bool(getattr(config, "frame_concat_print_trainable_params", False)):
    return
  lora_substrings = _csv_config(getattr(config, "lora_trainable_param_substrings", ""))
  substrings = lora_substrings if lora_substrings else _csv_config(getattr(config, "frame_concat_trainable_param_substrings", ""))
  if not substrings or jax.process_index() != 0:
    return
  print("FrameConcat trainable parameter substrings:", ",".join(substrings))
  matched = []
  for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
    path_str = _tree_path_to_str(path)
    if any(substring in path_str for substring in substrings):
      matched.append((path_str, getattr(value, "shape", None)))
  print(f"FrameConcat trainable parameter leaves: {len(matched)}")
  for path_str, shape in matched[:200]:
    print(f"  trainable {path_str} shape={shape}")
  if len(matched) > 200:
    print(f"  ... {len(matched) - 200} more trainable leaves omitted")

def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, noise_rng, timestep_rng, dropout_rng, new_rng = jax.random.split(rng, num=5)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on, :]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    cond_latents = data["cond_latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)

    bsz = latents.shape[0]
    cond_frames = cond_latents.shape[2]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    timesteps = jnp.reshape(timesteps, (bsz,))
    noise = jax.random.normal(key=noise_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, _ = scheduler.apply_flow_match(noise, latents, timesteps)
    # Match DiffSynth/ReCamMaster/HyDRA: weight each sample by the fixed per-timestep
    # weight precomputed over the full 1000-step grid (global normalization), looked up
    # by timestep. apply_flow_match's own weight renormalizes within the current batch,
    # which forces the batch-min-timestep sample to weight 0 and adds batch-dependent
    # variance -- a divergence from the reference recipe.
    training_weight = scheduler.training_weight(scheduler_state, timesteps)
    hidden_states = jnp.concatenate([cond_latents, noisy_latents], axis=2)
    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=hidden_states,
          timestep=timesteps,
          encoder_hidden_states=encoder_hidden_states,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
      )
      model_pred = model_pred[:, :, cond_frames:]

    with jax.named_scope("loss"):
      loss_tensor_unweighted = (training_target - model_pred) ** 2
      loss_tensor = loss_tensor_unweighted
      if not config.disable_training_weights:
        training_weight_for_loss = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
        loss_tensor = loss_tensor * training_weight_for_loss
      loss = jnp.mean(loss_tensor)

    debug_metrics = {
        "debug/latents_finite_frac": jnp.mean(jnp.isfinite(latents)),
        "debug/cond_latents_finite_frac": jnp.mean(jnp.isfinite(cond_latents)),
        "debug/encoder_hidden_states_finite_frac": jnp.mean(jnp.isfinite(encoder_hidden_states)),
        "debug/noise_finite_frac": jnp.mean(jnp.isfinite(noise)),
        "debug/noisy_latents_finite_frac": jnp.mean(jnp.isfinite(noisy_latents)),
        "debug/hidden_states_finite_frac": jnp.mean(jnp.isfinite(hidden_states)),
        "debug/model_pred_finite_frac": jnp.mean(jnp.isfinite(model_pred)),
        "debug/training_target_finite_frac": jnp.mean(jnp.isfinite(training_target)),
        "debug/loss_tensor_finite_frac": jnp.mean(jnp.isfinite(loss_tensor)),
        "debug/loss_unweighted_mean": jnp.mean(loss_tensor_unweighted),
        "debug/timesteps_min": jnp.min(timesteps),
        "debug/timesteps_max": jnp.max(timesteps),
        "debug/training_weight_min": jnp.min(training_weight),
        "debug/training_weight_max": jnp.max(training_weight),
        "debug/latents_max_abs": jnp.max(jnp.abs(latents)),
        "debug/cond_latents_max_abs": jnp.max(jnp.abs(cond_latents)),
        "debug/encoder_hidden_states_max_abs": jnp.max(jnp.abs(encoder_hidden_states)),
        "debug/noisy_latents_max_abs": jnp.max(jnp.abs(noisy_latents)),
        "debug/model_pred_max_abs": jnp.max(jnp.abs(model_pred)),
        "debug/training_target_max_abs": jnp.max(jnp.abs(training_target)),
    }

    return loss, debug_metrics

  _debug_print_trainable_paths(state.params, config)
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, debug_metrics), grads = grad_fn(state.params)
  raw_max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  raw_max_abs_grad = _tree_max_abs(grads)
  raw_grads_finite_frac = _finite_fraction(grads)
  raw_grads_all_finite = _tree_all_finite(grads)
  trainable_grad_mask = _make_trainable_grad_mask(grads, config)
  trainable_param_fraction = _mask_fraction(trainable_grad_mask)
  grads = _apply_trainable_grad_mask(grads, trainable_grad_mask)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = _tree_max_abs(grads)
  grads_finite_frac = _finite_fraction(grads)

  grads_all_finite = _tree_all_finite(grads)
  update_is_finite = jnp.isfinite(loss) & grads_all_finite
  skip_nonfinite_update = _as_bool(getattr(config, "frame_concat_skip_nonfinite_update", False))

  if skip_nonfinite_update:
    new_state = jax.lax.cond(
        update_is_finite,
        lambda _: state.apply_gradients(grads=grads),
        lambda _: state,
        operand=None,
    )
  else:
    new_state = state.apply_gradients(grads=grads)

  new_state = _restore_frozen_params(state, new_state, trainable_grad_mask)

  params_finite_frac_after_update = _finite_fraction(new_state.params)
  params_all_finite_after_update = _tree_all_finite(new_state.params)
  params_max_abs_after_update = _tree_max_abs(new_state.params)
  update_skipped = jnp.asarray(skip_nonfinite_update, dtype=jnp.bool_) & ~update_is_finite

  if _as_bool(getattr(config, "frame_concat_debug_print", False)):
    jax.debug.print(
        "FrameConcat debug: loss={loss} lat={lat} cond={cond} hidden={hidden} pred={pred} "
        "target={target} loss_tensor={loss_tensor} w=[{wmin},{wmax}] "
        "raw_grad={raw_grad} raw_grad_max={raw_grad_max} raw_grad_all={raw_grad_all} trainable_frac={trainable_frac} "
        "grad={grad} grad_max={grad_max} params_after={params_after} pred_max={pred_max} update_ok={update_ok} skipped={skipped}",
        loss=loss,
        lat=debug_metrics["debug/latents_finite_frac"],
        cond=debug_metrics["debug/cond_latents_finite_frac"],
        hidden=debug_metrics["debug/hidden_states_finite_frac"],
        pred=debug_metrics["debug/model_pred_finite_frac"],
        target=debug_metrics["debug/training_target_finite_frac"],
        loss_tensor=debug_metrics["debug/loss_tensor_finite_frac"],
        wmin=debug_metrics["debug/training_weight_min"],
        wmax=debug_metrics["debug/training_weight_max"],
        raw_grad=raw_grads_finite_frac,
        raw_grad_max=raw_max_abs_grad,
        raw_grad_all=raw_grads_all_finite,
        trainable_frac=trainable_param_fraction,
        grad=grads_finite_frac,
        grad_max=max_abs_grad,
        params_after=params_finite_frac_after_update,
        pred_max=debug_metrics["debug/model_pred_max_abs"],
        update_ok=update_is_finite,
        skipped=update_skipped,
    )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
          "debug/raw_max_grad_norm": raw_max_grad_norm,
          "debug/raw_max_abs_grad": raw_max_abs_grad,
          "debug/raw_grads_finite_frac": raw_grads_finite_frac,
          "debug/raw_grads_all_finite": raw_grads_all_finite.astype(jnp.float32),
          "debug/trainable_param_fraction": trainable_param_fraction,
          "debug/grads_finite_frac": grads_finite_frac,
          "debug/grads_all_finite": grads_all_finite.astype(jnp.float32),
          "debug/params_finite_frac_after_update": params_finite_frac_after_update,
          "debug/params_max_abs_after_update": params_max_abs_after_update,
          "debug/params_all_finite_after_update": params_all_finite_after_update.astype(jnp.float32),
          "debug/update_is_finite": update_is_finite.astype(jnp.float32),
          "debug/update_skipped": update_skipped.astype(jnp.float32),
          **debug_metrics,
      },
      "scalars": {},
  }

  return new_state, scheduler_state, metrics, new_rng


def eval_step(state, data, rng, scheduler_state, scheduler, config):
  """
  Computes the evaluation loss for a single batch without updating model weights.
  """

  # The loss function logic is identical to training. We are evaluating the model's
  # ability to perform its core training objective (e.g., denoising).
  def loss_fn(params, latents, cond_latents, encoder_hidden_states, timesteps, rng):
    # Reconstruct the model from its definition and parameters
    model = nnx.merge(state.graphdef, params, state.rest_of_state)

    cond_frames = cond_latents.shape[2]
    noise = jax.random.normal(key=rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)
    hidden_states = jnp.concatenate([cond_latents, noisy_latents], axis=2)
    # Get the model's prediction for the combined condition+target sequence,
    # then train only on the target slice.
    model_pred = model(
        hidden_states=hidden_states,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=True,
    )
    model_pred = model_pred[:, :, cond_frames:]

    # Calculate the loss against the target
    loss = (training_target - model_pred) ** 2
    if not config.disable_training_weights:
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight

    # Calculate the mean loss per sample across all non-batch dimensions.
    loss = loss.reshape(loss.shape[0], -1).mean(axis=1)

    return loss

  # --- Key Difference from train_step ---
  # Directly compute the loss without calculating gradients.
  # The model's state.params are used but not updated.
  # TODO(coolkp): Explore optimizing the creation of PRNGs in a vmap or statically outside of the loop
  bs = len(data["latents"])
  single_batch_size = config.global_batch_size_to_train_on
  losses = jnp.zeros(bs)
  for i in range(0, bs, single_batch_size):
    start = i
    end = min(i + single_batch_size, bs)
    latents = data["latents"][start:end, :].astype(config.weights_dtype)
    cond_latents = data["cond_latents"][start:end, :].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"][start:end, :].astype(config.weights_dtype)
    timesteps = data["timesteps"][start:end].astype("int64")
    _, new_rng = jax.random.split(rng, num=2)
    loss = loss_fn(state.params, latents, cond_latents, encoder_hidden_states, timesteps, new_rng)
    losses = losses.at[start:end].set(loss)

  # Structure the metrics for logging and aggregation
  metrics = {"scalar": {"learning/eval_loss": losses}}

  # Return the computed metrics and the new RNG key for the next eval step
  return metrics, new_rng
