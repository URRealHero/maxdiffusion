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
from maxdiffusion.checkpointing.wan_vace_checkpointer_2_1 import WanVaceCheckpointer2_1
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.base_wan_trainer import BaseWanTrainer
import tensorflow as tf


# ---------------------------------------------------------------------------
# Gradient-masking utilities (shared with frame-concat trainer pattern)
# ---------------------------------------------------------------------------

def _csv_config(value):
  if value is None:
    return ()
  return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _as_bool(value):
  return str(value).lower() == "true"


def _tree_path_to_str(path):
  try:
    return jax.tree_util.keystr(path)
  except Exception:
    pieces = []
    for entry in path:
      pieces.append(str(getattr(entry, "key", getattr(entry, "name", entry))))
    return ".".join(pieces)


def _make_trainable_grad_mask(grads, config):
  """Zero gradients for params whose path doesn't match vace_trainable_param_substrings.

  Defaults to "vace" so only VACE branch params are updated.
  Set vace_trainable_param_substrings="" in the config to train all params.
  """
  substrings = _csv_config(getattr(config, "vace_trainable_param_substrings", "vace"))
  if not substrings:
    return None

  def make_mask(path, grad):
    path_str = _tree_path_to_str(path)
    is_trainable = any(s in path_str for s in substrings)
    return jnp.ones_like(grad, dtype=jnp.bool_) if is_trainable else jnp.zeros_like(grad, dtype=jnp.bool_)

  path_leaves, treedef = jax.tree_util.tree_flatten_with_path(grads)
  mask_leaves = [make_mask(path, grad) for path, grad in path_leaves]
  return jax.tree_util.tree_unflatten(treedef, mask_leaves)


def _apply_trainable_grad_mask(grads, mask):
  if mask is None:
    return grads
  return jax.tree_util.tree_map(
      lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), grads, mask
  )


def _restore_frozen_params(old_state, new_state, mask):
  """After apply_gradients, restore any frozen params to their pre-update values."""
  if mask is None:
    return new_state
  restored = jax.tree_util.tree_map(
      lambda old, new, m: jnp.where(m, new, old),
      old_state.params, new_state.params, mask,
  )
  return new_state.replace(params=restored)


def _mask_fraction(mask):
  if mask is None:
    return jnp.array(1.0, dtype=jnp.float32)
  trainable = jnp.array(0.0, dtype=jnp.float32)
  total = jnp.array(0.0, dtype=jnp.float32)
  for leaf in jax.tree_util.tree_leaves(mask):
    trainable = trainable + jnp.sum(leaf.astype(jnp.float32))
    total = total + jnp.asarray(leaf.size, dtype=jnp.float32)
  return trainable / jnp.maximum(total, jnp.array(1.0, dtype=jnp.float32))


def _debug_print_trainable_paths(params, config):
  substrings = _csv_config(getattr(config, "vace_trainable_param_substrings", "vace"))
  if not substrings or not _as_bool(getattr(config, "vace_print_trainable_params", False)):
    return
  if jax.process_index() != 0:
    return
  print("VACE trainable param substrings:", ",".join(substrings))
  matched = []
  for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
    path_str = _tree_path_to_str(path)
    if any(s in path_str for s in substrings):
      matched.append((path_str, getattr(value, "shape", None)))
  print(f"VACE trainable parameter leaves: {len(matched)}")
  for path_str, shape in matched[:200]:
    print(f"  trainable {path_str} shape={shape}")
  if len(matched) > 200:
    print(f"  ... {len(matched) - 200} more omitted")


class WanVaceTrainer(BaseWanTrainer):

  def _get_checkpointer(self):
    return WanVaceCheckpointer2_1(config=self.config)

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "conditioning_latents": data_sharding,
    }
    return data_sharding

  def get_eval_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    data_sharding = {
        "latents": data_sharding,
        "encoder_hidden_states": data_sharding,
        "timesteps": data_sharding,
        "conditioning_latents": data_sharding,
    }
    return data_sharding

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config

    # If using synthetic data
    if config.dataset_type == "synthetic":
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          pipeline=pipeline,  # Pass pipeline to extract dimensions
          is_training=is_training,
      )

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

    def make_vace_conditioning(cond_latents):
      # VACE conditioning: inactive(C) + reactive(C) + mask(64) channels.
      # For WAN 2.2 TI2V-5B with 48-ch latents: 48+48+64 = 160 = vace_in_channels.
      # For HM-World TV2V smoke data, approximate this as inactive=zeros,
      # reactive=cond_latents, and an all-active mask.
      inactive_latents = tf.zeros_like(cond_latents)
      mask = tf.ones_like(cond_latents[:1])
      mask = tf.tile(mask, [64, 1, 1, 1])
      return tf.concat([inactive_latents, cond_latents, mask], axis=0)

    def prepare_sample_train(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      cond_latents = tf.io.parse_tensor(features["cond_latents"], out_type=tf.float32)
      conditioning_latents = make_vace_conditioning(cond_latents)
      return {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "conditioning_latents": conditioning_latents,
      }

    def prepare_sample_eval(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      cond_latents = tf.io.parse_tensor(features["cond_latents"], out_type=tf.float32)
      conditioning_latents = make_vace_conditioning(cond_latents)
      timesteps = features["timesteps"]
      return {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "conditioning_latents": conditioning_latents,
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


def step_optimizer(state, data, rng, scheduler_state, scheduler, config):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    data[k] = v[: config.global_batch_size_to_train_on, :]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    control_hidden_states = data["conditioning_latents"].astype(config.weights_dtype)

    bsz = latents.shape[0]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    timesteps = jnp.reshape(timesteps, (bsz,))
    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, _ = scheduler.apply_flow_match(noise, latents, timesteps)
    # Global-grid weight lookup matches DiffSynth/HyDRA (avoids per-batch renorm
    # that forces the batch-min-timestep sample to weight 0).
    training_weight = scheduler.training_weight(scheduler_state, timesteps)
    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=noisy_latents,
          timestep=timesteps,
          encoder_hidden_states=encoder_hidden_states,
          control_hidden_states=control_hidden_states,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
      )

    with jax.named_scope("loss"):
      model_pred = model_pred.astype(jnp.float32)
      training_target = training_target.astype(jnp.float32)
      loss_tensor = (training_target - model_pred) ** 2
      if not config.disable_training_weights:
        training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
        loss_tensor = loss_tensor * training_weight
      loss = jnp.mean(loss_tensor)

    aux = {
        "latents_finite_frac": jnp.mean(jnp.isfinite(latents)),
        "encoder_hidden_states_finite_frac": jnp.mean(jnp.isfinite(encoder_hidden_states)),
        "control_hidden_states_finite_frac": jnp.mean(jnp.isfinite(control_hidden_states)),
        "noisy_latents_finite_frac": jnp.mean(jnp.isfinite(noisy_latents)),
        "training_target_finite_frac": jnp.mean(jnp.isfinite(training_target)),
        "training_weight_finite_frac": jnp.mean(jnp.isfinite(training_weight)),
        "model_pred_finite_frac": jnp.mean(jnp.isfinite(model_pred)),
        "loss_tensor_finite_frac": jnp.mean(jnp.isfinite(loss_tensor)),
        "latents_max_abs": jnp.max(jnp.abs(latents.astype(jnp.float32))),
        "control_hidden_states_max_abs": jnp.max(jnp.abs(control_hidden_states.astype(jnp.float32))),
        "noisy_latents_max_abs": jnp.max(jnp.abs(noisy_latents.astype(jnp.float32))),
        "training_target_max_abs": jnp.max(jnp.abs(training_target.astype(jnp.float32))),
        "training_weight_max": jnp.max(training_weight.astype(jnp.float32)),
        "model_pred_max_abs": jnp.max(jnp.abs(model_pred.astype(jnp.float32))),
        "loss_tensor_max": jnp.max(loss_tensor.astype(jnp.float32)),
        "timestep_min": jnp.min(timesteps.astype(jnp.float32)),
        "timestep_max": jnp.max(timesteps.astype(jnp.float32)),
    }

    if str(getattr(config, "vace_debug_print", False)).lower() == "true":
      jax.debug.print(
          "VACE debug: loss={loss} lat={lat} cond={cond} noisy={noisy} target={target} "
          "pred={pred} loss_tensor={loss_tensor} pred_max={pred_max} cond_max={cond_max} t=[{tmin},{tmax}]",
          loss=loss,
          lat=aux["latents_finite_frac"],
          cond=aux["control_hidden_states_finite_frac"],
          noisy=aux["noisy_latents_finite_frac"],
          target=aux["training_target_finite_frac"],
          pred=aux["model_pred_finite_frac"],
          loss_tensor=aux["loss_tensor_finite_frac"],
          pred_max=aux["model_pred_max_abs"],
          cond_max=aux["control_hidden_states_max_abs"],
          tmin=aux["timestep_min"],
          tmax=aux["timestep_max"],
      )

    return loss, aux

  debug_forward_only = _as_bool(getattr(config, "vace_debug_forward_only", False))
  if debug_forward_only:
    loss, aux = loss_fn(state.params)
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "debug/vace_forward_only": jnp.array(1.0),
            **{f"debug/{k}": v for k, v in aux.items()},
        },
        "scalars": {},
    }
    return state, scheduler_state, metrics, new_rng

  _debug_print_trainable_paths(state.params, config)
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, aux), grads = grad_fn(state.params)

  raw_max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  raw_max_abs_grad = jax.tree_util.tree_reduce(
      lambda m, a: jnp.maximum(m, jnp.max(jnp.abs(a))), grads, initializer=jnp.array(0.0)
  )

  trainable_mask = _make_trainable_grad_mask(grads, config)
  trainable_param_fraction = _mask_fraction(trainable_mask)
  grads = _apply_trainable_grad_mask(grads, trainable_mask)

  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = jax.tree_util.tree_reduce(
      lambda m, a: jnp.maximum(m, jnp.max(jnp.abs(a))), grads, initializer=jnp.array(0.0)
  )

  new_state = state.apply_gradients(grads=grads)
  # Restore frozen params that the optimizer may have perturbed via momentum.
  new_state = _restore_frozen_params(state, new_state, trainable_mask)

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
          "debug/raw_max_grad_norm": raw_max_grad_norm,
          "debug/raw_max_abs_grad": raw_max_abs_grad,
          "debug/trainable_param_fraction": trainable_param_fraction,
          **{f"debug/{k}": v for k, v in aux.items()},
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
  def loss_fn(
      params,
      latents,
      encoder_hidden_states,
      timesteps,
      rng,
      conditioning_latents,
  ):
    # Reconstruct the model from its definition and parameters
    model = nnx.merge(state.graphdef, params, state.rest_of_state)

    noise = jax.random.normal(key=rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(noise, latents, timesteps)
    # Get the model's prediction
    model_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        control_hidden_states=conditioning_latents,
        deterministic=True,
    )

    # Calculate the loss against the target
    model_pred = model_pred.astype(jnp.float32)
    training_target = training_target.astype(jnp.float32)

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
    encoder_hidden_states = data["encoder_hidden_states"][start:end, :].astype(config.weights_dtype)
    conditioning_latents = data["conditioning_latents"][start:end, :].astype(config.weights_dtype)
    timesteps = data["timesteps"][start:end].astype("int64")
    _, new_rng = jax.random.split(rng, num=2)
    loss = loss_fn(
        state.params,
        latents,
        encoder_hidden_states,
        timesteps,
        new_rng,
        conditioning_latents,
    )
    losses = losses.at[start:end].set(loss)

  # Structure the metrics for logging and aggregation
  metrics = {"scalar": {"learning/eval_loss": losses}}

  # Return the computed metrics and the new RNG key for the next eval step
  return metrics, new_rng
