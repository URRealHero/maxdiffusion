# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

import json
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from maxdiffusion.checkpointing.checkpointing_utils import add_sharding_to_struct, get_cpu_mesh_and_sharding
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
from maxdiffusion import max_logging, max_utils
from maxdiffusion.pipelines.wan.wan_pipeline_2_2_fun_camera import WanPipeline2_2_FunCamera


def _lora_substrings(config):
  return [s.strip() for s in str(getattr(config, "lora_trainable_param_substrings", "")).split(",") if s.strip()]


def _path_str(path):
  """Stable string for an nnx flat-state path tuple (save and overlay must agree)."""
  return "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path)


class WanCheckpointer2_2_FunCamera(WanCheckpointer):

  def _create_optimizer(self, model, config, learning_rate):
    """Optimizer with optional parameter freezing for LoRA / partial fine-tune.

    When `lora_trainable_param_substrings` is set (e.g. "lora_"), only params
    whose path contains one of the (comma-separated) substrings are trained;
    everything else is frozen via optax.set_to_zero (no optimizer state, so the
    frozen base costs no Adam memory). Empty substrings => full fine-tune.
    """
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    base_tx = max_utils.create_optimizer(config, learning_rate_scheduler)

    substrings = [s.strip() for s in str(getattr(config, "lora_trainable_param_substrings", "")).split(",") if s.strip()]
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
        f"LoRA/partial fine-tune: training {n_train}/{len(leaves)} param tensors "
        f"(substrings={substrings}); base frozen via set_to_zero."
    )
    tx = optax.multi_transform({"train": base_tx, "freeze": optax.set_to_zero()}, labels)
    return tx, learning_rate_scheduler

  def load_wan_configs_from_orbax(self, step: Optional[int]):
    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest WAN checkpoint step: {step}")
      if step is None:
        max_logging.log("No WAN checkpoint found.")
        return None, None
    max_logging.log(f"Loading WAN checkpoint from step {step}")

    mesh, replicated_sharding = get_cpu_mesh_and_sharding()
    metadatas = self.checkpoint_manager.item_metadata(step)
    state = metadatas.wan_state
    target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, state)
    with mesh:
      abstract_train_state_with_sharding = jax.tree_util.tree_map(add_sharding_to_struct, state, target_shardings)

    restored_checkpoint = self.checkpoint_manager.restore(
        step=step,
        args=ocp.args.Composite(
            wan_config=ocp.args.JsonRestore(),
            wan_state=ocp.args.StandardRestore(abstract_train_state_with_sharding),
        ),
    )
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    return WanPipeline2_2_FunCamera.from_pretrained(self.config)

  def _overlay_lora_checkpoint(self, lora_flat):
    """Load the base Fun-camera pipeline (PAI weights) and overlay saved LoRA adapters."""
    pipeline = self.load_diffusers_checkpoint()
    transformer = pipeline.transformer
    state = nnx.state(transformer, nnx.Param)
    flat = dict(nnx.to_flat_state(state))
    applied = 0
    for p, v in flat.items():
      sp = _path_str(p)
      if sp in lora_flat:
        target = v.value
        new_val = jnp.asarray(lora_flat[sp], dtype=target.dtype)
        sharding = getattr(target, "sharding", None)
        v.value = jax.device_put(new_val, sharding) if sharding is not None else new_val
        applied += 1
    if applied != len(lora_flat):
      missing = set(lora_flat) - {_path_str(p) for p in flat}
      raise ValueError(
          f"LoRA overlay mismatch: applied {applied}/{len(lora_flat)} saved adapters; "
          f"unmatched saved keys: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
      )
    nnx.update(transformer, nnx.from_flat_state(flat))
    max_logging.log(f"LoRA-only checkpoint: overlaid {applied} adapter tensors onto the PAI base.")
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_2_FunCamera, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      if restored_checkpoint.wan_config.get("_lora_only", False):
        # LoRA-only checkpoint: base from PAI + overlay adapters. No optimizer
        # state is saved, so resume starts a fresh optimizer (acceptable for LoRA).
        pipeline = self._overlay_lora_checkpoint(restored_checkpoint.wan_state)
      else:
        pipeline = WanPipeline2_2_FunCamera.from_checkpoint(self.config, restored_checkpoint)
        if "opt_state" in restored_checkpoint.wan_state.keys():
          opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default Fun camera pipeline.")
      pipeline = self.load_diffusers_checkpoint()
    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: WanPipeline2_2_FunCamera, train_states: dict):
    """Saves the training state and model configuration."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving checkpoint for step {train_step}")
    wan_config = config_to_json(pipeline.transformer)

    substrings = _lora_substrings(self.config)
    save_lora_only = bool(getattr(self.config, "save_lora_only", False)) and len(substrings) > 0
    if save_lora_only:
      # Save ONLY the trainable (LoRA) params, as a flat {path: array} pytree. The
      # frozen base is restored from PAI weights and the adapters overlaid on top,
      # so the checkpoint is a few MB instead of the full ~5B-param state.
      # `train_states` is the full TrainState when save_optimizer=True; take .params
      # so we never serialize optimizer tensors (which the strict overlay rejects).
      lora_source = getattr(train_states, "params", train_states)
      flat = dict(nnx.to_flat_state(lora_source))
      lora_flat = {
          _path_str(p): v.value
          for p, v in flat.items()
          if any(s in _path_str(p) for s in substrings)
      }
      if len(lora_flat) == 0:
        raise ValueError(
            f"save_lora_only=True but no params matched {substrings}; refusing to save an empty checkpoint."
        )
      wan_config["_lora_only"] = True
      wan_config["_lora_substrings"] = substrings
      state_save = ocp.args.StandardSave(lora_flat)
      max_logging.log(f"  LoRA-only checkpoint: {len(lora_flat)} adapter tensors (base frozen, not saved).")
    else:
      state_save = ocp.args.StandardSave(train_states)

    items = {
        "wan_config": ocp.args.JsonSave(wan_config),
        "wan_state": state_save,
    }

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
