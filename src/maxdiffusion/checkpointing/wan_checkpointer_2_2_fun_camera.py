# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

import json
from typing import Optional, Tuple

import jax
import orbax.checkpoint as ocp

from maxdiffusion.checkpointing.checkpointing_utils import add_sharding_to_struct, get_cpu_mesh_and_sharding
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
from maxdiffusion import max_logging
from maxdiffusion.pipelines.wan.wan_pipeline_2_2_fun_camera import WanPipeline2_2_FunCamera


class WanCheckpointer2_2_FunCamera(WanCheckpointer):

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

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_2_FunCamera, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
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
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
        "wan_state": ocp.args.StandardSave(train_states),
    }

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
