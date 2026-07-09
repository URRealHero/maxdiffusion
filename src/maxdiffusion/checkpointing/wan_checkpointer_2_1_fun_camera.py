# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

# Checkpointer for Wan2.1-Fun-V1.1-1.3B-Control-Camera (model_type I2V-CC).
# Identical machinery to the 2.2 Fun-camera checkpointer (masked optimizer for
# LoRA/partial fine-tune, LoRA-only save/overlay); only the pipeline class and
# the restore return type differ.

from typing import Optional, Tuple

from maxdiffusion import max_logging
from maxdiffusion.checkpointing.wan_checkpointer_2_2_fun_camera import WanCheckpointer2_2_FunCamera
from maxdiffusion.pipelines.wan.wan_pipeline_2_1_fun_camera import WanPipeline2_1_FunCamera


class WanCheckpointer2_1_FunCamera(WanCheckpointer2_2_FunCamera):

  def load_diffusers_checkpoint(self):
    return WanPipeline2_1_FunCamera.from_pretrained(self.config)

  def load_checkpoint(self, step=None) -> Tuple[WanPipeline2_1_FunCamera, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      if restored_checkpoint.wan_config.get("_lora_only", False):
        pipeline = self._overlay_lora_checkpoint(restored_checkpoint.wan_state)
      else:
        pipeline = WanPipeline2_1_FunCamera.from_checkpoint(self.config, restored_checkpoint)
        if "opt_state" in restored_checkpoint.wan_state.keys():
          opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default Wan2.1 Fun camera pipeline.")
      pipeline = self.load_diffusers_checkpoint()
    return pipeline, opt_state, step
