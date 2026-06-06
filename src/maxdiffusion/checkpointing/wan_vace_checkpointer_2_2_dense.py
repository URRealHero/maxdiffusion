"""Copyright 2025 Google LLC

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

from typing import Optional, Tuple

from maxdiffusion.checkpointing.wan_vace_checkpointer_2_1 import WanVaceCheckpointer2_1
from ..pipelines.wan.wan_vace_pipeline_2_2_dense import VaceWanPipeline2_2_Dense
from .. import max_logging


class WanVaceCheckpointer2_2_Dense(WanVaceCheckpointer2_1):
  """Orbax checkpointer for the WAN 2.2 Dense VACE trainer.

  Identical save/load logic to WanVaceCheckpointer2_1; only the pipeline
  class is swapped to VaceWanPipeline2_2_Dense.
  """

  def load_diffusers_checkpoint(self):
    return VaceWanPipeline2_2_Dense.from_pretrained(self.config)

  def load_checkpoint(self, step=None) -> Tuple[VaceWanPipeline2_2_Dense, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      max_logging.log("Loading WAN 2.2 Dense VACE pipeline from checkpoint")
      pipeline = VaceWanPipeline2_2_Dense.from_checkpoint(self.config, restored_checkpoint)
      if "opt_state" in restored_checkpoint.wan_state.keys():
        opt_state = restored_checkpoint.wan_state["opt_state"]
    else:
      max_logging.log("No checkpoint found, loading default WAN 2.2 Dense VACE pipeline.")
      pipeline = self.load_diffusers_checkpoint()
    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: VaceWanPipeline2_2_Dense, train_states):
    import json
    import orbax.checkpoint as ocp

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    max_logging.log(f"Saving VACE 2.2 dense checkpoint for step {train_step}")
    self.checkpoint_manager.save(
        train_step,
        args=ocp.args.Composite(
            wan_config=ocp.args.JsonSave(config_to_json(pipeline.transformer)),
            wan_state=ocp.args.StandardSave(train_states),
        ),
    )
    max_logging.log(f"VACE 2.2 dense checkpoint for step {train_step} saved.")
