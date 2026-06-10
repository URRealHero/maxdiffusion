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

# Baseline trainer for the WAN 2.2 **TI2V-5B dense** (single-transformer) model.
#
# The training objective is *identical* to WAN 2.1: flow-match on cached
# (latents, encoder_hidden_states) TFRecords with a single transformer —
# `noisy = apply_flow_match(noise, latents, t)`, `pred = model(noisy, t, text)`,
# weighted MSE against the flow-match target. That entire step (load_dataset,
# get_train_step, step_optimizer, eval_step) already lives in `WanTrainer`
# (trainers/wan_trainer.py), so we inherit it unchanged.
#
# The ONLY 2.2 difference is the checkpointer/pipeline:
#   * WanCheckpointer2_2_Dense -> WanPipeline2_2_Dense: ONE transformer
#     (the dense TI2V-5B) + the Wan 2.2 VAE (48-ch latents).
#   * (The non-dense WanCheckpointer2_2 -> WanPipeline2_2 loads the A14B MoE:
#      high/low-noise dual experts via `transformer` + `transformer_2`; that is
#      NOT the TI2V-5B and would fail on the 5B config — hence this dense trainer.)
#
# Note: `BaseWanTrainer.start_training` overwrites `pipeline.scheduler` with the
# `FlaxFlowMatchScheduler` (configured `flow_shift`) before the loop, so the
# inherited step's `sample_timesteps`/`apply_flow_match` calls resolve correctly.

from maxdiffusion.checkpointing.wan_checkpointer_2_2_dense import WanCheckpointer2_2_Dense
from maxdiffusion.trainers.wan_trainer import WanTrainer


class Wan2_2DenseTrainer(WanTrainer):
  """WAN 2.2 TI2V-5B dense baseline (no VACE / frame-concat / camera conditioning).

  Reuses the entire WAN 2.1 flow-match training step; only swaps in the dense
  2.2 checkpointer so the single TI2V-5B transformer + Wan 2.2 VAE are loaded.
  """

  def _get_checkpointer(self):
    return WanCheckpointer2_2_Dense(config=self.config)
