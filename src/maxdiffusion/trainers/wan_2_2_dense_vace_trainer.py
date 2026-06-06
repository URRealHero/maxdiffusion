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

from maxdiffusion.checkpointing.wan_vace_checkpointer_2_2_dense import WanVaceCheckpointer2_2_Dense
from maxdiffusion.trainers.wan_vace_trainer import WanVaceTrainer


class Wan2_2DenseVaceTrainer(WanVaceTrainer):
  """VACE trainer for the single-transformer WAN 2.2 Dense (TI2V-5B) model.

  Data contract (same as WanVaceTrainer):
    latents              : target video latents  [B, C, F, H, W]
    conditioning_latents : VACE 96-ch context    [B, 96, F, H, W]
                           (16 inactive + 16 reactive + 64 mask)
    encoder_hidden_states: T5 text embeddings

  Trainable params: vace_* only by default (vace_trainable_param_substrings="vace").
  All base transformer weights are frozen.
  """

  def _get_checkpointer(self):
    return WanVaceCheckpointer2_2_Dense(config=self.config)
