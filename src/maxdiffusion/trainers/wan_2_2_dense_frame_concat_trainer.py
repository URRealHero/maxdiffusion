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

from maxdiffusion.checkpointing.wan_checkpointer_2_2_dense import WanCheckpointer2_2_Dense
from maxdiffusion.trainers.wan_frame_concat_trainer import WanFrameConcatTrainer


class Wan2_2DenseFrameConcatTrainer(WanFrameConcatTrainer):
  """Frame-concat TV2V trainer for the single-transformer WAN 2.2 dense model.

  The data contract is the same as WanFrameConcatTrainer:
    - latents: target video latents [C, F, H, W]
    - cond_latents: condition video latents [C, F, H, W]
    - encoder_hidden_states: text encoder embeddings

  The training objective concatenates condition and noisy target latents along
  latent time and computes loss only on the target slice.
  """

  def _get_checkpointer(self):
    return WanCheckpointer2_2_Dense(config=self.config)

