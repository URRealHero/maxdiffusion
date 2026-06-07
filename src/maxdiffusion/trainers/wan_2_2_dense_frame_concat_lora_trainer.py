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

"""LoRA trainer for the WAN 2.2 Dense (TI2V-5B) frame-concat pipeline.

Usage
-----
Add to your training config::

    lora_rank: 32
    lora_alpha: 0       # 0 (or any value <= 0) is a sentinel meaning "use rank",
                        # giving effective scale = alpha/rank = 1.0 (DiffSynth default).
                        # Set lora_alpha: 32 to get the same scale explicitly; a
                        # value like 1.0 would yield scale 1/32 -- a 32x weaker delta.
    lora_trainable_param_substrings: "lora_"

This freezes all base transformer weights and trains only the lora_A / lora_B
adapter kernels (one (A,B) pair per self-attn, cross-attn, and FFN residual
stream in every WanTransformerBlock).

A compact LoRA sidecar (lora_weights_<step>.npz or .safetensors) is written
to checkpoint_dir at every checkpoint save, in addition to the full Orbax
checkpoint.  The sidecar is self-contained and can be loaded independently for
LoRA merging or inference without the full 5 B parameter checkpoint.
"""

from maxdiffusion.checkpointing.wan_checkpointer_2_2_dense_lora import WanCheckpointerLoRA2_2Dense
from maxdiffusion.trainers.wan_2_2_dense_frame_concat_trainer import Wan2_2DenseFrameConcatTrainer


class Wan2_2DenseFrameConcatLoRATrainer(Wan2_2DenseFrameConcatTrainer):
  """Frame-concat trainer for LoRA fine-tuning of WAN 2.2 Dense (TI2V-5B).

  Inherits all training logic from Wan2_2DenseFrameConcatTrainer; overrides
  only the checkpointer so that a LoRA weight sidecar is written alongside each
  Orbax checkpoint.

  Grad masking (parameter freezing) is handled by the module-level
  _make_trainable_grad_mask in wan_frame_concat_trainer.py, which reads
  lora_trainable_param_substrings from the config and zeros gradients for any
  parameter whose path does not contain one of the listed substrings.
  """

  def _get_checkpointer(self):
    return WanCheckpointerLoRA2_2Dense(config=self.config)
