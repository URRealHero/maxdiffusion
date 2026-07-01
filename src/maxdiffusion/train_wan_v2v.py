"""
Copyright 2026 Google LLC

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

# V2V-1c entry point: fine-tunes Wan2.1-T2V-1.3B with the V2V-1a per-block camera
# scaffold on frame-concat (cond|tgt) latents. Trains only the camera/projector/
# self-attn params (HyDRA baseline). Reads cached TFRecords with
# (cond_latents, tgt_latents, cam_emb_con, cam_emb_tgt, encoder_hidden_states).

from typing import Sequence

from absl import app
import flax
import jax

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.train_utils import transformer_engine_context, validate_train_config


def train(config):
  from maxdiffusion.trainers.wan_v2v_concat_trainer import WanV2VConcatTrainer

  trainer = WanV2VConcatTrainer(config)
  trainer.start_training()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=True)
  config = pyconfig.config
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  train(config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
