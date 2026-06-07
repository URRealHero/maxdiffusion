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

"""LoRA-aware checkpointer for WAN 2.2 Dense.

Extends WanCheckpointer2_2_Dense with a LoRA sidecar file written alongside
every Orbax checkpoint.  The sidecar contains only the parameters whose path
includes "lora_" (i.e. lora_A / lora_B kernels), making it easy to share or
inspect the adapter weights independently of the full model.

Sidecar format: NumPy compressed archive (.npz) saved as
    <checkpoint_dir>/lora_weights_<step>.npz

If the `safetensors` package is available it is preferred (.safetensors).

GCS: when checkpoint_dir is a gs:// path the sidecar is written to /tmp first
and then uploaded via google-cloud-storage, matching the existing upload_blob
pattern used elsewhere in maxdiffusion.
"""

import os
import numpy as np
import jax

from maxdiffusion import max_logging
from maxdiffusion.checkpointing.wan_checkpointer_2_2_dense import WanCheckpointer2_2_Dense


class WanCheckpointerLoRA2_2Dense(WanCheckpointer2_2_Dense):
  """WAN 2.2 Dense checkpointer that also writes a LoRA weight sidecar."""

  def save_checkpoint(self, train_step, pipeline, train_states):
    super().save_checkpoint(train_step, pipeline, train_states)
    if jax.process_index() == 0:
      self._save_lora_sidecar(train_step, train_states)

  # ------------------------------------------------------------------
  def _save_lora_sidecar(self, train_step, train_states):
    """Extract lora_ params and write a small sidecar file."""
    # train_states is either a full TrainState (save_optimizer=True)
    # or state.params directly (save_optimizer=False).
    params = train_states.params if hasattr(train_states, "params") else train_states

    flat_items, _ = jax.tree_util.tree_flatten_with_path(params)
    lora_dict = {}
    for path, arr in flat_items:
      path_str = jax.tree_util.keystr(path)
      if "lora_" in path_str:
        # Normalize to a key safe for npz/safetensors: strip leading dot, use /
        flat_key = path_str.lstrip(".").replace(".", "/")
        lora_dict[flat_key] = np.array(arr, dtype=np.float32)

    if not lora_dict:
      max_logging.log("LoRA sidecar: no lora_ params found — skipping.")
      return

    filename = f"lora_weights_{train_step}"
    checkpoint_dir = self.config.checkpoint_dir

    # Try safetensors first (smaller, faster); fall back to .npz.
    try:
      from safetensors.numpy import save_file as st_save_file
      _save_sidecar(lora_dict, filename + ".safetensors", checkpoint_dir, use_safetensors=True)
      max_logging.log(f"LoRA sidecar ({len(lora_dict)} tensors) → {filename}.safetensors")
    except ImportError:
      _save_sidecar(lora_dict, filename + ".npz", checkpoint_dir, use_safetensors=False)
      max_logging.log(f"LoRA sidecar ({len(lora_dict)} tensors) → {filename}.npz")


def _save_sidecar(lora_dict, filename, checkpoint_dir, use_safetensors=False):
  """Write sidecar, handling both local and GCS checkpoint_dirs."""
  local_path = os.path.join("/tmp", filename)

  if use_safetensors:
    from safetensors.numpy import save_file as st_save_file
    st_save_file(lora_dict, local_path)
  else:
    np.savez(local_path, **lora_dict)

  if checkpoint_dir.startswith("gs://"):
    from maxdiffusion.max_utils import upload_blob
    gcs_dest = checkpoint_dir.rstrip("/") + "/" + filename
    upload_blob(gcs_dest, local_path)
    os.remove(local_path)
  else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    dest = os.path.join(checkpoint_dir, filename)
    os.replace(local_path, dest)
