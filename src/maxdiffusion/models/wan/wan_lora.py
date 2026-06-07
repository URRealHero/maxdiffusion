"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""LoRA adapter primitives for WAN transformers.

Design
------
Each ``WanLoRAAdapter`` is a single (A, B) low-rank pair that can be added
**in parallel** with any sub-block inside ``WanTransformerBlock``:

    output = sub_block(x) + lora_adapter(norm_x)

where ``norm_x`` is the layer-normed input to the sub-block.  This is the
standard parallel-adapter pattern used in many LoRA variants (it is equivalent
to adding LoRA on the residual stream at that position).

Three adapters are added per block:
  * ``lora_attn1`` — self-attention residual stream
  * ``lora_attn2`` — cross-attention residual stream
  * ``lora_ffn``   — feed-forward residual stream

Trainable param paths all start with ``lora_``, so the existing
``_make_trainable_grad_mask`` in ``wan_vace_trainer.py`` can freeze everything
else when ``lora_trainable_param_substrings = "lora_"``.

Checkpoint compatibility
------------------------
LoRA weights are saved under ``lora_attn1.lora_A.kernel``,
``lora_attn1.lora_B.kernel``, etc.  At inference the saved weights are merged
into the base model via the existing ``lora_nnx.merge_lora`` utility.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class WanLoRAAdapter(nnx.Module):
  """One (A, B) LoRA pair for a single residual-stream position.

  Parameters
  ----------
  dim:          Hidden dimension of the residual stream (e.g. 3072 for 5B).
  rank:         LoRA rank (e.g. 32).
  alpha:        LoRA alpha; effective scale = alpha / rank.
  dtype:        Activation dtype.
  weights_dtype: Parameter storage dtype.
  precision:    JAX matmul precision.
  rngs:         NNX RNG container.
  """

  def __init__(
      self,
      dim: int,
      rank: int,
      alpha: float,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: jax.lax.Precision,
      rngs: nnx.Rngs,
  ):
    self.scale = alpha / rank

    # A: down-project to rank.  Normal init (std = 1/sqrt(rank)).
    # B: up-project back to dim.  Zero init → delta starts at 0.
    # Both are replicated (None, None) because rank is small (~32).
    self.lora_A = nnx.Linear(
        in_features=dim,
        out_features=rank,
        use_bias=False,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(stddev=1.0 / rank**0.5),
            (None, None),
        ),
        rngs=rngs,
    )
    self.lora_B = nnx.Linear(
        in_features=rank,
        out_features=dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            (None, None),
        ),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Return lora_B(lora_A(x)) * scale.  x shape: (B, seq, dim)."""
    return self.lora_B(self.lora_A(x)) * self.scale
