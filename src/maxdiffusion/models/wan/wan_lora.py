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

"""LoRA adapter for WAN transformers — standard matrix LoRA.

Matches the official DiffSynth / Captain-Safari recipe:
  target modules: q, k, v, o  (self-attn + cross-attn), ffn.0, ffn.2
  rank:   32  (default)
  alpha:  32  (= rank → effective scale = 1.0)
  init:   A ~ N(0, 1/sqrt(rank)),  B = 0  → delta starts at zero

The adapter is applied INSIDE each targeted linear layer:
  output = W(x) + lora_B(lora_A(x)) * scale

Trainable param paths all contain "lora_", so
  lora_trainable_param_substrings: "lora_"
in the training config freezes all base weights automatically.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class WanLoRAAdapter(nnx.Module):
  """One (A, B) LoRA pair for a single linear layer.

  Parameters
  ----------
  in_features:  Input dimension of the target linear layer.
  out_features: Output dimension of the target linear layer.
  rank:         LoRA rank (e.g. 32).
  alpha:        LoRA alpha; effective scale = alpha / rank.
                Set alpha = rank for scale = 1.0 (DiffSynth default).
  dtype:        Activation dtype.
  weights_dtype: Parameter storage dtype.
  precision:    JAX matmul precision.
  rngs:         NNX RNG container.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      rank: int,
      alpha: float,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: jax.lax.Precision,
      rngs: nnx.Rngs,
  ):
    self.scale = alpha / rank

    # A: down-project to rank.  Kaiming/LeCun normal init (std = 1/sqrt(rank)).
    # B: up-project to out_features.  Zero init → delta starts at 0.
    # Both replicated (None, None) — rank is small (~32) so no sharding needed.
    self.lora_A = nnx.Linear(
        in_features=in_features,
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
        out_features=out_features,
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
    """Return lora_B(lora_A(x)) * scale."""
    return self.lora_B(self.lora_A(x)) * self.scale
