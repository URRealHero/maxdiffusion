# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

"""JAX/flax.nnx port of Captain-Safari's pose-conditioned MemoryRetriever
(the "librarian"): given the query camera pose + the previous keyframes' poses
and their StreamVGGT memory features, predict the memory features for the query
viewpoint. Output feeds dit.memory_emb -> per-block memory cross-attention.

Faithful port of captain_safari/diffsynth/models/wan_video_dit.py:
  precompute_freqs_cis(_3d), rope_apply, RMSNorm, SelfAttention/JointSelfAttention,
  CrossAttention, UnifiedJointBlock, RetrievalBlock, MemoryRetriever.

Shapes (our case): 4 keyframes (T), 4 StreamVGGT layers (L), 21x37 grid (H,W);
per-frame tokens = L*(5 + H*W) = 4*782 = 3128; joint sequence = 1 + 3128 = 3129.
Retriever dim=1024, heads=8 -> head_dim=128. Inference-only (no CE-loss head).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# ------------------------- 3D RoPE -------------------------
# Rotary position encoding, but the head_dim is split across three axes —
# time (f), height (h), width (w) — so each token's position is (frame, row, col).
# Memory tokens get real positions; the query token gets identity (position 0).

def precompute_freqs_cis(dim: int, end: int = 2048, theta: float = 10000.0) -> np.ndarray:
  """1D RoPE freqs as complex [end, dim//2] (numpy complex128, like CS's torch.polar)."""
  freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2].astype(np.float64) / dim))
  freqs = np.outer(np.arange(end, dtype=np.float64), freqs)  # [end, dim//2]
  return np.exp(1j * freqs)  # complex128, |.|=1 (== torch.polar(ones, freqs))


def precompute_freqs_cis_3d(head_dim: int, end: int = 2048, theta: float = 10000.0):
  """Split head_dim into time/height/width sub-rotations (CS precompute_freqs_cis_3d)."""
  f = precompute_freqs_cis(head_dim - 2 * (head_dim // 3), end, theta)  # [end, f//2]
  h = precompute_freqs_cis(head_dim // 3, end, theta)                   # [end, h//2]
  w = precompute_freqs_cis(head_dim // 3, end, theta)                   # [end, w//2]
  return f, h, w


@functools.lru_cache(maxsize=8)
def _cached_freqs_cis_3d(head_dim: int, end: int, theta: float):
  """Memoized numpy 3D RoPE tables. Computed lazily at call time (not stored as a
  module attribute) so they stay concrete numpy through nnx.eval_shape/split/merge —
  storing arrays as plain attributes gets them abstracted to ShapeDtypeStruct."""
  return precompute_freqs_cis_3d(head_dim, end, theta)


def build_3d_freqs_unified(T, L, H, W, f_freqs, h_freqs, w_freqs,
                           include_pose_token=True, use_time_encoding=True) -> np.ndarray:
  """Per-token 3D freqs for the joint sequence -> [seq, head_dim//2] complex.

  Token layout per time step t: [pose_token] + L * ([5 special] + [H*W image]).
  - time index = t for memory (use_time_encoding=True), 0 for the query.
  - special tokens (1 camera + 4 register): spatial (h,w)=0.
  - image tokens: real (h,w) on the 21x37 grid. (Faithful to CS.)
  """
  rows = []
  def combo(f_i, h_i, w_i):
    return np.concatenate([f_freqs[f_i:f_i + 1], h_freqs[h_i:h_i + 1], w_freqs[w_i:w_i + 1]], axis=-1)
  for t in range(T):
    f_t = (min(t, len(f_freqs) - 1) if use_time_encoding else 0)
    if include_pose_token:
      rows.append(combo(f_t, 0, 0))                       # pose token: spatial 0
    for _ in range(L):
      for _ in range(5):                                  # camera + 4 registers
        rows.append(combo(f_t, 0, 0))
      for h in range(H):
        for w in range(W):
          rows.append(combo(f_t, min(h, len(h_freqs) - 1), min(w, len(w_freqs) - 1)))
  return np.concatenate(rows, axis=0)  # [seq, head_dim//2]


def rope_apply(x: jax.Array, freqs: jax.Array, num_heads: int) -> jax.Array:
  """Apply rotary via complex multiply. x:[B,S,dim], freqs:[S, head_dim//2] complex."""
  b, s, dim = x.shape
  head_dim = dim // num_heads
  xc = x.reshape(b, s, num_heads, head_dim // 2, 2).astype(jnp.float32)
  xc = jax.lax.complex(xc[..., 0], xc[..., 1])           # [b,s,n,head_dim//2]
  out = xc * freqs[None, :, None, :]                      # broadcast over batch + heads
  out = jnp.stack([out.real, out.imag], axis=-1).reshape(b, s, dim)
  return out.astype(x.dtype)


# ------------------------- norms & attention -------------------------

class RMSNorm(nnx.Module):
  """CS RMSNorm: normalize in fp32, scale by a learned per-channel weight."""
  def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
    self.eps = eps
    self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))

  def __call__(self, x):
    xf = x.astype(jnp.float32)
    n = xf * jax.lax.rsqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + self.eps)
    return (n * self.weight.value).astype(x.dtype)


def _mha(q, k, v, num_heads):
  """Plain multi-head scaled-dot-product attention. q/k/v: [B, S, dim]."""
  b, sq, dim = q.shape
  hd = dim // num_heads
  sh = lambda t: t.reshape(t.shape[0], t.shape[1], num_heads, hd).transpose(0, 2, 1, 3)
  qh, kh, vh = sh(q), sh(k), sh(v)                        # [B, n, S, hd]
  attn = jnp.einsum("bnqd,bnkd->bnqk", qh, kh) / jnp.sqrt(hd).astype(q.dtype)
  attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
  out = jnp.einsum("bnqk,bnkd->bnqd", attn, vh)
  return out.transpose(0, 2, 1, 3).reshape(b, sq, dim)


class JointSelfAttention(nnx.Module):
  """Self-attention over the joint [query|memory] sequence with 3D RoPE on q&k.
  The query token carries identity freqs (position 0) -> effectively no rotation."""
  def __init__(self, dim, num_heads, eps=1e-6, *, rngs: nnx.Rngs):
    self.num_heads = num_heads
    self.q = nnx.Linear(dim, dim, rngs=rngs)
    self.k = nnx.Linear(dim, dim, rngs=rngs)
    self.v = nnx.Linear(dim, dim, rngs=rngs)
    self.o = nnx.Linear(dim, dim, rngs=rngs)
    self.norm_q = RMSNorm(dim, eps, rngs=rngs)
    self.norm_k = RMSNorm(dim, eps, rngs=rngs)

  def __call__(self, x, freqs):
    q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
    return self.o(_mha(q, k, self.v(x), self.num_heads))


class CrossAttention(nnx.Module):
  """Cross-attention: query x attends to context y (no RoPE). Used for the
  retriever's query->encoded-memory read (and mirrors the per-block injection)."""
  def __init__(self, dim, num_heads, eps=1e-6, *, rngs: nnx.Rngs):
    self.num_heads = num_heads
    self.q = nnx.Linear(dim, dim, rngs=rngs)
    self.k = nnx.Linear(dim, dim, rngs=rngs)
    self.v = nnx.Linear(dim, dim, rngs=rngs)
    self.o = nnx.Linear(dim, dim, rngs=rngs)
    self.norm_q = RMSNorm(dim, eps, rngs=rngs)
    self.norm_k = RMSNorm(dim, eps, rngs=rngs)

  def __call__(self, x, y):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(y))
    return self.o(_mha(q, k, self.v(y), self.num_heads))


class _FFN(nnx.Module):
  """Linear -> GELU(tanh) -> Linear (4x), matching CS UnifiedJointBlock.ffn."""
  def __init__(self, dim, mult=4, *, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(dim, mult * dim, rngs=rngs)
    self.fc2 = nnx.Linear(mult * dim, dim, rngs=rngs)

  def __call__(self, x):
    return self.fc2(nnx.gelu(self.fc1(x), approximate=True))


class UnifiedJointBlock(nnx.Module):
  """Pre-Norm transformer block used for both memory-encoding and query attention:
  x += joint_self_attn(LN(x), freqs);  x += ffn(LN(x))."""
  def __init__(self, dim, num_heads, eps=1e-6, ffn_mult=4, *, rngs: nnx.Rngs):
    self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
    self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
    self.joint_self_attn = JointSelfAttention(dim, num_heads, eps, rngs=rngs)
    self.ffn = _FFN(dim, ffn_mult, rngs=rngs)

  def __call__(self, x, freqs):
    x = x + self.joint_self_attn(self.norm1(x), freqs)
    x = x + self.ffn(self.norm2(x))
    return x


class RetrievalBlock(nnx.Module):
  """One retrieval block: encode each keyframe's (pose|memory), run the query
  (pose|learnable_query) through joint attention, then cross-attend query->memory."""
  def __init__(self, dim, num_heads, eps=1e-6, mem_enc_layers=1, joint_layers=1, *, rngs: nnx.Rngs):
    self.dim = dim
    self.mem_enc_blocks = nnx.data([UnifiedJointBlock(dim, num_heads, eps, 4, rngs=rngs) for _ in range(mem_enc_layers)])
    self.joint_attn_blocks = nnx.data([UnifiedJointBlock(dim, num_heads, eps, 4, rngs=rngs) for _ in range(joint_layers)])
    self.cross_attn = CrossAttention(dim, num_heads, eps, rngs=rngs)
    self.norm_cross = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)

  def __call__(self, q_tok, pose_tokens, memory_tokens, query_freqs, memory_freqs, learnable_query):
    # q_tok:[B,1,D]  pose_tokens:[B,T,D]  memory_tokens:[B,T,3128,D]  learnable_query:[B,3128,D]
    B, T = pose_tokens.shape[:2]
    per = memory_tokens.shape[2] + 1  # 1 + 3128
    encoded = []
    for t in range(T):
      joint_t = jnp.concatenate([pose_tokens[:, t:t + 1, :], memory_tokens[:, t]], axis=1)  # [B,1+3128,D]
      freqs_t = memory_freqs[t * per:(t + 1) * per]
      for blk in self.mem_enc_blocks:
        joint_t = blk(joint_t, freqs_t)
      encoded.append(joint_t)
    encoded_memory = jnp.concatenate(encoded, axis=1)  # [B, T*(1+3128), D]

    joint_query = jnp.concatenate([q_tok, learnable_query], axis=1)  # [B, 1+3128, D]
    for blk in self.joint_attn_blocks:
      joint_query = blk(joint_query, query_freqs)
    joint_output = joint_query + self.cross_attn(self.norm_cross(joint_query), encoded_memory)
    return joint_output  # [B, 1+3128, D]; inference ignores the CE-loss encoded_pose


class MemoryRetriever(nnx.Module):
  """Pose-conditioned memory retriever (inference path).

  forward(pose_token[B,1,9], key_pose_token[B,T,9], memory_context[B, T*3128, 1024])
    -> memory_pred [B, 3128, 1024]
  """
  def __init__(self, dim=1024, num_heads=8, num_blocks=1, theta=10000.0, eps=1e-6,
               n_layers=4, grid_h=21, grid_w=37, *, rngs: nnx.Rngs):
    assert dim % num_heads == 0 and (dim // num_heads) % 2 == 0
    self.dim, self.num_heads = dim, num_heads
    self.n_layers, self.grid_h, self.grid_w = n_layers, grid_h, grid_w
    self.per_frame = n_layers * (5 + grid_h * grid_w)  # 3128

    # pose MLP (9->dim->dim) and memory MLP (1024->dim->dim)
    self.embed_0 = nnx.Linear(9, dim, rngs=rngs)
    self.embed_2 = nnx.Linear(dim, dim, rngs=rngs)
    self.memory_embed_0 = nnx.Linear(1024, dim, rngs=rngs)
    self.memory_embed_2 = nnx.Linear(dim, dim, rngs=rngs)
    self.learnable_query = nnx.Param(jnp.zeros((1, self.per_frame, dim), jnp.float32))
    self.retrieval_blocks_list = nnx.data([
        RetrievalBlock(dim, num_heads, eps, mem_enc_layers=1, joint_layers=1, rngs=rngs)
        for _ in range(num_blocks)
    ])
    self.memory_proj = nnx.Linear(dim, 1024, rngs=rngs)

    # 3D RoPE freqs are computed lazily at call time (see _cached_freqs_cis_3d): storing
    # them as plain array attributes would get them abstracted to ShapeDtypeStruct by the
    # pipeline's nnx.eval_shape. Keep only the scalar params (static -> survive eval_shape).
    self._head_dim = dim // num_heads
    self._theta = theta
    self._freq_end = 2048

  def _embed_pose(self, p):
    # CS uses EXACT gelu (nn.GELU(), erf) for embed/memory_embed — NOT the tanh approx
    # that flax's nnx.gelu defaults to (and that the retriever-block FFN does use).
    return self.embed_2(nnx.gelu(self.embed_0(p), approximate=False))

  def _embed_mem(self, m):
    return self.memory_embed_2(nnx.gelu(self.memory_embed_0(m), approximate=False))

  def __call__(self, pose_token, key_pose_token, memory_context):
    B, T = key_pose_token.shape[:2]
    memory_tokens = memory_context.reshape(B, T, self.per_frame, 1024)
    key_pose = self._embed_pose(key_pose_token)                  # [B,T,dim]
    key_mem = self._embed_mem(memory_tokens)                     # [B,T,3128,dim]
    q_tok = self._embed_pose(pose_token)                         # [B,1,dim]
    learnable_query = jnp.broadcast_to(self.learnable_query.value, (B, self.per_frame, self.dim))

    _f, _h, _w = _cached_freqs_cis_3d(self._head_dim, self._freq_end, self._theta)
    qf = jnp.asarray(build_3d_freqs_unified(1, self.n_layers, self.grid_h, self.grid_w,
                                            _f, _h, _w, True, False))
    mf = jnp.asarray(build_3d_freqs_unified(T, self.n_layers, self.grid_h, self.grid_w,
                                            _f, _h, _w, True, True))

    joint_output = None
    for blk in self.retrieval_blocks_list:
      joint_output = blk(q_tok, key_pose, key_mem, qf, mf, learnable_query)
      learnable_query = joint_output[:, 1:, :]                   # feed next block
    return self.memory_proj(joint_output[:, 1:, :])              # [B, 3128, 1024]


# ------------------------- CS checkpoint loading -------------------------
# Map our nnx param paths -> Captain-Safari torch keys (under "memory_retriever.").
_RENAME = {
    "embed_0": "embed.0", "embed_2": "embed.2",
    "memory_embed_0": "memory_embed.0", "memory_embed_2": "memory_embed.2",
}


def _nnx_path_to_cs_key(path):
  """nnx flat-state path -> (cs_key, transpose_kernel)."""
  toks = [str(getattr(p, "key", p)) for p in path]
  leaf = toks[-1]
  body = toks[:-1]
  transpose = False
  if leaf == "kernel":      # nnx Linear weight [in,out] -> CS [out,in]
    cs_leaf, transpose = "weight", True
  elif leaf == "scale":     # nnx LayerNorm gamma -> CS weight
    cs_leaf = "weight"
  else:                     # bias / RMSNorm weight / learnable_query
    cs_leaf = leaf
  out, i = [], 0
  while i < len(body):
    t = body[i]
    if t == "ffn" and i + 1 < len(body) and body[i + 1] in ("fc1", "fc2"):
      out.append("ffn.0" if body[i + 1] == "fc1" else "ffn.2"); i += 2; continue
    out.append(_RENAME.get(t, t)); i += 1
  tail = (out + [cs_leaf]) if cs_leaf != "learnable_query" else ["learnable_query"]
  return "memory_retriever." + ".".join(tail), transpose


def load_cs_memory_retriever_weights(model, safetensors_path, dtype=jnp.float32):
  """Load Captain-Safari memory_retriever.* weights into our nnx MemoryRetriever."""
  from safetensors import safe_open

  flat = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
  loaded, missing = 0, []
  with safe_open(safetensors_path, framework="flax") as f:
    keys = set(f.keys())
    for path, var in flat.items():
      cs_key, transpose = _nnx_path_to_cs_key(path)
      if cs_key not in keys:
        missing.append((tuple(str(p) for p in path), cs_key)); continue
      w = jnp.asarray(f.get_tensor(cs_key), dtype=dtype)
      if transpose:
        w = w.T
      if tuple(w.shape) != tuple(var.value.shape):
        raise ValueError(f"shape mismatch {cs_key}: {w.shape} vs {tuple(var.value.shape)} ({path})")
      var.value = w
      loaded += 1
  if missing:
    raise KeyError(f"{len(missing)} params had no CS key, e.g. {missing[:3]}")
  nnx.update(model, nnx.from_flat_state(flat))
  return loaded
