# Phase H (Captain-Safari 3D memory) — Status & Handoff

Branch: `wan22-memory` (off `wan22-dense`). Goal: port Captain-Safari (CS) memory-conditioned
**Wan2.2-Fun-5B-Control-Camera** to JAX/TPU (maxdiffusion) and match CS's GPU `validate_lora` output.

## TL;DR current state
- The full memory pipeline **runs end-to-end on TPU** and produces video (gs://data_us_central1_a/maxdiffusion/wan/cs_memory/gen/).
- **Open bug:** our 704×1280×121 generation has **content but a noisy TAIL** (last few frames) vs CS's clean output. Does NOT change with any memory fix.
- We are mid-investigation with a **stage-by-stage numerical parity** vs a CS GPU dump.

## What is DONE + VERIFIED CORRECT (committed)
- `models/wan/memory_retriever.py` — nnx port of CS MemoryRetriever. Retriever parity vs CS = **rel 9e-5**.
  - FIX `de9c2e77`: embed/memory_embed use **exact** gelu (`approximate=False`), matching CS `nn.GELU()` (NOT tanh).
  - 3D RoPE freqs computed lazily via `functools.lru_cache` (not stored attrs) — else `eval_shape` abstracts them to ShapeDtypeStruct.
- `models/wan/wan_utils.py::load_wan_memory` — loads CS memory weights (memory_emb + per-block memory_cross_attn/norm_memory scan-stacked + retriever). 364/364 keys validated. Int block-indices preserved (str/int sort bug fixed `c8c8c55`).
- `models/wan/memory_pose.py` — numpy pose encoding (convert_to_local_coordinates + extri_intri_to_pose_encoding + PyTorch3D mat_to_quat). vs CS = **rel ~1e-7**.
- `models/wan/transformers/transformer_wan.py` — memory wiring: per-block `memory_cross_attn`(=CrossAttention) + `norm_memory`(LayerNorm), injected `x + memory_cross_attn(norm_memory(x), memory_context)` after text cross-attn, before ffn (matches CS DiTBlock:283, ungated). `compute_memory_context()` runs retriever+memory_emb once. mem_out cast to hidden dtype (scan-carry dtype fix).
- `pipelines/wan/wan_pipeline_2_2_fun_camera.py` + `wan_pipeline.py` — thread memory through generation (computed once, doubled for CFG).
- `generate_wan_2_2_fun_camera.py` — `build_memory_inputs_from_config`; **`memory_n_key=0` = ALL keyframes** (FIX `cf31e9b6`: was hardcoded 4; data has 16, CS uses all → T=16).
- VAE encode batch→redundant-axis padding fix (`256bc7c0`) — needed for batch=1 full-res.
- Config: `use_memory`, `memory_dim/heads/blocks`, `wan_memory_path`, `memory_n_key`, `memory_path`, pose paths.

## VERIFIED to MATCH CS (so NOT the bug)
- Memory module (retriever + emb + injection): rel 9e-5; gelu/keyframe fixes applied.
- Conditioning mechanism: Fun-5B config `in_dim=100` (concat y), single timestep, no fuse → matches our `_append_y` + scalar timestep. (CS `wan_video_dit.py:1264-1280`.)
- Camera control: interpolated extrinsics bit-exact vs CS (**1.4e-6**), valid rotations; the ~2% control diff is bf16 rounding in CS's dump.

## HBM / run constraints
- v6e-8 = 8 chips. Full-res 704×1280×121 **OOMs at batch 2**; needs `per_device_batch_size=0.125` (batch 1). `vae_spatial=4` required. CS recipe: 50 steps, flow_shift=5.0 (=CS sigma_shift), CFG 5.0, long Chinese negative.

## OPEN BUG investigation (noisy tail)
Tooling (all on `wan22-memory`, run on tpu-v6e-8-debug; CS dumps at `gs://data_us_central1_a/maxdiffusion/wan/cs_memory/parity/`):
- `src/eval/cs_full_forward_dump_torch.py` — run on **GPU** (CS env). Dumps model_fn inputs + noise_pred + **per-DiT-block outputs** (`block_00..29`) + block-stack input (`blkin_x/context/t_mod/memory_context`). Uses `PAI_DIR` for no-download.
- `src/eval/tpu_noise_pred_parity.py` (+ `examples/tpu_noise_pred_parity.sh`) — feed CS inputs to our DiT, diff noise_pred.
- `src/eval/tpu_block_parity.py` (+ `examples/tpu_block_parity.sh`) — feed CS's block input through our blocks one at a time (scan_layers=False, float32), flag first divergence.

### Findings so far
- noise_pred parity (bf16): NaN — but a **harness artifact** (feeding CS's FlowMatch-scheduled latent to our bf16 model overflows). Real generation is NOT NaN (has content).
- **fp32+flash parity:** base forward (no memory) **FINITE** (std 1.94); **WITH memory → NaN even in fp32**. So NaN enters the **memory-injection path**, structurally (not bf16).
- `memory_context` itself is **finite/healthy** (std 0.42) → NOT a memory-data problem.
- **IN PROGRESS:** block-by-block parity to localize whether a specific block's `memory_cross_attn` diverges from CS (code bug) vs values just accumulating across 30 blocks (parity scale artifact).

### Leading hypotheses for codex to pursue
1. Numerical instability in the memory injection (memory_cross_attn `_mha` scale / RMSNorm eps / softmax) at the full 704×1280 sequence — would explain tail noise (late frames blow up) + parity NaN.
2. Sampler: ours = `FlaxUniPCMultistepScheduler`, CS = diffsynth FlowMatch-Euler (same shift). Not yet excluded for the tail.
3. VAE temporal decode tail (CS uses `tiled=True`; check our temporal tiling/causal boundary on the last latent frame).

## Key facts
- CS checkpoint `epoch-4.safetensors` = base PAI + LoRA r32 (alpha=1) + memory. See memory note `cs-checkpoint-structure`.
- Memory layout: `(K,4,782,1024)` = K keyframes × 4 VGGT layers × 782 tokens × 1024. Retriever T = #keyframes (16 in demo). per_frame=3128=4×782.
- Run gen: `examples/generate_cs_memory_demo.sh` (RUN_TAG/HEIGHT/WIDTH/NUM_FRAMES/STEPS/PER_DEVICE_BS env). Assets at gs://…/cs_memory/.

## UPDATE — parity harness caveats (important for codex)
Both quick parities have **confounds** — do not trust their NaNs as real model bugs:
- `tpu_noise_pred_parity.py`: feeds CS's step-0 latent (from CS's FlowMatch scheduler) → out of our model's expected scale → bf16 overflow → NaN. (Real gen uses our scheduler's latent, no NaN.)
- `tpu_block_parity.py`: hand-feeds CS's `blkin_t_mod` + a recomputed `rotary` to each block → **block 0 NaNs even WITHOUT memory**, i.e. the BASE path NaNs → the temb/rotary I feed is wrong. So its per-block NaNs are a harness artifact, NOT a memory bug.

**Reliable facts:** real bf16+flash+scan generation = **content + noisy TAIL** (finite, not NaN). Memory/conditioning/camera all match CS numerically; `memory_context` finite.

**Clean next step (TODO):** capture per-block outputs DURING the real full forward (model computes temb/rotary/context itself) by instrumenting `WanModel._run_all_blocks` scan to emit per-layer `ys`; feed CS's latents/y/context; diff each layer's output vs CS `block_NN`. That isolates the first real divergence without the temb/rotary or latent-scale confounds. Then chase the noisy-tail (sampler UniPC-vs-FlowMatch, or VAE temporal-decode tail) on the finite output.
