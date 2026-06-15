# Phase H (Captain-Safari 3D memory) — Status & Handoff

Branch: `wan22-memory` (off `wan22-dense`). Goal: port Captain-Safari (CS) memory-conditioned
**Wan2.2-Fun-5B-Control-Camera** to JAX/TPU (maxdiffusion) and match CS's GPU `validate_lora` output.

## TL;DR current state
- The full memory pipeline **runs end-to-end on TPU** and produces video (gs://data_us_central1_a/maxdiffusion/wan/cs_memory/gen/).
- **RESOLVED (bfece212):** wrong output was the launcher loading memory but NOT the CS LoRA (wan_lora_path unset). With wan_lora_path=epoch-4 the full model (base+LoRA+memory) generates correctly, matching CS. See ROOT CAUSE section.
- Generation now matches CS. ALWAYS load both LoRA (wan_lora_path) and memory (wan_memory_path) from epoch-4.

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

## ROOT CAUSE of the wrong/noisy generation (FIXED)
`epoch-4.safetensors` = base PAI **+ LoRA r32 + memory**. The generate launcher set
`wan_memory_path=epoch-4` (loads memory) and `lora_rank=32` (builds zero-init adapters) but
**did NOT set `wan_lora_path`** → `load_wan_lora` never ran → the **CS LoRA was never loaded**.
Every prior run was `base + zero-LoRA + memory` (the LoRA half of the checkpoint dropped),
which is why no memory-side change affected the output. Fixed by adding
`wan_lora_path=epoch-4.safetensors` to `examples/generate_cs_memory_demo.sh`. Confirm both
`WAN LoRA: mapped …` and `WAN memory: filled …` appear in the gen log.

REMINDER: load EVERY param from CS's checkpoint — LoRA (load_wan_lora) AND memory
(load_wan_memory), both from epoch-4. Re-verify in bf16+flash (the real config).

## Parity tooling (facts; both quick parities have input confounds — their NaNs are harness artifacts, not model bugs)
- `src/eval/cs_full_forward_dump_torch.py` (run on GPU/CS env) — dumps model_fn inputs + noise_pred + per-block outputs (`block_00..29`) + block input. `PAI_DIR` for no-download. CS dumps at `gs://data_us_central1_a/maxdiffusion/wan/cs_memory/parity/`.
- `src/eval/tpu_noise_pred_parity.py` — NaNs because it feeds CS's FlowMatch-scaled latent into our model.
- `src/eval/tpu_block_parity.py` — NaNs because it hand-feeds temb/rotary (block 0 NaNs even without memory).
- `src/eval/vae_roundtrip_tail.py` — VAE encode→decode round-trip tail check.

## Key facts
- CS checkpoint `epoch-4.safetensors` = base PAI + LoRA r32 (alpha=1) + memory. See memory note `cs-checkpoint-structure`.
- Memory layout: `(K,4,782,1024)` = K keyframes × 4 VGGT layers × 782 tokens × 1024. Retriever T = #keyframes (16 in demo). per_frame=3128=4×782.
- Run gen: `examples/generate_cs_memory_demo.sh` (RUN_TAG/HEIGHT/WIDTH/NUM_FRAMES/STEPS/PER_DEVICE_BS env). Assets at gs://…/cs_memory/.
