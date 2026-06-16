# maxdiffusion — Project State & Onboarding (for an AI collaborator)

You are joining an in-progress effort to build **camera-controlled WAN 2.2 video
diffusion training** on TPU, in a fork of Google's `maxdiffusion`. This doc is
the big picture: goal, infra, what's done, what's next, the code map, and the
hard-won gotchas. Read it fully before touching anything.

Owner: Shu (ML researcher). Working dir: `/home/spu9/maxdiffusion`. The owner
also keeps a read-only study worktree at `/home/spu9/maxdiffusion-learn`
(branch `wan22-dense-learn`) — **never run git mutations there**.

---

## 0. LATEST — Phase H (Captain-Safari 3D memory) is DONE; sanity-check guide

> NOTE (sections 1–11 below predate Phase H and call memory "out of scope" — that is
> now superseded). Phase H ported Captain-Safari's (CS) memory-conditioned
> **Wan2.2-Fun-5B-Control-Camera** to JAX/TPU. It generates end-to-end and matches CS.
> Full status + the debugging trail is in **`PHASE_H_STATUS.md`** (read it for context).

**Branch:** `wan22-memory` (off `wan22-dense`). **Full CS model = base PAI + LoRA r32 + memory.**

**★ #1 gotcha (cost a full day):** `epoch-4.safetensors` holds BOTH a LoRA AND the memory
modules. They load via SEPARATE knobs — `wan_lora_path` (→`load_wan_lora`) and
`wan_memory_path` (→`load_wan_memory`). `lora_rank=32` alone only builds zero-init (no-op)
adapters. You MUST set **both `wan_lora_path=epoch-4` AND `wan_memory_path=epoch-4`**. Confirm
BOTH `WAN LoRA: mapped 600 tensors` AND `WAN memory: filled …` appear in the gen log, else
you're running base+zero-LoRA+memory (wrong output, and no memory change will fix it).

**Phase H code map (all on `wan22-memory`):**
- `src/maxdiffusion/models/wan/memory_retriever.py` — nnx port of CS MemoryRetriever (+ the per-block `CrossAttention`/`RMSNorm`; `_mha` now uses `jax.nn.dot_product_attention`). `load_cs_memory_retriever_weights`, `_nnx_path_to_cs_key`.
- `src/maxdiffusion/models/wan/wan_utils.py` — `load_wan_memory` (memory weights), `load_wan_lora`/`init_wan_lora_params` (LoRA). Both preserve int block-indices (non-scan).
- `src/maxdiffusion/models/wan/memory_pose.py` — host-side numpy pose encoding (local-coords + `extri_intri_to_pose_encoding` + PyTorch3D `mat_to_quat`).
- `src/maxdiffusion/models/wan/transformers/transformer_wan.py` — memory wiring: per-block `memory_cross_attn`+`norm_memory` injection, `compute_memory_context()`, `use_memory` gate.
- `src/maxdiffusion/pipelines/wan/wan_pipeline_2_2_fun_camera.py` + `wan_pipeline.py` — thread `memory_context` through generation (computed once, doubled for CFG); `load_wan_memory` overlay; VAE-encode redundant-axis padding.
- `src/maxdiffusion/generate_wan_2_2_fun_camera.py` — `build_memory_inputs_from_config` (`memory_n_key=0`=ALL keyframes). Config: `src/maxdiffusion/configs/base_wan_2_2_fun_5b_camera.yml` (use_memory/memory_*/wan_memory_path).
- Launchers (untracked-ish, run locally): `examples/generate_cs_memory_demo.sh` (the gen), `examples/tpu_*_parity.sh`, `examples/vae_roundtrip_tail.sh`.
- Parity/eval tools: `src/eval/cs_full_forward_dump_torch.py` (run on GPU/CS env → dumps), `tpu_noise_pred_parity.py`, `tpu_block_parity.py`, `vae_roundtrip_tail.py`, `cs_memory_*.py`. **NOTE: both quick parities have input confounds — their NaNs are harness artifacts, not model bugs (see PHASE_H_STATUS.md).**

**Infra for Phase H:**
- TPU: `tpu-v6e-8-debug` (project `priors-medical-ai`, zone `us-central1-a`), single-host worker 0. v6e-8 = 8 chips. Full-res 704×1280×121 needs `per_device_batch_size=0.125` (batch 1; batch 2 OOMs) and `vae_spatial=4`.
- GCS: `gs://data_us_central1_a/maxdiffusion/wan/cs_memory/` — `epoch-4.safetensors` + demo assets (memory.npy, camera matrices, input_frame0.png); `…/gen/` outputs; `…/parity/` CS dumps.
- CS reference (GPU/torch): conda env `/data1/spu9/misc/miniconda3/envs/captain_safari` (or `/data2/spu9/envs/captain_safari`); CS repo `/home/spu9/Captain-Safari/captain_safari`; PAI base local at `/data2/spu9/models/PAI/Wan2.2-Fun-5B-Control-Camera` (use `PAI_DIR` to skip re-download).

**Codebase sanity-check checklist (for codex):**
1. `git checkout wan22-memory && git pull`; `python -m py_compile` the Phase H files above (all should compile).
2. Confirm the gen launcher sets BOTH `wan_lora_path` and `wan_memory_path` (grep `examples/generate_cs_memory_demo.sh`).
3. Memory-load key coverage: `load_wan_memory` maps 364 new keys + retriever (validated). Re-verify against `epoch-4` if touched.
4. Pose-encoding parity: `memory_pose.build_memory_pose_tokens` vs CS (`cs_memory_full_ref_torch.py`) → rel ~1e-7.
5. Retriever parity: `cs_memory_parity.py` vs CS dump → rel ~9e-5.
6. Smoke generation (480×832×49, fast): `RUN_TAG=sanity HEIGHT=480 WIDTH=832 NUM_FRAMES=49 STEPS=20 bash examples/generate_cs_memory_demo.sh` → expect `WAN LoRA: mapped` + `WAN memory: filled` + a video in `…/cs_memory/gen/`.
7. Reference good output: `gs://…/cs_memory/gen/cs-mem-flashattn.mp4` (full CS model, 704×1280×121) should look like CS's `validate_lora` output.

**Next direction:** Phase AR (autoregressive Fun-camera via minWM + causal-forcing) — see Shu (plan pending post-meeting). Memory module is the substrate for AR rollout.

---

## 1. The goal

Build, on top of upstream maxdiffusion, a working **WAN 2.2 Fun-5B
Camera-Control** training stack (LoRA + full fine-tune) on the owner's
HM-World dataset, runnable on TPU v6e-256 pods. This is the owner's research
baseline — they will build their own project on it. We are NOT replicating
Captain-Safari's full pipeline (its memory-retrieval augmentation is out of
scope); we use Fun-5B's image + camera-trajectory conditioning only.

Method that has worked the whole way: **anchor every step to a known-good
reference and gate on a concrete check before moving on.** Upstream WAN 2.1
training is the numerics reference; DiffSynth/Captain-Safari are the
architecture/recipe reference; official PAI weights certify inference.

---

## 2. Repo / branch structure

- **`wan22-dense`** — the production branch. Everything verified lives here.
  Built on upstream main `4a3ec4f3` + our commits. This is what TPU workers run.
- **`wan22-dense-learn`** — owner's study worktree. Do not touch.
- Fork remotes available locally as branches (from the owner's earlier work):
  - `fork-training` (= `wan22-training`): WAN 2.2 TI2V-5B + frame-concat + LoRA + VACE
  - `fork-camera` (= `wan22-fun-camera-control`): adds Fun camera-control
  - These are the SOURCE we port from; they hit NaN in training and had drifted
    ~6 PRs behind upstream. We rebuild cleanly onto upstream instead of using
    them directly.
- Origin = `https://github.com/URRealHero/maxdiffusion.git` (the owner's fork).
  Upstream = `AI-Hypercomputer/maxdiffusion`.

Git rule: **always verify `git branch --show-current` == `wan22-dense` before any
git mutation in `/home/spu9/maxdiffusion`.** A past incident checked out the
wrong branch silently.

---

## 3. Infrastructure

- **TPUs** (project `priors-medical-ai`, zone `us-central1-a`): pods
  `tpu-v6e-256-3` (our main), `tpu-v6e-256-2`, `tpu-v6e-256-0` (shared — only
  `-3` and `-2` are authorized for our use). `tpu-v6e-8-debug` was a single-host
  slice.
- A v6e-256 pod = **64 worker VMs × 4 chips = 256 chips**. `--worker=all` fans a
  command to all 64. Multihost SPMD: one train step spans all 256 chips.
- **gcloud**: binary at `/home/spu9/google-cloud-sdk/bin/gcloud` (NOT on bash
  PATH). Always `export CLOUDSDK_CONFIG=$HOME/.config/gcloud`. Account
  `spu9@ucsc.edu`.
- **Shared model cache (Hyperdisk ML)**: 250GB `wan-model-cache`, READ_ONLY_MANY,
  mounted on all 64 workers of `-3` at `/home/spu9/maxdiffusion_storage`;
  `HF_HUB_CACHE` points there via `~/maxdiffusion_env.sh`. Holds PAI Fun-Camera,
  TI2V-5B, 2.1-1.3B, 2.1-14B, Captain-Safari LoRA. It is **immutable** (one-way
  RO flip). Ops scripts + the add-model workflow: `~/TPU/hyperdisk/` (read its
  README). Per-worker boot disks are 97G and stateless: code from git, models
  from the RO disk, data from GCS, outputs to GCS.
- **Data** (GCS bucket `data_us_central1_a`):
  - `hmworld_data/fun_camera_encoded_full/` — **the current training dataset**:
    55,961 records, 480×832×153, WAN 2.2 VAE (48ch). Features per record:
    `latents [48,39,30,52]`, `latent_condition [48,39,30,52]` (VAE of
    first-frame+zeros, for Fun y-conditioning), `encoder_hidden_states
    [512,4096]`, `camera_extrinsic [153,3,4]`, `camera_intrinsic [153,3,3]`.
    Verified: 0 dups, all finite, correct shapes.
  - `wan_tfr_dataset_pusa_v1/`, `tv2v_ti2v5b_encoded_full/` — older datasets.

---

## 4. Verified architecture facts (do not re-derive — these are load-bearing)

- **Mesh**: `mesh_axes = [data, fsdp, context, tensor]`, product = 256.
  - `context × tensor` must DIVIDE the attention head count (1.3B=12, TI2V-5B=24,
    14B=40) — flash attention shards heads over (context,tensor).
  - `data × fsdp` must divide the global train batch (= per_device_bs × 256).
  - 5B + Adam needs fsdp≥4 to fit 31GB HBM → verified 5B mesh: `d16/f4/c4/t1`.
  - 1.3B verified mesh at bs=0.25: `d64/f1/c4/t1`.
- **Remat** (`remat_policy`): use `FULL`. `NONE` OOMs (video seqs need ~150G HBM).
  `HIDDEN_STATE_WITH_OFFLOAD` is **NOT SAFE on v6e** — silent NaN / loss blowup
  even with the README XLA flags (offload host-transfer races). Documented in
  `examples/TRAINING_CONFIG.md`. Shu has created a branch `debug/multihost-visibility` and add a worktree in `~/maxdiffusion-debug`. We are in main developing branch `wan22-dense`, do not need to care about the debug branch.
- **The two VAEs**: 2.1 = `autoencoder_kl_wan.py` (16ch, 8× spatial); 2.2 =
  `autoencoder_kl_wan_2p2.py` (48ch, **16× spatial** via a 2× pixel-shuffle
  patchify on top of conv downsamples). The 2.2 `vae_scale_factor_spatial` MUST
  be 16, not 8 (a fixed bug — see `WanPipeline2_2_Dense.__init__`). Latent
  normalization constants (`latents_mean/std`, 48 values) match DiffSynth's
  `WanVideoVAE38` exactly (verified).
- **VAE sharding**: `vae_spatial` must divide the latent width AND the per-host
  batch must be divisible by `device_count / vae_spatial`. Encoder used
  `vae_spatial=1/2` with batch 4/2 on 4-chip hosts; training/inference uses
  `vae_spatial=4`.
- **dtypes**: bf16 weights+activations, fp32 VAE. The owner's decision: do NOT
  add dtype changes; match the verified WAN 2.1 numerics. (A fp32-loss island
  was considered and dropped — 2.1 trains NaN-free with bf16 loss.)
- **Loss**: flow-match MSE. The training-weight normalization is over the FIXED
  1000-step grid, NOT the sampled batch (a fixed bug — batch-normalization made
  loss scale ∝ 1/batch_size; see `scheduling_flow_match_flax.py
  _calculate_training_weights`). After the fix, loss is comparable across batch
  sizes (~0.15–0.2 scale for the fun-camera model).
- **Multihost logging**: jax process 0 ≠ GCE worker 0 (topology-assigned).
  Every worker logs `[proc N] completed step ...` (our debug patch) so a silent
  worker-0 log can't masquerade as a hang.

---

## 5. Phase history (what's done, what each taught us)

The rebuild ladder, all on `wan22-dense`:

- **Stage 0** — verified upstream WAN 2.1 trains NaN-free on Pusa + HM-World data
  (886 steps, loss 1.44→1.19). Infra fixes: DEBUG env injection, per-host step
  logging, `.tfrec`-only dataset glob, **file-level dataset sharding** (record-
  level sharding made every host stream the whole dataset → ~hour-long first
  batch on 1.4TB).
- **R1–R4** — ported & reconciled WAN 2.2 TI2V-5B dense from the fork:
  config-key drift, `load_vae` override for the 2p2 VAE, training whitelist,
  the 16× scale-factor fix. VAE certified (re-encode corr 0.999999 vs dataset).
  5B inference smoke OK.
- **R5** — 2.2 dense training (no camera): 2000 steps, stable, no NaN.
- **E1** — Fun-5B camera-control inference ported (`WanCameraSimpleAdapter`,
  Plücker, `control_camera_latents_input`, PAI `.pth` loaders). Certified vs
  Captain-Safari reference videos in BOTH camera modes (preset Left/0.01 +
  explicit extrinsic/intrinsic). Commit `2a9cf293`.
- **E2** — dataset re-encoded with `latent_condition` (the Fun y-conditioning =
  VAE of [first_frame, zeros×152]). 55,961 records verified.
- **E3** — built the camera trainer: `models/wan/camera_plucker.py` (JAX Plücker,
  on-device, parity-verified vs the NumPy inference path to 2e-6),
  `trainers/wan_2_2_fun_camera_trainer.py` (100ch input = noisy48|mask4|cond48,
  per-token timesteps with frame-0 clean, loss on frames 1+,
  `adapter_grad_norm` metric), `train_wan_2_2_fun_camera.py`,
  `examples/train_fun_camera_5b.sh`.
- **E4** — smoke: 2000 steps, loss 0.209→0.156, zero NaN, 1.84 s/step,
  adapter_grad_norm ≈ total grad norm (camera path carries the learning).
  Causality probe on the PRETRAINED model: margin −2.2% (no clear effect yet —
  expected, OOD data + pretrained not-yet-adapted; rerun after a long train).
  Probe: `src/maxdiffusion/eval_fun_camera_probe.py`.

**Phase E is CLOSED.** Camera-conditioned 2.2 training works end-to-end.

---

## 6. CURRENT STATE — Phase F1 (LoRA), uncommitted, tests pass

Just implemented, sitting UNCOMMITTED in `/home/spu9/maxdiffusion` on
`wan22-dense` (owner will review + commit):

- `models/wan/wan_lora.py` (new) — `WanLoRAAdapter`: `B(A(x))·(alpha/rank)`,
  A=kaiming-uniform / B=0 init (delta starts at 0). Exact DiffSynth/CS recipe.
- LoRA threaded through `transformer_wan.py` (ApproximateGELU=ffn.0,
  WanFeedForward=ffn.2, WanTransformerBlock, WanModel) and `attention_flax.py`
  (q/k/v/o in FlaxWanAttention, both self- and cross-attn).
- **Configurable targets**: `lora_target_modules` (default
  `"q,k,v,o,ffn.0,ffn.2"` = full CS set). Per-module gating; ungated modules get
  no adapter. `lora_rank: 0` → bit-identical to base model.
  Limitation: under `scan_layers=True` (5B uses it), per-BLOCK targeting isn't
  possible (all layers share one block def); per-MODULE works fully.
- Config keys wired in `pipelines/wan/wan_pipeline.py` +
  `configs/base_wan_2_2_fun_5b_camera.yml`.
- `tests/wan_lora_test.py` (new) — **9/9 pass** on CPU (worker 1). Tests:
  adapter zero-output & scale, param names/count (20 leaves/block),
  no-params-when-rank-0, FFN & block zero-delta (rewritten to toggle LoRA on ONE
  module — the fork's two-module comparison diverged on base weights due to extra
  LoRA rng draws; also asserts nonzero-B DOES change output).

Run the tests:
```
cd ~/maxdiffusion && JAX_PLATFORMS=cpu python -m pytest src/maxdiffusion/tests/wan_lora_test.py -q
```

---

## 7. The plan ahead (Phase F)

- **F2 — load CS LoRA weights + mechanical smoke.** Write the PEFT→nnx LoRA
  key remap in `models/wan/wan_utils.py` (CS keys look like
  `blocks.N.self_attn.q.lora_A.default.weight [32,3072]`; ignore the
  memory-retrieval keys — `memory_*`, `norm_memory`, `cross_attn`). Load the
  600 LoRA tensors from
  `Captain-Safari/.../Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors`
  (staged on hyperdisk) onto our Fun-Camera model. NOTE: CS's LoRA was
  co-trained WITH its memory modules, which we don't have, so this is a
  **mechanical load test** (weights apply, output changes), NOT inference parity.
- **F3 — LoRA-only training.** optax-masked optimizer (freeze base, train only
  params whose path contains `lora_`; config key
  `lora_trainable_param_substrings: "lora_"` already exists). LoRA checkpointer
  (saves only LoRA params). Gate: short run, loss drops with base frozen, only
  LoRA params get gradients, save/load roundtrip.
- **F4 — the real run.** LoRA fine-tune (and/or full FT) on the owner's dataset,
  the project baseline.

Also pending (not blocking): long fun-camera training run (set
`save_final_checkpoint=True`, larger `MAX_STEPS`, review lr schedule — note the
warmup is `0.1 × MAX_STEPS`, so a 100k-step run warms up for 10k steps); rerun
the causality probe against a trained checkpoint with distant-record shuffles;
delete the superseded `concat_camera_encoded_full` dataset.

---

## 8. Code map (where things live)

- Entry/config: `pyconfig.py`, `train_wan_2_2_fun_camera.py`,
  `train_wan_2_2_dense.py`, `configs/base_wan_2_2_fun_5b_camera.yml`,
  `configs/base_wan_ti2v_5b.yml`.
- Model: `models/wan/transformers/transformer_wan.py` (WanModel, blocks, camera
  adapter, LoRA threading), `models/attention_flax.py` (FlaxWanAttention, flash
  kernels, LoRA on q/k/v/o), `models/wan/wan_lora.py`,
  `models/wan/camera_plucker.py`, `models/wan/autoencoder_kl_wan_2p2.py`,
  `models/gradient_checkpoint.py` (remat policies).
- Pipelines: `pipelines/wan/wan_pipeline.py` (shared loaders + wan_config
  assembly — LoRA/camera/channel overrides are here),
  `wan_pipeline_2_2_dense.py`, `wan_pipeline_2_2_fun_camera.py`.
- Weight loading + key remaps: `models/wan/wan_utils.py` (PyTorch→nnx,
  `is_wan_2p2`, PAI/DiffSynth aliases).
- Data: `input_pipeline/_tfds_data_processing.py` (file-level sharding),
  `multihost_dataloading.py`, `tpu_tfrecord_encoder/encode_concat_camera.py`
  (the encoder that built the dataset; writes `latent_condition`).
- Training loop: `trainers/base_wan_trainer.py` (start_training/training_loop,
  non-finite hard-stop), `trainers/wan_trainer.py` (2.1/2.2-dense step),
  `trainers/wan_2_2_fun_camera_trainer.py` (camera step),
  `schedulers/scheduling_flow_match_flax.py`.
- Launchers: `examples/train_fun_camera_5b.sh`, `examples/train_pusa.sh`,
  `examples/infer_fun_camera.sh`, `examples/TRAINING_CONFIG.md` (the config
  reference — read it).

---

## 9. Hard-won gotchas (every one of these cost real debugging time)

1. **Disk-full on 97G boot disks** (bit us 3×): HF caches fill the disk → cryptic
   OOM/hang at model load. Now solved by the RO hyperdisk; if a worker downloads
   anyway, `df -h /` and clear `~/maxdiffusion_cache`.
2. **`pkill -f X` over ssh kills its own ssh session** if X appears anywhere in
   the remote command line (incl. a filename in a later `nohup python .../X.py`).
   Use a bracket pattern: `pkill -f "[t]rain_wan"`. Separate kill and launch into
   different ssh calls.
3. **SIGTERM wedges JAX mid-collective** (log freezes, proc survives in
   futex_wait). Use `pkill -9` and re-verify with a fresh sweep.
4. **Single-process job on a pod worker hangs at TPU init** (waits for all 64
   hosts). Prefix with `TPU_PROCESS_BOUNDS=1,1,1
   TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` to claim that host's 4 chips standalone.
   Also `MAXDIFFUSION_FORCE_LOCAL_DEVICE_MESH=1` so the pipeline builds a
   per-host mesh (every process sees all 256 devices even without
   jax.distributed; without this, per-host encoder/probe jobs build pod-wide
   256-chip meshes → divisibility errors).
5. **Editable install**: the venv must `pip install -e ~/maxdiffusion`; the setup
   script's bootstrap does a NON-editable install, so a fresh venv silently runs
   a code SNAPSHOT until something runs `pip install -e`. Training launchers do
   it; one-off tools don't. Check:
   `python -c "import maxdiffusion,os; print(os.path.dirname(maxdiffusion.__file__))"`
   must be under `/home/spu9/maxdiffusion/src/`.
6. **Stale `/tmp/libtpu_lockfile`** after a crash blocks the next TPU process:
   `sudo rm -f /tmp/libtpu_lockfile` before relaunch.
7. **Only ONE launcher at a time** on a pod — two processes (e.g. the owner's
   tmux + an agent's detached run) fight over the TPU lock and poison each other.
   Announce launches; if the owner runs manually, the agent stands down.
8. **RO hyperdisk** triggers benign `huggingface_hub` "Could not cache
   non-existence ... Read-only file system" warnings (it can't write `.no_exist`
   markers). Harmless; silence with `HF_HUB_OFFLINE=1`.

---

## 10. Working conventions

- **Pre-launch ritual**: sweep all workers for stale procs (`pgrep -fc
  "[t]rain_wan"` must be 0), check `df -h /`, confirm workers on the right commit.
- **Detached runs**: `setsid nohup python ... > ~/log 2>&1 < /dev/null &` so the
  job survives ssh timeouts and both owner + agent can tail the log. (An agent's
  local ssh wrapper often exits 143 right after launching a detached job — that's
  the wrapper timing out, not the job dying.)
- **Sync code to workers** (committed): `git fetch <fork-url> wan22-dense &&
  git reset --hard FETCH_HEAD` via `--worker=all`. (Uncommitted: scp a `git diff`
  patch + `git apply`.) `-3`'s git origin currently points at UPSTREAM, so fetch
  the branch by the fork URL explicitly.
- **Gates over vibes**: every phase ends with a concrete pass/fail check
  (numerical parity, a decode video, a NaN-free step count, a test). Verify on
  the cheapest hardware that's valid (CPU for shape/parity tests, one worker for
  single-host, full pod only for real training).
- **Each bug fix is one reviewable commit** on `wan22-dense` with a message
  explaining the why. The owner reviews and commits the bigger feature work
  themselves.

---

## 11. Immediate next action

F1 is implemented and tested (9/9), uncommitted, awaiting the owner's review +
commit. After that: F2 (LoRA weight remap + load CS weights). Do not start a TPU
run without coordinating — the owner may be using the pod.
