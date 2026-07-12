#!/usr/bin/env bash
# V2V-baseline (HyDRA reproduction): fine-tune Wan2.1-T2V-1.3B with the per-block
# camera-injection scaffold on frame-concat (cond|tgt) latents. Trains ONLY the
# camera/projector/self-attn params (v2v_trainable_param_substrings). Entry =
# train_wan_v2v.py / WanV2VConcatTrainer; config = base_wan_2_1_v2v.yml.
#
# Reads the re-encoded WAN2.1 tv2v tfrecs (WITH camera):
#   latents/cond_latents [16,20,60,104], cam_emb_con/tgt [20,12], eh_states [512,4096].
#
# Mesh (v6e-128 = 128 chips): CONTEXT PARALLELISM PRODUCES NaN in this v2v model
# (validated on tpu-v6e-8-debug 2026-06-30: fsdp2/ctx4 -> NaN loss+grads at step 0/1;
# fsdp8/ctx1 -> finite loss 0.095/0.118, checkpoint OK). The 'context' axis shards
# BOTH the self-attn heads (['context','tensor']) and the f*h*w=62400 sequence, and
# one of those sharded paths goes non-finite for the per-block camera injection. So
# DO NOT use context (or tensor) parallelism here: keep ici_context=ici_tensor=1 and
# fill the slice with FSDP. Default fsdp=128 (data=1) => batch bs*128 shards over 128.
# Each device then holds 1 sample x full 62400 seq (validated to fit with remat=FULL,
# ~35 s/step, ~22 TFLOP/s/device -- functional but low util; a context-parallel speedup
# would need the v2v NaN fixed first). PER_DEVICE_BS must be >=1 (global batch = 128*bs
# must be divisible by data*fsdp=128). REMAT: NEVER HIDDEN_STATE_WITH_OFFLOAD (NaNs on
# v6e); FULL fits the 1.3B with minimum HBM (NONE OOMs: 128G vs 31G).
#
# Usage:
#   ./train_wan_v2v.sh                                  # v6e-128, fsdp128/ctx1, remat FULL
#   MAX_STEPS=3 PER_DEVICE_BS=1 CHECKPOINT_EVERY=2 RUN_NAME=v2v-smoke \
#     ICI_FSDP=8 ICI_CONTEXT=1 TPU_NAME=tpu-v6e-8-debug ./train_wan_v2v.sh   # 8-chip smoke (validated)
# Any UPPER_CASE var below can be overridden from the environment.
set -euo pipefail

# ---------------- Overridable config (defaults = standard v6e-128 run) --------
TPU_NAME="${TPU_NAME:-tpu-v6e-128-1}"          # the re-encode slice; frees when done
PROJECT="${PROJECT:-priors-medical-ai}"
ZONE="${ZONE:-us-central1-a}"

BRANCH="${BRANCH:-v2v-baseline}"

LR="${LR:-1e-5}"
# Opt-in official HyDRA recipe parity knobs. Historical runs used False/flax_lecun_normal.
HYDRA_OFFICIAL_SCHEDULER="${HYDRA_OFFICIAL_SCHEDULER:-True}"
HYDRA_TOKENIZER_INIT="${HYDRA_TOKENIZER_INIT:-torch_conv3d}"
# HyDRA schedule (paper 5.1): 10K iterations, global batch 32, lr=1e-5 AdamW constant.
# NOTE on batch: exact global-batch-32 on 128 chips needs tensor/context parallelism
# (ctx hangs; tensor untested). Fallback = global batch 128 (fsdp=128) x 2500 steps
# = same 320K samples seen. Validate the batch-32 mesh on the slice before committing.
MAX_STEPS="${MAX_STEPS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
RUN_NAME="${RUN_NAME:-v2v-baseline-hydra-1.3b-lr-${LR}}"

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-gs://data_us_central1_a/hmworld_data/wan_2_1/tv2v_encoded_full-hydra-style-camera}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://data_us_central1_a/maxdiffusion/wan/v2v-baseline-hydra}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${OUTPUT_DIR}/jax_cache/}"

# bs must be >=1: global batch = bs*128 shards over data*fsdp=128 (context=1). bs=1 => 1/device.
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"

# remat: FULL is stable + minimum HBM for the 1.3B. Do NOT use HIDDEN_STATE_WITH_OFFLOAD.
REMAT="${REMAT:-FULL}"

# Flash attention tiling. Speed round 2026-07-04 (v6e, 62.4K-token v2v): block 2048 =
# -29% step time vs default 512; 4096 OOMs vmem. MASK_PAD must stay True: splash pads
# 62400 -> next block multiple (2048 -> +1088 pad tokens), and mask_padding_tokens=False
# leaves that padding UNMASKED in attention -> NaN at scale (baseline step 2009, 2026-07-04).
FLASH_BLOCK="${FLASH_BLOCK:-2048}"
MASK_PAD="${MASK_PAD:-True}"

# Mesh (product must equal chips on the slice). CONTEXT/TENSOR PARALLELISM NaNs this
# model -> keep them at 1 and fill the slice with FSDP (see header note).
# Mesh: batch shards over data*fsdp; weights shard over fsdp only. For the 1.3B,
# fsdp=8 fits params+opt easily; d16/f8 shrinks the weight all-gather group 16x vs
# f128 (round-4 2026-07-04: d-meshes ~4% over f8 at 8-chip; at-scale gain measured live).
ICI_DATA="${ICI_DATA:-16}"
ICI_FSDP="${ICI_FSDP:-8}"
ICI_CONTEXT="${ICI_CONTEXT:-1}"
ICI_TENSOR="${ICI_TENSOR:-1}"

EVAL_EVERY="${EVAL_EVERY:-100000000}"          # effectively never (no eval-loss path anyway)
DEBUG="${DEBUG:-0}"
# ------------------------------------------------------------------------------

echo "V2V-baseline (HyDRA) '${RUN_NAME}'  lr=${LR}  steps=${MAX_STEPS}  ckpt_every=${CHECKPOINT_EVERY}"
echo "  official_sched=${HYDRA_OFFICIAL_SCHEDULER}  tokenizer_init=${HYDRA_TOKENIZER_INIT}"
echo "  ${TPU_NAME} (${ZONE})  mesh: d${ICI_DATA}/f${ICI_FSDP}/c${ICI_CONTEXT}/t${ICI_TENSOR}  bs=${PER_DEVICE_BS}  remat=${REMAT}"
echo "  data=${TRAIN_DATA_DIR}  ->  ${OUTPUT_DIR}/${RUN_NAME}"

DEBUG_ENV=""
if [ "${DEBUG}" = "1" ]; then
  DEBUG_ENV="export JAX_LOG_COMPILES=1; export JAX_TRACEBACK_FILTERING=off;"
  echo "  DEBUG on -> JAX_LOG_COMPILES=1  JAX_TRACEBACK_FILTERING=off"
fi

gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --worker=all \
  --command="$(cat <<EOF
set -euo pipefail
source ~/maxdiffusion_env.sh
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion
${DEBUG_ENV}

# Pull the committed code -- the TPU repo may be on an older commit (e.g. the
# re-encode left it at the encoder-camera commit, which predates train_wan_v2v.py).
git fetch -q origin && (git stash -u 2>/dev/null || true) && git checkout -B ${BRANCH} origin/${BRANCH} && git reset --hard origin/${BRANCH}

# Debug toggle (default empty = camera injection ON). Set V2V_DBG_NO_CAM=1 to skip
# the per-block camera injection (isolates it from the base attention under sharding).
export V2V_DBG_NO_CAM="${V2V_DBG_NO_CAM:-}"
# Optional XLA/libtpu perf flags (async collective fusion etc., the ti2v README set).
if [ -n "${XLA_PERF:-}" ]; then export LIBTPU_INIT_ARGS="${XLA_PERF:-}"; fi

# GCS read timeouts: turn an infinite stall into a retryable error.
export GCS_READ_REQUEST_TIMEOUT_SECS=60
export GCS_REQUEST_CONNECTION_TIMEOUT_SECS=30
export GCS_METADATA_REQUEST_TIMEOUT_SECS=30
export GCS_WRITE_REQUEST_TIMEOUT_SECS=600
# NOTE: do NOT force HF_HUB_OFFLINE=1 -- the pipeline needs the full Wan2.1 repo
# (model_index.json + vae/ + transformer/). If a worker's HF cache is incomplete,
# offline mode fails ("does not appear to have a file named config.json"); leaving
# it online lets HF fetch only the missing files (cached ones are reused).

# Make sure the EDITED src/ is what gets imported (not stale site-packages).
pip install -e ~/maxdiffusion --no-deps -q

WID=\$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" || echo 0)
mkdir -p \$HOME/train_logs
LOG=\$HOME/train_logs/${RUN_NAME}_w\${WID}.log

setsid nohup python src/maxdiffusion/train_wan_v2v.py \
  ${CONFIG_YML:-src/maxdiffusion/configs/base_wan_2_1_v2v.yml} \
  hydra=${HYDRA:-False} \
  hydra_official_training_scheduler=${HYDRA_OFFICIAL_SCHEDULER} \
  wan_hydra_tokenizer_init=${HYDRA_TOKENIZER_INIT} \
  attention=${ATTENTION:-flash} flash_min_seq_length=0 \
  "flash_block_sizes={\"block_q\":${FLASH_BLOCK},\"block_kv_compute\":${FLASH_BLOCK},\"block_kv\":${FLASH_BLOCK},\"block_q_dkv\":${FLASH_BLOCK},\"block_kv_dkv\":${FLASH_BLOCK},\"block_kv_dkv_compute\":${FLASH_BLOCK},\"block_q_dq\":${FLASH_BLOCK},\"block_kv_dq\":${FLASH_BLOCK},\"use_fused_bwd_kernel\":false}" \
  mask_padding_tokens=${MASK_PAD} \
  weights_dtype=bfloat16 activations_dtype=bfloat16 \
  dataset_type=tfrecord \
  train_data_dir=${TRAIN_DATA_DIR} \
  v2v_split_mode=${SPLIT_MODE:-all} \
  v2v_heldout_sids_file=${HELDOUT_SIDS:-gs://data_us_central1_a/hmworld_data/fun_camera_eval_set/heldout_test_sids_seed1234_n1000.txt} \
  cache_latents_text_encoder_outputs=True \
  load_tfrecord_cached=True \
  per_device_batch_size=${PER_DEVICE_BS} \
  ici_data_parallelism=${ICI_DATA} ici_fsdp_parallelism=${ICI_FSDP} ici_context_parallelism=${ICI_CONTEXT} ici_tensor_parallelism=${ICI_TENSOR} \
  dcn_data_parallelism=1 dcn_fsdp_parallelism=1 dcn_context_parallelism=1 dcn_tensor_parallelism=1 \
  allow_split_physical_axes=True \
  remat_policy=${REMAT} \
  max_train_steps=${MAX_STEPS} \
  checkpoint_every=${CHECKPOINT_EVERY} \
  save_optimizer=True \
  save_final_checkpoint=True \
  output_dir=${OUTPUT_DIR} \
  run_name=${RUN_NAME} \
  jax_cache_dir=${JAX_CACHE_DIR} \
  learning_rate=${LR} \
  warmup_steps_fraction=${WARMUP:-0.0} \
  log_period=1 \
  enable_data_shuffling=True \
  opt_enable_grad_global_norm_clipping=${CLIP:-False} \
  max_grad_norm=1.0 \
  eval_every=${EVAL_EVERY} \
  enable_generate_video_for_eval=False \
  enable_profiler=False \
  > "\$LOG" 2>&1 </dev/null &

echo "worker \${WID}: started pid \$! -> \$LOG"
EOF
)"
