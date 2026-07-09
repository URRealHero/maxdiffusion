#!/usr/bin/env bash
# LoRA fine-tune of PAI Wan2.1-Fun-V1.1-1.3B-Control-Camera (I2V-CC) on the
# HM-World concat+camera 2.1 records WITH clip_feature (*_full_clip dataset,
# built by tpu_tfrecord_encoder/{encode_clip_sidecar,merge_clip_sidecar}.py).
# Mirrors train_fun_camera_lora.sh (2.2 5B); trainer = Wan2_1FunCameraTrainer.
#
# Contract (parity-gated vs official, 2026-07-09): input 32ch = noisy(16) +
# y(16, first-frame latent only), clip [257,1280] -> img cross-attn, camera
# Plucker -> SimpleAdapter dscale 8, scalar timestep, flow_shift 5.
#
# Mesh = the fun-camera LoRA line's standard: d/f/c/t = 16/4/4/1 on a v6e-256,
# 8/4/4/1 on a v6e-128 (this launcher's default). ⚠ ctx=4 is VALIDATED for the
# 2.2 Fun-5B (15,210 tokens); this 1.3B carries 60,840 tokens (39 latent frames)
# = the v2v-class sequence where flash BWD deadlocked under ctx-sharding (§9.1).
# The 3-step debug-8 smoke below is the gate: if it deadlocks/wedges, fall back
# to ctx=1 (fsdp only) or ulysses@512.
#
# The transformer comes from the PAI repo; the T5/VAE/tokenizer/scheduler from
# Wan-AI/Wan2.1-T2V-1.3B-Diffusers (already in the hyperdisk HF cache). Point
# TRANSFORMER_PATH at a local dir holding the PAI diffusion_pytorch_model
# .safetensors + config.json to stay fully offline (HF_OFFLINE=1).
#
# Examples:
#   TPU_NAME=tpu-v6e-8-debug ICI_DATA=1 ICI_FSDP=2 ICI_CONTEXT=4 \
#     PER_DEVICE_BS=0.25 RUN_NAME=w21fc-smoke MAX_STEPS=3 \
#     ./train_wan21_fun_camera.sh                                     # smoke
#   TPU_NAME=tpu-v6e-128-1 RUN_NAME=w21fc-lora-10k MAX_STEPS=10000 \
#     EXCLUDE_SIDS=gs://data_us_central1_a/hmworld_data/fun_camera_eval_set/heldout_test_sids_seed1234_n1000.txt \
#     ./train_wan21_fun_camera.sh
set -euo pipefail

TPU_NAME="${TPU_NAME:-tpu-v6e-128-1}"
PROJECT="${PROJECT:-priors-medical-ai}"
ZONE="${ZONE:-us-central1-a}"

RUN_NAME="${RUN_NAME:-wan21-fun-camera-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://data_us_central1_a/maxdiffusion/wan/wan21_fun_camera_lora}"
DATASET_DIR="${DATASET_DIR:-gs://data_us_central1_a/hmworld_data/wan_2_1/concat_camera_encoded_full_clip/}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${OUTPUT_DIR}/jax_cache/}"
SEED="${SEED:-$RANDOM}"

TRAIN_PY="${TRAIN_PY:-src/maxdiffusion/train_wan_2_1_fun_camera.py}"
CONFIG="${CONFIG:-src/maxdiffusion/configs/base_wan_2_1_fun_camera.yml}"
# local dir with the PAI DiT (diffusion_pytorch_model.safetensors + config.json);
# "" => download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera (needs HF_OFFLINE=0
# and a WRITABLE HF_HUB_CACHE — the hyperdisk cache is read-only).
TRANSFORMER_PATH="${TRANSFORMER_PATH:-$HOME/wan21_fun_ckpt}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-153}"
MAX_STEPS="${MAX_STEPS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"   # fleet churns; keep dense
REMAT="${REMAT:-FULL}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# --- LoRA knobs (CS/DiffSynth recipe; control_adapter is PRETRAINED here,
# so default trainable set = LoRA only; add ",control_adapter" to also FT it) ---
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-0}"
LORA_TARGET="${LORA_TARGET:-q,k,v,o,ffn.0,ffn.2}"
LORA_TRAINABLE="${LORA_TRAINABLE:-lora_}"
WAN_LORA_PATH="${WAN_LORA_PATH:-}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-false}"
EXCLUDE_SIDS="${EXCLUDE_SIDS:-}"

# --- eval generation ---
EVAL_EVERY="${EVAL_EVERY:--1}"
EVAL_GEN_SAMPLES="${EVAL_GEN_SAMPLES:-2}"
EVAL_STEPS="${EVAL_STEPS:-20}"
EVAL_GS="${EVAL_GS:-1.0}"

# fun-camera LoRA mesh: 8/4/4/1 on a 128 (16/4/4/1 on a 256). See header re ctx.
ICI_DATA="${ICI_DATA:-8}"
ICI_FSDP="${ICI_FSDP:-4}"
ICI_CONTEXT="${ICI_CONTEXT:-4}"
ICI_TENSOR="${ICI_TENSOR:-1}"
PER_DEVICE_BS="${PER_DEVICE_BS:-0.25}"

DEBUG="${DEBUG:-0}"
XLA="${XLA-}"

echo "Wan2.1 Fun camera LoRA '${RUN_NAME}'  ${WIDTH}x${HEIGHT}x${NUM_FRAMES}  steps=${MAX_STEPS} lr=${LEARNING_RATE}"
echo "  lora: rank=${LORA_RANK} alpha=${LORA_ALPHA} targets=[${LORA_TARGET}] trainable=[${LORA_TRAINABLE}] init=${WAN_LORA_PATH:-fresh}"
echo "  ${TPU_NAME} (${ZONE})  mesh: d${ICI_DATA}/f${ICI_FSDP}/c${ICI_CONTEXT}/t${ICI_TENSOR}  bs=${PER_DEVICE_BS}  remat=${REMAT}"
echo "  data=${DATASET_DIR}  ->  ${OUTPUT_DIR}/${RUN_NAME}"

DEBUG_ENV=""
if [ "${DEBUG}" = "1" ]; then
  DEBUG_ENV="export JAX_LOG_COMPILES=1; export JAX_TRACEBACK_FILTERING=off;"
else
  DEBUG_ENV="unset JAX_LOG_COMPILES; unset JAX_TRACEBACK_FILTERING;"
fi

gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --worker=all \
  --command="$(cat <<EOF
set -euo pipefail
source ~/maxdiffusion_env.sh
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion
${DEBUG_ENV}
pip install -e ~/maxdiffusion --no-deps -q

WID=\$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" || echo 0)
mkdir -p \$HOME/train_logs
LOG=\$HOME/train_logs/${RUN_NAME}_w\${WID}.log

export LIBTPU_INIT_ARGS='${XLA}'
export HF_HUB_OFFLINE=${HF_OFFLINE:-1}

setsid nohup python ${TRAIN_PY} \
  ${CONFIG} \
  attention=flash weights_dtype=bfloat16 activations_dtype=bfloat16 \
  wan_transformer_pretrained_model_name_or_path='${TRANSFORMER_PATH}' \
  flow_shift=5.0 fps=15 \
  skip_jax_distributed_system=False \
  run_name=${RUN_NAME} \
  output_dir=${OUTPUT_DIR} \
  train_data_dir=${DATASET_DIR} \
  load_tfrecord_cached=True \
  height=${HEIGHT} width=${WIDTH} num_frames=${NUM_FRAMES} \
  jax_cache_dir=${JAX_CACHE_DIR} \
  max_train_steps=${MAX_STEPS} \
  checkpoint_every=${CHECKPOINT_EVERY} \
  learning_rate=${LEARNING_RATE} \
  lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} \
  lora_target_modules='${LORA_TARGET}' \
  lora_trainable_param_substrings='${LORA_TRAINABLE}' \
  wan_lora_path='${WAN_LORA_PATH}' \
  save_lora_only=${SAVE_LORA_ONLY} \
  exclude_sample_ids_path='${EXCLUDE_SIDS}' \
  enable_profiler=${ENABLE_PROFILER:-True} \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
  warmup_steps_fraction=${WARMUP_FRACTION:-0.1} \
  save_final_checkpoint=${SAVE_FINAL:-False} \
  seed=${SEED} \
  remat_policy=${REMAT} \
  flash_min_seq_length=0 \
  per_device_batch_size=${PER_DEVICE_BS} \
  ici_data_parallelism=${ICI_DATA} ici_fsdp_parallelism=${ICI_FSDP} ici_context_parallelism=${ICI_CONTEXT} ici_tensor_parallelism=${ICI_TENSOR} \
  allow_split_physical_axes=True \
  replicate_vae=True vae_spatial=4 \
  eval_every=${EVAL_EVERY} \
  enable_generate_video_for_eval=$([ "${EVAL_EVERY}" != "-1" ] && echo True || echo False) \
  eval_num_generate_samples=${EVAL_GEN_SAMPLES} \
  eval_num_inference_steps=${EVAL_STEPS} \
  eval_guidance_scale=${EVAL_GS} \
  log_period=1 \
  > "\$LOG" 2>&1 </dev/null &

echo "worker \${WID}: started pid \$! -> \$LOG"
EOF
)"
