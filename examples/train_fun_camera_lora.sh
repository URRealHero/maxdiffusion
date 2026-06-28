#!/usr/bin/env bash
# F3/F4: LoRA fine-tune of PAI WAN 2.2 Fun-5B Camera-Control on the HM-World
# dataset. Base weights are FROZEN (optax.set_to_zero); only params matching
# LORA_TRAINABLE (default "lora_") are trained. Trainer = Wan2_2FunCameraTrainer
# (same camera step as the full FT); the freezing happens in the optimizer.
#
# Knobs you'll typically set:
#   LORA_RANK=32 LORA_ALPHA=0                 # 0 alpha => scale = 1.0 (DiffSynth)
#   LORA_TARGET="q,k,v,o,ffn.0,ffn.2"         # which modules get an adapter
#   LORA_TRAINABLE="lora_"                    # train only LoRA (freeze base)
#   WAN_LORA_PATH=""                          # ""=fresh LoRA; or a CS .safetensors to continue
#   LEARNING_RATE=1e-4                        # CS LoRA recipe lr
#
# Examples:
#   RUN_NAME=lora-fresh MAX_STEPS=500 ./train_fun_camera_lora.sh
#   RUN_NAME=lora-from-cs WAN_LORA_PATH=/path/epoch-4.safetensors ./train_fun_camera_lora.sh
#   # also train the camera adapter, not just lora:
#   LORA_TRAINABLE="lora_,control_adapter" ./train_fun_camera_lora.sh
#
# PRE-LAUNCH (examples/TRAINING_CONFIG.md): idle sweep; workers on the branch
# tip; dataset _done markers complete.
set -euo pipefail

TPU_NAME="${TPU_NAME:-tpu-v6e-256-3}"
PROJECT="${PROJECT:-priors-medical-ai}"
ZONE="${ZONE:-us-central1-a}"

RUN_NAME="${RUN_NAME:-fun-camera-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://data_us_central1_a/maxdiffusion/wan/fun_camera_lora}"
DATASET_DIR="${DATASET_DIR:-gs://data_us_central1_a/hmworld_data/fun_camera_encoded_full/}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${OUTPUT_DIR}/jax_cache/}"
SEED="${SEED:-$RANDOM}"

TRAIN_PY="${TRAIN_PY:-src/maxdiffusion/train_wan_2_2_fun_camera.py}"
CONFIG="${CONFIG:-src/maxdiffusion/configs/base_wan_2_2_fun_5b_camera.yml}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-153}"
MAX_STEPS="${MAX_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
REMAT="${REMAT:-FULL}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# --- LoRA knobs ---
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-0}"
LORA_TARGET="${LORA_TARGET:-q,k,v,o,ffn.0,ffn.2}"
LORA_TRAINABLE="${LORA_TRAINABLE:-lora_}"      # freeze base, train only matches
WAN_LORA_PATH="${WAN_LORA_PATH:-}"             # ""=fresh; else continue from a LoRA file
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-false}"      # true => save only adapter tensors (few MB), restore = PAI base + overlay
EXCLUDE_SIDS="${EXCLUDE_SIDS:-}"               # gs:// or local txt of sample_ids to EXCLUDE from training (held-out test set)

# --- eval generation (camera-control videos saved during training) ---
EVAL_EVERY="${EVAL_EVERY:--1}"                 # >0 => every N steps, sample the batch + generate; -1 disables
EVAL_GEN_SAMPLES="${EVAL_GEN_SAMPLES:-2}"      # how many records to save per eval
EVAL_STEPS="${EVAL_STEPS:-20}"                 # denoise steps for eval gen (fewer = faster)
EVAL_GS="${EVAL_GS:-1.0}"                      # eval guidance scale (1.0 = no CFG, no text encoder)

ICI_DATA="${ICI_DATA:-16}"
ICI_FSDP="${ICI_FSDP:-4}"
ICI_CONTEXT="${ICI_CONTEXT:-4}"
ICI_TENSOR="${ICI_TENSOR:-1}"
PER_DEVICE_BS="${PER_DEVICE_BS:-0.25}"

DEBUG="${DEBUG:-0}"
XLA="${XLA-}"

echo "Fun camera LoRA '${RUN_NAME}'  ${WIDTH}x${HEIGHT}x${NUM_FRAMES}  steps=${MAX_STEPS} lr=${LEARNING_RATE}"
echo "  lora: rank=${LORA_RANK} alpha=${LORA_ALPHA} targets=[${LORA_TARGET}] trainable=[${LORA_TRAINABLE}] init=${WAN_LORA_PATH:-fresh}"
echo "  ${TPU_NAME} (${ZONE})  mesh: d${ICI_DATA}/f${ICI_FSDP}/c${ICI_CONTEXT}/t${ICI_TENSOR}  bs=${PER_DEVICE_BS}  remat=${REMAT}"
echo "  data=${DATASET_DIR}  ->  ${OUTPUT_DIR}/${RUN_NAME}"
if [ "${EVAL_EVERY}" != "-1" ]; then
  echo "  eval-gen: every ${EVAL_EVERY} steps, ${EVAL_GEN_SAMPLES} sample(s), ${EVAL_STEPS} steps, gs=${EVAL_GS}  -> ${OUTPUT_DIR}/${RUN_NAME}/eval/"
fi

DEBUG_ENV=""
if [ "${DEBUG}" = "1" ]; then
  DEBUG_ENV="export JAX_LOG_COMPILES=1; export JAX_TRACEBACK_FILTERING=off;"
  echo "  DEBUG on -> JAX_LOG_COMPILES=1  JAX_TRACEBACK_FILTERING=off"
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
export HF_HUB_OFFLINE=1

setsid nohup python ${TRAIN_PY} \
  ${CONFIG} \
  attention=flash weights_dtype=bfloat16 activations_dtype=bfloat16 \
  flow_shift=5.0 fps=16 \
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
  enable_profiler=True \
  skip_first_n_steps_for_profiler=3 \
  profiler_steps=3 \
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
