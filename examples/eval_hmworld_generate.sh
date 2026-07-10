#!/usr/bin/env bash
# Route A — multi-host generation for the HM-World (HyDRA-protocol) metric eval.
# Fans eval_fun_camera_visualize.py across ALL pod workers: each worker generates a
# strided shard (records[WID::HOST_COUNT]) of the curated eval set, named by sample_id,
# and uploads {sid}_gen.mp4 (+ {sid}_gt.mp4 if SAVE_GT) to VIZ_OUT. With 1000 clips on a
# 32-host v6e-128 that's ~31 clips/worker.
#
# PRE-LAUNCH RITUAL (the pod is SHARED): before running, confirm the pod is clear --
#   gcloud ... ssh TPU --worker=all --command='pgrep -fc "[p]ython"; ls /tmp/libtpu_lockfile'
# must show no maxdiffusion job / no lockfile. Only ONE launcher at a time on a pod.
#
# Checkpoint dispatch (WanCheckpointer2_2_FunCamera.load_checkpoint):
#   CKPT_RUN_DIR=""                              -> PAI base (the comparison baseline row)
#   CKPT_RUN_DIR=gs://.../lora-fix CKPT_STEP=95000 LORA_RANK=32 -> the trained LoRA
#
#   EVAL_DATA_DIR=gs://.../fun_camera_eval_set/encoded_seed1234_n1000 \
#   TPU_NAME=tpu-v6e-128-0 HOST_COUNT=32 \
#   CKPT_RUN_DIR=gs://data_us_central1_a/maxdiffusion/wan/fun_camera_lora/lora-fix \
#   CKPT_STEP=95000 LORA_RANK=32 RUN_TAG=lora-fix-95k ./eval_hmworld_generate.sh
set -euo pipefail

TPU_NAME="${TPU_NAME:-tpu-v6e-128-0}"
PROJECT="${PROJECT:-priors-medical-ai}"
ZONE="${ZONE:-us-central1-a}"
HOST_COUNT="${HOST_COUNT:-32}"          # #workers on the pod (v6e-128=32, v6e-256=64)

EVAL_DATA_DIR="${EVAL_DATA_DIR:?set EVAL_DATA_DIR to the curated eval-set TFRecord dir (encoded_seed1234_n1000)}"
CKPT_RUN_DIR="${CKPT_RUN_DIR:-}"        # training run's output_dir/run_name; ""=PAI base
CKPT_STEP="${CKPT_STEP:-}"              # ""=latest, else step number (e.g. 95000)
LORA_RANK="${LORA_RANK:-0}"            # MUST match the run for a LoRA ckpt (lora-fix=32); 0 for base
RUN_TAG="${RUN_TAG:-hmworld-eval}"
VIZ_OUT="${VIZ_OUT:-gs://data_us_central1_a/maxdiffusion/wan/hmworld_eval/${RUN_TAG}}"

if [ -n "${CKPT_RUN_DIR}" ]; then
  OUT_DERIVED="${CKPT_RUN_DIR%/*}"; NAME_DERIVED="${CKPT_RUN_DIR##*/}"
else
  OUT_DERIVED="${VIZ_OUT%/*}"; NAME_DERIVED="base-no-ckpt"   # no checkpoints there -> PAI base
fi

# HyDRA-protocol generation recipe: 480x832x153 (cond+tgt concat), CFG 5.0.
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-153}"
STEPS="${STEPS:-30}"                    # denoise steps (30 = prior viz; 50 = HyDRA infer default)
GUIDANCE="${GUIDANCE:-5.0}"            # CFG (certified recipe); real umT5 negative
SAVE_GT="${SAVE_GT:-False}"            # metrics use RAW tgt.mp4 as GT; VAE-decoded GT optional
PER_DEVICE_BS="${PER_DEVICE_BS:-0.25}"
VAE_SPATIAL="${VAE_SPATIAL:-4}"
TEXT_MODE="${TEXT_MODE:-embedding}"    # embedding = exact training cond (stored umT5 embeds)
NEG_PROMPT="${NEG_PROMPT:-}"

GC="${GC:-/home/spu9/google-cloud-sdk/bin/gcloud}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}"
RUN_ID="${RUN_ID:-hmworld-eval-$(date +%Y%m%d-%H%M%S)}"
# Shared pod safety: never kill another process unless the operator opts in
# after completing the pre-launch check above.
CLEAN_STALE_PROCESSES="${CLEAN_STALE_PROCESSES:-False}"

STEP_ARG=""
[ -n "${CKPT_STEP}" ] && STEP_ARG="eval_checkpoint_step=${CKPT_STEP}"

echo "HM-World eval-gen '${RUN_TAG}'  ${WIDTH}x${HEIGHT}x${NUM_FRAMES}  steps=${STEPS} gs=${GUIDANCE} lora_rank=${LORA_RANK}"
echo "  ${TPU_NAME} --worker=all  host_count=${HOST_COUNT}  ckpt=${CKPT_RUN_DIR:-PAI-base}${CKPT_STEP:+ @${CKPT_STEP}}"
echo "  eval=${EVAL_DATA_DIR}  ->  ${VIZ_OUT}   (log: ~/eval_logs/${RUN_ID}_w<WID>.log)"

# Run cleanup in a separate remote command. If pkill and launch share one
# command line, pkill -f can match the launcher shell itself because the later
# Python command contains the target process name, yielding a misleading SSH
# 255 before generation starts.
if [ "${CLEAN_STALE_PROCESSES}" = "True" ]; then
  "${GC}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" --worker=all \
    --command='pkill -9 -f "[e]val_fun_camera_visualize" 2>/dev/null || true; pkill -9 -f "[e]ncode_concat_camera" 2>/dev/null || true; sleep 2; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true'
fi

"${GC}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --worker=all \
  --command="$(cat <<EOF
set -euo pipefail
source ~/maxdiffusion_env.sh
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion
# NOTE: do NOT 'pip install -e' here — running it on all workers at once holds the SSH
# sessions long enough to trip gcloud's parallel ssh (exit 255). The script + config are
# read from the repo path directly, and the installed package is the correct branch.
# Per-host index from GCE metadata (same mechanism the encoders use).
WID=\$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" || echo 0)
# Single-host topology: each worker claims only its own 2x2 chips.
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
mkdir -p \$HOME/eval_logs
LOG=\$HOME/eval_logs/${RUN_ID}_w\${WID}.log
setsid nohup python -u src/eval/eval_fun_camera_visualize.py \
  src/maxdiffusion/configs/base_wan_2_2_fun_5b_camera.yml \
  eval_data_dir=${EVAL_DATA_DIR} \
  ${STEP_ARG} \
  run_name=${NAME_DERIVED} \
  output_dir=${OUT_DERIVED} \
  eval_output_dir=${VIZ_OUT} \
  eval_host_count=${HOST_COUNT} eval_host_index=\${WID} \
  lora_rank=${LORA_RANK} \
  height=${HEIGHT} width=${WIDTH} num_frames=${NUM_FRAMES} \
  num_inference_steps=${STEPS} eval_num_inference_steps=${STEPS} \
  guidance_scale=${GUIDANCE} eval_guidance_scale=${GUIDANCE} \
  eval_num_generate_samples=0 \
  eval_save_gt_video=${SAVE_GT} \
  eval_text_mode=${TEXT_MODE} \
  eval_negative_prompt='${NEG_PROMPT}' \
  per_device_batch_size=${PER_DEVICE_BS} vae_spatial=${VAE_SPATIAL} \
  skip_jax_distributed_system=True \
  > "\$LOG" 2>&1 </dev/null &
echo "worker \${WID}: started pid \$! -> \$LOG"
EOF
)"
echo "Launched on all ${HOST_COUNT} workers. Tail one: ~/eval_logs/${RUN_ID}_w0.log"
echo "Count outputs: ${GC} storage ls ${VIZ_OUT}/ | grep -c _gen.mp4   (expect ${HOST_COUNT}x shards -> ~1000)"
