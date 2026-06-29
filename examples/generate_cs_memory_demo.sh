#!/usr/bin/env bash
# Memory-conditioned generation on the Captain-Safari demo sample (Phase H).
# Single-host inference on ONE worker of tpu-v6e-8-debug.
#
# Loads PAI base + CS epoch-4 (LoRA r32 + memory) and generates with 3D memory:
#   StreamVGGT memory + pose tokens (from query/key camera matrices) -> retriever
#   -> per-block memory cross-attn. Camera clip -> Plucker control (fps 4->24).
#
# Assets are staged to GCS by stage_cs_memory_demo.sh, then pulled to TPU local disk
# here (np.load + safe_open need local paths). Output video -> GCS.
set -euo pipefail

TPU_NAME="${TPU_NAME:-tpu-v6e-8-debug}"
PROJECT="${PROJECT:-priors-medical-ai}"
ZONE="${ZONE:-us-central1-a}"
WORKER="${WORKER:-0}"

ASSET_GCS="${ASSET_GCS:-gs://data_us_central1_a/maxdiffusion/wan/cs_memory}"
OUT_GCS="${OUT_GCS:-gs://data_us_central1_a/maxdiffusion/wan/cs_memory/gen}"
RUN_TAG="${RUN_TAG:-cs-mem-demo}"

# CS res by default; downscale via env if 4 chips OOM at 704x1280x121.
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-121}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-0.125}"   # 8 chips * 0.125 = global batch 1
CHIP_BOUNDS="${CHIP_BOUNDS:-2,4,1}"       # all 8 chips of v6e-8 -> 2x HBM headroom (704x1280x121 is borderline on 4)
VAE_SPATIAL="${VAE_SPATIAL:-4}"   # must divide the latent spatial dim (matches working viz; 8 fails at 480x832)
# CS certified negative (the long Chinese WAN negative). Override with NEG_PROMPT.
NEG_PROMPT="${NEG_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"

GC="${GC:-/home/spu9/google-cloud-sdk/bin/gcloud}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}"

PROMPT="$(cat "$(dirname "$0")/.cs_demo_prompt.txt" 2>/dev/null || echo '')"

echo "Generate '${RUN_TAG}'  ${WIDTH}x${HEIGHT}x${NUM_FRAMES}  steps=${STEPS} gs=${GUIDANCE}"
echo "  ${TPU_NAME} worker=${WORKER}  assets=${ASSET_GCS}  ->  ${OUT_GCS}"

"${GC}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --worker="${WORKER}" \
  --command="$(cat <<EOF
set -euo pipefail
source ~/maxdiffusion_env.sh
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion
git fetch -q origin && (git stash -u 2>/dev/null || true) && git checkout -B wan22-memory origin/wan22-memory && git reset --hard origin/wan22-memory
pip install -e ~/maxdiffusion --no-deps -q

# Single-host topology: claim all 8 chips of this v6e-8 worker (2x HBM headroom).
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=${CHIP_BOUNDS}

# Pull assets to local disk (np.load + safe_open need local paths).
A=/tmp/cs_mem_assets
rm -rf \$A && mkdir -p \$A
# Pull ONLY the top-level asset files (NOT subdirs like parity/ or gen/ which now live
# under cs_memory/ and would waste disk / break the copy).
gsutil -m cp "${ASSET_GCS}/epoch-4.safetensors" "${ASSET_GCS}/memory.npy" \
  "${ASSET_GCS}/input_frame0.png" "${ASSET_GCS}/extrinsic_clip.npy" "${ASSET_GCS}/intrinsic_clip.npy" \
  "${ASSET_GCS}/extrinsic_query.npy" "${ASSET_GCS}/intrinsic_query.npy" \
  "${ASSET_GCS}/extrinsic_key.npy" "${ASSET_GCS}/intrinsic_key.npy" \$A/ 2>&1 | tail -2
ls -la \$A | head

python -m maxdiffusion.generate_wan_2_2_fun_camera \
  src/maxdiffusion/configs/base_wan_2_2_fun_5b_camera.yml \
  run_name="${RUN_TAG}" \
  model_name=wan2.2 model_type=TI2V-CC \
  per_device_batch_size=${PER_DEVICE_BS} vae_spatial=${VAE_SPATIAL} \
  height=${HEIGHT} width=${WIDTH} num_frames=${NUM_FRAMES} \
  num_inference_steps=${STEPS} guidance_scale=${GUIDANCE} fps=24 \
  lora_rank=32 lora_alpha=0 \
  wan_lora_path=\$A/epoch-4.safetensors \
  use_memory=True \
  wan_memory_path=\$A/epoch-4.safetensors \
  input_image_path=\$A/input_frame0.png \
  extrinsic_clip_path=\$A/extrinsic_clip.npy intrinsic_clip_path=\$A/intrinsic_clip.npy \
  memory_path=\$A/memory.npy memory_n_key=0 \
  extrinsic_query_path=\$A/extrinsic_query.npy intrinsic_query_path=\$A/intrinsic_query.npy \
  extrinsic_key_path=\$A/extrinsic_key.npy intrinsic_key_path=\$A/intrinsic_key.npy \
  prompt="${PROMPT}" negative_prompt="${NEG_PROMPT}" \
  output_video_name=/tmp/${RUN_TAG}.mp4

gsutil -m cp /tmp/${RUN_TAG}*.mp4 "${OUT_GCS}/"
echo "DONE -> ${OUT_GCS}/${RUN_TAG}*.mp4"
EOF
)"
