# TPU TFRecord Encoder for WAN TV2V

This directory contains TPU-VM-side preprocessing utilities for raw HMWorld-style TV2V data.

The encoder reads a JSONL manifest with text plus condition/target video paths, loads the WAN VAE and text encoder from MaxDiffusion, and writes MaxDiffusion-compatible TFRecords with:

- `latents`: target video VAE latents, channel-first `[C, F, H, W]`
- `cond_latents`: condition video VAE latents, channel-first `[C, F, H, W]`
- `encoder_hidden_states`: text encoder output `[512, hidden]`

The trainer interface stays unchanged. This script only changes where preprocessing happens.

## Minimal run

Run from inside the TPU VM with the MaxDiffusion venv active:

```bash
source ~/maxdiffusion_env.sh
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion

python /home/spu9/CaptainAtlas/tpu_tfrecord_encoder/encode_tv2v.py \
  --config src/maxdiffusion/configs/base_wan_ti2v_5b.yml \
  --manifest gs://YOUR_BUCKET/path/to/raw_tv2v_manifest.jsonl \
  --output-dir gs://YOUR_BUCKET/path/to/encoded_tv2v_tfrecords \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --records-per-shard 128 \
  --config-arg per_device_batch_size=0.0625 \
  --config-arg compile_text_encoder=False
```

For multi-host TPU slices, run the same command with `gcloud ... --worker=all`. Each host uses `jax.process_index()` and writes shard names prefixed with its host index.

## Manifest schema

Default fields:

```json
{"sample_id":"...", "caption":"...", "cond_video":"gs://.../cond.mp4", "tgt_video":"gs://.../tgt.mp4"}
```

Use `--caption-field`, `--condition-video-field`, and `--target-video-field` if your field names differ.
