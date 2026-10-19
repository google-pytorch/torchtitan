#!/usr/bin/env bash
# ==============================================================================
# Gemma 4 E2B Post-Training Workshop Runner (TorchTitan-TorchTPU + vLLM-TorchTPU)
# ==============================================================================
# Usage:
#   bash run_workshop.sh [download|cpt|sft|prepare_vllm|infer_offline|infer_serve|all]
#
# Environment variables (with defaults):
#   NPROC_PER_NODE : Number of TPU chips on the host (default: 4 for v6e-4)
#   HF_BASE_MODEL  : Base HF model for CPT (default: google/gemma-4-E2B)
#   HF_IT_MODEL    : Base HF model for SFT (default: google/gemma-4-E2B-it)
#   ASSETS_DIR     : Local directory for downloaded HF assets (default: ./assets/hf)
#   STEPS          : Number of training steps per lab (default: 200)
# ==============================================================================

set -euo pipefail

STAGE="${1:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
HF_BASE_MODEL="${HF_BASE_MODEL:-google/gemma-4-E2B}"
HF_IT_MODEL="${HF_IT_MODEL:-google/gemma-4-E2B-it}"
ASSETS_DIR="${ASSETS_DIR:-./assets/hf}"
STEPS="${STEPS:-200}"

BASE_MODEL_NAME="$(basename "${HF_BASE_MODEL}")"
IT_MODEL_NAME="$(basename "${HF_IT_MODEL}")"
BASE_ASSETS_PATH="${ASSETS_DIR}/${BASE_MODEL_NAME}"
IT_ASSETS_PATH="${ASSETS_DIR}/${IT_MODEL_NAME}"

CPT_DUMP_FOLDER="./outputs/gemma4_e2b_cpt"
SFT_DUMP_FOLDER="./outputs/gemma4_e2b_sft"
WORKSHOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clean_tpu_locks() {
  rm -f /tmp/libtpu_lockfile /tmp/libtpu* 2>/dev/null || true
}

run_download() {
  echo "=== [Step 0] Downloading Hugging Face Assets to ${ASSETS_DIR} ==="
  TOKEN_ARGS=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_ARGS=(--hf_token "${HF_TOKEN}")
  fi
  python3 -m torchtitan.scripts.download_hf_assets \
    --repo_id "${HF_BASE_MODEL}" \
    --local_dir "${ASSETS_DIR}" \
    --all \
    "${TOKEN_ARGS[@]}"

  if [[ "${HF_IT_MODEL}" != "${HF_BASE_MODEL}" ]]; then
    python3 -m torchtitan.scripts.download_hf_assets \
      --repo_id "${HF_IT_MODEL}" \
      --local_dir "${ASSETS_DIR}" \
      --all \
      "${TOKEN_ARGS[@]}"
  fi
}

run_cpt() {
  echo "=== [Lab 2] Continuous Pre-Training (CPT) on codeparrot/codeparrot-clean-valid ==="
  clean_tpu_locks
  PJRT_DEVICE=TPU torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.transformers_modeling_backend \
    --config=gemma4_e2b_cpt_codeparrot \
    --hf_model="${HF_BASE_MODEL}" \
    --hf_assets_path="${BASE_ASSETS_PATH}" \
    --dump_folder="${CPT_DUMP_FOLDER}" \
    --training.steps="${STEPS}" \
    --checkpoint.enable \
    --checkpoint.initial_load_in_hf \
    --checkpoint.initial_load_model_only \
    --checkpoint.last_save_model_only \
    --checkpoint.last_save_in_hf
}

run_sft() {
  echo "=== [Lab 3] Supervised Fine-Tuning (SFT) on iamtarun/python_code_instructions_18k_alpaca ==="
  clean_tpu_locks
  EXTRA_LOAD_ARGS=()
  CPT_FINAL_CKPT="$(pwd)/${CPT_DUMP_FOLDER}/checkpoint/step-${STEPS}"
  if [[ -d "${CPT_FINAL_CKPT}" ]]; then
    echo "Found Lab 2 CPT HF checkpoint at ${CPT_FINAL_CKPT}; chaining into Lab 3 SFT..."
    EXTRA_LOAD_ARGS=(--checkpoint.initial_load_path="${CPT_FINAL_CKPT}")
  fi

  PJRT_DEVICE=TPU torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    -m torchtitan.experiments.tpu.train \
    --module=torchtitan.experiments.tpu.transformers_modeling_backend \
    --config=gemma4_e2b_sft_alpaca \
    --hf_model="${HF_IT_MODEL}" \
    --hf_assets_path="${IT_ASSETS_PATH}" \
    --dump_folder="${SFT_DUMP_FOLDER}" \
    --training.steps="${STEPS}" \
    --checkpoint.enable \
    --checkpoint.initial_load_in_hf \
    --checkpoint.initial_load_model_only \
    --checkpoint.last_save_model_only \
    --checkpoint.last_save_in_hf \
    "${EXTRA_LOAD_ARGS[@]}"
}

run_prepare_vllm() {
  echo "=== [Lab 4 Prep] Packaging SFT Checkpoint Metadata for vLLM-TorchTPU ==="
  SFT_FINAL_CKPT="${SFT_DUMP_FOLDER}/checkpoint/step-${STEPS}"
  python3 "${WORKSHOP_DIR}/prepare_hf_checkpoint_for_vllm.py" \
    --checkpoint_dir "${SFT_FINAL_CKPT}" \
    --hf_assets_path "${IT_ASSETS_PATH}"
}

run_infer_offline() {
  echo "=== [Lab 4A] Offline Inference with vLLM-TorchTPU ==="
  clean_tpu_locks
  SFT_FINAL_CKPT="${SFT_DUMP_FOLDER}/checkpoint/step-${STEPS}"
  export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
  export TORCHINDUCTOR_AUTOGRAD_CACHE=0
  python3 "${WORKSHOP_DIR}/lab4_vllm_torchtpu_inference.py" \
    --model "${SFT_FINAL_CKPT}" \
    --tensor_parallel_size 1 \
    --max_model_len 1024 \
    --max_tokens 256
}

run_infer_serve() {
  echo "=== [Lab 4B] Online OpenAI-Compatible Serving with vLLM-TorchTPU ==="
  clean_tpu_locks
  SFT_FINAL_CKPT="${SFT_DUMP_FOLDER}/checkpoint/step-${STEPS}"
  export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
  export TORCHINDUCTOR_AUTOGRAD_CACHE=0
  vllm serve "${SFT_FINAL_CKPT}" \
    --tensor_parallel_size=1 \
    --max-model-len=1024 \
    --max-num-batched-tokens=1024 \
    --attention-backend CUSTOM
}

case "${STAGE}" in
  download)
    run_download
    ;;
  cpt)
    run_cpt
    ;;
  sft)
    run_sft
    ;;
  prepare_vllm)
    run_prepare_vllm
    ;;
  infer_offline)
    run_infer_offline
    ;;
  infer_serve)
    run_infer_serve
    ;;
  all)
    run_download
    run_cpt
    run_sft
    run_prepare_vllm
    run_infer_offline
    ;;
  *)
    echo "Unknown stage '${STAGE}'. Expected: download | cpt | sft | prepare_vllm | infer_offline | infer_serve | all"
    exit 1
    ;;
esac
