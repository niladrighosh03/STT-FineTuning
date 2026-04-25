#!/usr/bin/env bash
# ============================================================
# RunPod H200 — Whisper Medium Fine-tuning on LibriSpeech
# Run from: /workspace/STT-FineTuning/
#
# Usage:
#   chmod +x scripts/run_training.sh
#   bash scripts/run_training.sh            # foreground (with nohup redirect)
#   nohup bash scripts/run_training.sh &    # detached background
# ============================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/STT-FineTuning"
LOG_DIR="${WORKSPACE}/whisper-medium-librispeech/logs"
OUT_DIR="${WORKSPACE}/whisper-medium-librispeech/outputs"

# ── Create all required dirs on volume disk ──────────────────
mkdir -p "${LOG_DIR}" "${OUT_DIR}" \
         "${WORKSPACE}/hf_home/hub" \
         "${WORKSPACE}/hf_home/datasets"

# ── HuggingFace cache → volume disk (must be set before pip install too) ─
export HF_HOME="${WORKSPACE}/hf_home"
export HF_DATASETS_CACHE="${WORKSPACE}/hf_home/datasets"
export TRANSFORMERS_CACHE="${WORKSPACE}/hf_home/hub"
export HF_HUB_CACHE="${WORKSPACE}/hf_home/hub"

# ── CUDA / PyTorch tuning for H200 ──────────────────────────
export TOKENIZERS_PARALLELISM=false    # suppress HF tokenizer warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "🖥️  System info"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h / /workspace 2>/dev/null || true

echo ""
echo "============================================================"
echo "📦  Installing / verifying dependencies"
echo "============================================================"
cd "${REPO_DIR}"

# Use uv if available, otherwise pip
if command -v uv &>/dev/null; then
    uv pip install --quiet -e .
else
    pip install --quiet -e .
fi

# torchcodec is NOT required — we use torchaudio/soundfile instead.
# Ensure soundfile is present (should already be in pyproject.toml).
pip install --quiet soundfile

echo ""
echo "============================================================"
echo "🚀  Starting Whisper Medium fine-tuning"
echo "============================================================"
python scripts/train.py 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅  Training script finished."
echo "    Model saved to: ${OUT_DIR}"
echo "    Logs in:        ${LOG_DIR}"
