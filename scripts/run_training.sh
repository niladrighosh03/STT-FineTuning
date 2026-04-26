#!/usr/bin/env bash
# ============================================================
# RunPod H200 — Whisper Medium Fine-tuning on LibriSpeech
# Package manager: uv  (https://github.com/astral-sh/uv)
#
# Usage:
#   chmod +x scripts/run_training.sh
#   bash scripts/run_training.sh            # foreground — logs tee'd to output.log
#   nohup bash scripts/run_training.sh &    # detached background
# ============================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/STT-FineTuning"
LOG_DIR="${WORKSPACE}/whisper-medium-librispeech/logs"
OUT_DIR="${WORKSPACE}/whisper-medium-librispeech/outputs"

# Repo-root output.log (same location as before, easy to find)
REPO_LOG="${REPO_DIR}/output.log"

# ── Create all required dirs on volume disk ──────────────────
mkdir -p "${LOG_DIR}" "${OUT_DIR}" \
         "${WORKSPACE}/hf_home/hub" \
         "${WORKSPACE}/hf_home/datasets"

# ── HuggingFace cache → volume disk (set before any HF/uv call) ──────────
export HF_HOME="${WORKSPACE}/hf_home"
export HF_DATASETS_CACHE="${WORKSPACE}/hf_home/datasets"
export TRANSFORMERS_CACHE="${WORKSPACE}/hf_home/hub"
export HF_HUB_CACHE="${WORKSPACE}/hf_home/hub"

# ── CUDA / PyTorch tuning for H200 ──────────────────────────
export TOKENIZERS_PARALLELISM=false    # suppress HF tokenizer warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1              # force real-time output even when piped through tee

# ── All output goes to: repo/output.log  AND  /workspace/.../logs/train_<ts>.log
TIMESTAMPED_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# Redirect everything from this point on — both stdout and stderr.
# tee writes to output.log (repo root) and the timestamped volume log.
exec > >(tee -a "${REPO_LOG}" "${TIMESTAMPED_LOG}") 2>&1

echo "============================================================"
echo "🖥️  System info — $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h / /workspace 2>/dev/null || true

echo ""
echo "============================================================"
echo "📦  Installing / verifying dependencies (uv)"
echo "============================================================"
cd "${REPO_DIR}"

# uv is required — install it if missing (one-time, no sudo needed)
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
fi

echo "uv version: $(uv --version)"

# Sync the project's locked deps from uv.lock into the .venv
uv sync 

# torchcodec is NOT required — we use torchaudio/soundfile instead.
# soundfile is already in pyproject.toml; this is a safety-net.
# uv pip install --quiet soundfile

echo ""
echo "============================================================"
echo "🚀  Starting Whisper Medium fine-tuning"
echo "============================================================"
echo "📝  Logging to:"
echo "    ${REPO_LOG}"
echo "    ${TIMESTAMPED_LOG}"
echo ""

# Run inside the uv-managed venv
uv run python scripts/train.py

echo ""
echo "✅  Training script finished."
echo "    Model saved to: ${OUT_DIR}"
echo "    Logs in:        ${LOG_DIR}"
echo "    Repo log:       ${REPO_LOG}"
