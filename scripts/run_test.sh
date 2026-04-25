#!/usr/bin/env bash
# ============================================================
# RunPod H200 — SMOKE TEST (tiny subset, ~2–3 min)
# Package manager: uv
#
# Purpose:
#   Runs the FULL training pipeline on just 16 train / 8 val
#   samples for 1 epoch to verify correctness before the real run.
#
# Usage:
#   chmod +x scripts/run_test.sh
#   bash scripts/run_test.sh
# ============================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/STT-FineTuning"
LOG_DIR="${WORKSPACE}/whisper-smoke-test/logs"
OUT_DIR="${WORKSPACE}/whisper-smoke-test/outputs"
TEST_LOG="${REPO_DIR}/test_output.log"
TIMESTAMPED_LOG="${LOG_DIR}/smoke_test_$(date +%Y%m%d_%H%M%S).log"

# ── Create dirs on volume disk ───────────────────────────────
mkdir -p "${LOG_DIR}" "${OUT_DIR}" \
         "${WORKSPACE}/hf_home/hub" \
         "${WORKSPACE}/hf_home/datasets"

# ── HuggingFace cache → volume disk ─────────────────────────
export HF_HOME="${WORKSPACE}/hf_home"
export HF_DATASETS_CACHE="${WORKSPACE}/hf_home/datasets"
export TRANSFORMERS_CACHE="${WORKSPACE}/hf_home/hub"
export HF_HUB_CACHE="${WORKSPACE}/hf_home/hub"

# ── CUDA tuning ──────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1       # force real-time output even when piped through tee

# ── All stdout+stderr → test_output.log + timestamped log ───
exec > >(tee -a "${TEST_LOG}" "${TIMESTAMPED_LOG}") 2>&1

echo "============================================================"
echo "🧪  SMOKE TEST — Whisper Medium / LibriSpeech"
echo "    16 train samples | 8 val samples | 1 epoch | batch=4"
echo "    $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
df -h / /workspace 2>/dev/null || true

echo ""
echo "📦  Syncing deps with uv..."
cd "${REPO_DIR}"
uv sync --quiet

echo ""
echo "🚀  Launching smoke-test training..."
echo "    Config : configs/config_test.yaml"
echo "    Log    : ${TEST_LOG}"
echo ""

# Pass config path as env var so train.py can pick it up
TRAIN_CONFIG="${REPO_DIR}/configs/config_test.yaml" \
    uv run python -u scripts/train.py

EXIT_CODE=$?

echo ""
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "✅  Smoke test PASSED — pipeline is correct!"
    echo "    All stages completed: data load → model build →"
    echo "    preprocessing → training → eval → save"
else
    echo "❌  Smoke test FAILED (exit code: ${EXIT_CODE})"
    echo "    Check logs above or: cat ${TEST_LOG}"
fi

echo ""
echo "📝  Shell log: ${TIMESTAMPED_LOG}"
echo "    Python will also self-log to: /workspace/STT-FineTuning/test_output.log"
echo ""

exit "${EXIT_CODE}"
