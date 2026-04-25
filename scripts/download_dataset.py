#!/usr/bin/env python3
"""
scripts/download_dataset.py
═══════════════════════════
One-time dataset download for RunPod.

Downloads LibriSpeech 'clean' (train.360 + validation) from HuggingFace,
decodes all audio, and saves the materialised dataset to /workspace/datasets/
using save_to_disk().  Subsequent training runs load from disk instantly
— no internet needed, no re-streaming.

Disk space required (estimated):
  train.360  → ~26 GB  (104k samples, decoded float32 arrays)
  validation →  ~0.5 GB (2760 samples)

Usage (from repo root):
    uv run python scripts/download_dataset.py

    # Only validation (quick sanity check):
    uv run python scripts/download_dataset.py --splits validation

    # Smoke-test split (tiny):
    uv run python scripts/download_dataset.py --splits train.100,validation --max 500
"""

import os
import sys
import argparse

# ── HF cache → volume disk ───────────────────────────────────
os.environ.setdefault("HF_HOME",             "/workspace/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE",   "/workspace/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE",  "/workspace/hf_home/hub")
os.environ.setdefault("HF_HUB_CACHE",        "/workspace/hf_home/hub")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset, Audio, Dataset
from src.dataset import _decode_audio_safe


# ── Config ───────────────────────────────────────────────────
DATASET_NAME = "librispeech_asr"
SUBSET       = "clean"
SAVE_ROOT    = "/workspace/datasets/librispeech_clean"

# split name on HF  →  folder name under SAVE_ROOT
DEFAULT_SPLITS = {
    "train.360" : "train",       # full 360-hour training set (~104k samples)
    "validation": "validation",  # clean validation set (~2760 samples)
}


def download_split(split_name: str, save_dir: str, max_samples: int | None = None):
    """Stream one split, decode audio, save to disk."""
    if os.path.exists(save_dir):
        print(f"  ✅ '{split_name}' already on disk at {save_dir} — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Downloading split: {split_name}")
    if max_samples:
        print(f"  (capped at {max_samples:,} samples)")
    print(f"  Save path: {save_dir}")
    print(f"{'='*60}")

    # Stream with auto-decode disabled (avoids torchcodec)
    stream = load_dataset(
        DATASET_NAME,
        name=SUBSET,
        split=split_name,
        streaming=True,
    )
    stream = stream.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def sample_generator():
        for i, ex in enumerate(stream):
            yield ex  # 🚀 MAGIC HAPPENS HERE: Yield RAW compressed bytes (Saves 35GB of disk space!)
            if (i + 1) % 1000 == 0:
                print(f"    ... {i+1:,} samples processed", flush=True)
            if max_samples and (i + 1) >= max_samples:
                break

    print(f"  → Decoding and building dataset on-disk chunk-by-chunk (Memory Safe)...", flush=True)
    # from_generator streams directly to cached Arrow files without holding all objects in RAM!
    dataset = Dataset.from_generator(sample_generator)

    print(f"  → Saving to {save_dir} ...", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)

    print(f"  ✅ Saved {len(dataset):,} samples → {save_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech to /workspace/datasets/")
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS.keys()),
        help=f"Comma-separated HF split names. Default: {','.join(DEFAULT_SPLITS.keys())}",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=30000,
        help="Cap each split at this many samples (default 30,000 to save space).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the split already exists on disk.",
    )
    args = parser.parse_args()

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"\n🌍  HF_HOME           : {os.environ['HF_HOME']}")
    print(f"📂  Dataset save root : {SAVE_ROOT}")
    print(f"📋  Splits to download: {split_list}")
    print(f"🔢  Max samples/split : {args.max or 'all'}\n")

    for split in split_list:
        folder = DEFAULT_SPLITS.get(split, split.replace(".", "_"))
        save_dir = os.path.join(SAVE_ROOT, folder)

        if args.force and os.path.exists(save_dir):
            import shutil
            print(f"  --force: removing existing {save_dir}")
            shutil.rmtree(save_dir)

        download_split(split, save_dir, max_samples=args.max)

    print(f"\n🎉  All done! Dataset stored in: {SAVE_ROOT}")
    print("    Use  uv run python scripts/train.py  to start training.")
    print("    load_data() will auto-detect and load from disk.\n")


if __name__ == "__main__":
    main()
