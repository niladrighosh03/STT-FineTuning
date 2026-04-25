#!/usr/bin/env python3
"""
scripts/merge_adapter.py
════════════════════════
Merge a LoRA adapter into the base Whisper model and save the result as a
standard HuggingFace model directory.

After merging you can load the model with plain transformers — no PEFT needed:

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model     = WhisperForConditionalGeneration.from_pretrained("/workspace/whisper-medium-librispeech/merged")
    processor = WhisperProcessor.from_pretrained("/workspace/whisper-medium-librispeech/merged")

Usage
─────
    # Auto-detect paths from config.yaml:
    uv run python scripts/merge_adapter.py

    # Explicit paths:
    uv run python scripts/merge_adapter.py \
        --adapter_dir  /workspace/whisper-smoke-test/outputs \
        --merged_dir   /workspace/whisper-smoke-test/merged  \
        --base_model   openai/whisper-medium
"""

import os
import sys
import argparse

# Point HF cache to volume disk before any HF import
os.environ.setdefault("HF_HOME",            "/workspace/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE",  "/workspace/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_home/hub")
os.environ.setdefault("HF_HUB_CACHE",       "/workspace/hf_home/hub")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def merge(adapter_dir: str, merged_dir: str, base_model: str):
    """Load LoRA adapter, merge into base model, save full model to merged_dir."""

    print(f"\n{'='*60}")
    print("🔀  Merging LoRA adapter into base model")
    print(f"{'='*60}")
    print(f"  Base model   : {base_model}")
    print(f"  Adapter dir  : {adapter_dir}")
    print(f"  Output dir   : {merged_dir}")
    print()

    # ── 1. Load the PEFT model (base + adapter) ───────────────────────────────
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("📥  Loading base model...", flush=True)
    base = WhisperForConditionalGeneration.from_pretrained(base_model)

    print("📎  Loading LoRA adapter...", flush=True)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)

    # ── 2. Merge adapter weights into base & discard PEFT wrapper ─────────────
    print("🔀  Merging and unloading adapter...", flush=True)
    merged_model = peft_model.merge_and_unload()

    # ── 3. Save the merged model ──────────────────────────────────────────────
    os.makedirs(merged_dir, exist_ok=True)
    print(f"💾  Saving merged model to {merged_dir} ...", flush=True)
    merged_model.save_pretrained(merged_dir)

    # ── 4. Save the processor alongside the model ─────────────────────────────
    print("💾  Saving processor...", flush=True)
    processor = WhisperProcessor.from_pretrained(adapter_dir)
    processor.save_pretrained(merged_dir)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    import shutil
    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(merged_dir)
        for f in filenames
    ) / 1e9

    print()
    print(f"✅  Done!  Merged model saved to: {merged_dir}")
    print(f"   Total size on disk: {size_gb:.2f} GB")
    print()
    print("💡  Load for inference (no PEFT required):")
    print(f"    from transformers import WhisperForConditionalGeneration, WhisperProcessor")
    print(f"    model     = WhisperForConditionalGeneration.from_pretrained('{merged_dir}')")
    print(f"    processor = WhisperProcessor.from_pretrained('{merged_dir}')")


def main():
    # ── Try to read defaults from config.yaml ─────────────────────────────────
    default_adapter = "/workspace/whisper-medium-librispeech/outputs"
    default_merged  = "/workspace/whisper-medium-librispeech/merged"
    default_base    = "openai/whisper-medium"

    try:
        import yaml, os
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        default_adapter = cfg["training"].get("output_dir", default_adapter)
        default_merged  = default_adapter.rstrip("/") + "_merged"
        default_base    = cfg.get("model_name", default_base)
    except Exception:
        pass  # Silently fall back to hardcoded defaults

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base Whisper model")
    parser.add_argument("--adapter_dir", default=default_adapter,
                        help="Path to the saved LoRA adapter folder (output_dir from training)")
    parser.add_argument("--merged_dir",  default=default_merged,
                        help="Where to save the merged standalone model")
    parser.add_argument("--base_model",  default=default_base,
                        help="Base model name or path (default: openai/whisper-medium)")
    args = parser.parse_args()

    merge(args.adapter_dir, args.merged_dir, args.base_model)


if __name__ == "__main__":
    main()
