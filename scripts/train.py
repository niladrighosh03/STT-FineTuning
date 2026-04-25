import os
import sys
import torch

# ──────────────────────────────────────────────────────────────────────────────
# RunPod: redirect ALL HuggingFace caches to the 60 GB volume disk (/workspace)
# so the 20 GB container disk never fills up.  Must be set BEFORE any HF import.
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME",             "/workspace/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE",   "/workspace/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE",  "/workspace/hf_home/hub")
os.environ.setdefault("HF_HUB_CACHE",        "/workspace/hf_home/hub")

# Ensure the cache dirs exist
for _d in [
    "/workspace/hf_home",
    "/workspace/hf_home/datasets",
    "/workspace/hf_home/hub",
    "/workspace/whisper-medium-librispeech/outputs",
    "/workspace/whisper-medium-librispeech/logs",
]:
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from src.dataset import load_data
from src.model import build_model
from src.utils import preprocess_function
from src.collator import DataCollatorSpeechSeq2Seq
from src.metrics import compute_metrics
from src.trainer import get_trainer


def print_gpu_info():
    """Log GPU info so we know what hardware we are running on."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name}  |  VRAM: {vram:.1f} GB", flush=True)
    else:
        print("  ⚠️  No CUDA GPU detected — training on CPU (will be slow)", flush=True)


def print_disk_usage():
    """Print disk usage for container and volume disks."""
    import shutil
    for path, label in [("/", "Container disk"), ("/workspace", "Volume disk")]:
        try:
            total, used, free = shutil.disk_usage(path)
            print(
                f"  {label} ({path}): "
                f"{used/1e9:.1f} GB used / {total/1e9:.1f} GB total "
                f"({free/1e9:.1f} GB free)",
                flush=True,
            )
        except Exception:
            pass


def main():
    print("=" * 60, flush=True)
    print("🚀  Whisper Medium Fine-tuning  (H200 / RunPod optimised)", flush=True)
    print("=" * 60, flush=True)

    print_gpu_info()

    print("\n💾  Disk status:", flush=True)
    print_disk_usage()

    print("\n📄  Loading config...", flush=True)
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n📦  Loading datasets...", flush=True)
    train_dataset, val_dataset = load_data(config)

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    print("\n🏗️   Building model and processor...", flush=True)
    model, processor = build_model(config)

    # ------------------------------------------------------------------
    # 3. Pre-process audio → log-mel features
    #    Use multiple workers to saturate the H200 data pipeline.
    # ------------------------------------------------------------------
    num_proc = min(8, os.cpu_count() or 1)
    print(f"\n⚙️   Preprocessing datasets (num_proc={num_proc})...", flush=True)

    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, processor),
        num_proc=num_proc,
        desc="Preprocessing train",
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, processor),
        num_proc=num_proc,
        desc="Preprocessing val",
    )

    # Remove raw audio column to save memory
    cols_to_remove = [
        c for c in ["audio", "file", "speaker_id", "chapter_id", "id"]
        if c in train_dataset.column_names
    ]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        val_dataset   = val_dataset.remove_columns(
            [c for c in cols_to_remove if c in val_dataset.column_names]
        )

    # ------------------------------------------------------------------
    # 4. Data collator + trainer
    # ------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2Seq(processor)

    trainer = get_trainer(
        config,
        model,
        processor,
        train_dataset,
        val_dataset,
        data_collator,
        compute_metrics,
    )

    print(f"\n📊  Train samples : {len(train_dataset):,}", flush=True)
    print(f"📊  Val   samples : {len(val_dataset):,}", flush=True)

    # ------------------------------------------------------------------
    # 5. Train!
    # ------------------------------------------------------------------
    print("\n🔥  Starting training...", flush=True)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.", flush=True)

    # ------------------------------------------------------------------
    # 6. Save model + processor to /workspace
    # ------------------------------------------------------------------
    output_dir = config["training"]["output_dir"]
    print(f"\n💾  Saving model to {output_dir}...", flush=True)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("✅  Done!", flush=True)

    print("\n💾  Final disk status:", flush=True)
    print_disk_usage()


if __name__ == "__main__":
    main()