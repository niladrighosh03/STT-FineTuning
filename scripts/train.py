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


# ==============================================================================
# Python-level Tee: mirrors ALL stdout + stderr to output.log in real time.
# This runs INSIDE Python, so every print(), warning, traceback, and HF log
# line gets captured — regardless of how the script was launched.
# ==============================================================================

class _Tee:
    """
    Write to two streams simultaneously (e.g. stdout + log file).
    Line-buffered (buffering=1) so every line appears in the file immediately.
    """
    def __init__(self, primary_stream, filepath: str, mode: str = "a"):
        self._primary = primary_stream
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._file = open(filepath, mode, buffering=1, encoding="utf-8", errors="replace")

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._file.flush()

    def fileno(self):
        # Return the real fd so subprocesses / C-extensions can write to it
        return self._primary.fileno()

    def isatty(self) -> bool:
        return False

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    # Make it usable as a context manager
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def setup_python_logging(log_path: str):
    """
    Redirect Python's stdout AND stderr to both the terminal and log_path.
    Call this once, near the top of main(), BEFORE any other output.
    """
    sys.stdout = _Tee(sys.__stdout__, log_path)
    sys.stderr = _Tee(sys.__stderr__, log_path)
    print(f"📝  All Python output mirrored to: {log_path}", flush=True)


# ==============================================================================

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
    # ------------------------------------------------------------------
    # 0. Determine log path & set up Python-level tee logging
    #    output.log lives in the repo root (easy to find).
    #    For smoke tests the env var TRAIN_CONFIG is set, so we pick a
    #    separate test_output.log automatically.
    # ------------------------------------------------------------------
    _default_cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    config_path  = os.environ.get("TRAIN_CONFIG", _default_cfg)

    is_test = "config_test" in os.path.basename(config_path)
    log_filename = "test_output.log" if is_test else "output.log"
    log_path = os.path.join(os.path.dirname(__file__), "..", log_filename)
    log_path = os.path.abspath(log_path)

    setup_python_logging(log_path)   # ← ALL output from here on goes to log

    print("=" * 60, flush=True)
    print("🚀  Whisper Medium Fine-tuning  (H200 / RunPod optimised)", flush=True)
    print("=" * 60, flush=True)

    print_gpu_info()

    print("\n💾  Disk status:", flush=True)
    print_disk_usage()

    print("\n📄  Loading config...", flush=True)
    print(f"    Config : {os.path.abspath(config_path)}", flush=True)
    print(f"    Log    : {log_path}", flush=True)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n📦  Loading datasets...", flush=True)
    train_dataset, val_dataset = load_data(config)

    # ------------------------------------------------------------------
    # 2. Load processor ONLY
    #    (We must map the dataset before building the PyTorch model, 
    #     otherwise multiprocessing deadlocks due to CUDA initialization).
    # ------------------------------------------------------------------
    print("\n🏗️   Loading processor...", flush=True)
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(
        config["model_name"], language="en", task="transcribe"
    )

    # ------------------------------------------------------------------
    # 3. Pre-process audio → log-mel features
    #    Use multiple workers to saturate the H200 data pipeline.
    # ------------------------------------------------------------------
    # Cap num_proc: HF datasets deadlocks when workers >> samples.
    # Rule: at most 1 worker per 4 samples, and never more than cpu_count.
    num_proc = min(
        max(1, len(train_dataset) // 4),
        min(8, os.cpu_count() or 1),
    )
    print(f"\n⚙️   Preprocessing datasets (num_proc={num_proc}, "
          f"train={len(train_dataset)}, val={len(val_dataset)})...", flush=True)

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
    # 3.5 Build model (AFTER multiprocessing to prevent CUDA deadlocks)
    # ------------------------------------------------------------------
    print("\n🏗️   Building model...", flush=True)
    model, _ = build_model(config)

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
    print(f"\n💾  Saving LoRA adapter to {output_dir}...", flush=True)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("✅  Adapter saved!", flush=True)

    # ------------------------------------------------------------------
    # 7. Merge LoRA adapter → full standalone model (inference-ready)
    # ------------------------------------------------------------------
    merged_dir = output_dir.rstrip("/") + "_merged"
    print(f"\n🔀  Merging adapter into base model → {merged_dir}", flush=True)
    try:
        from merge_adapter import merge
        merge(
            adapter_dir = output_dir,
            merged_dir  = merged_dir,
            base_model  = config["model_name"],
        )
        print(f"✅  Merged model ready at: {merged_dir}", flush=True)
    except Exception as exc:
        # Non-fatal: adapter is still usable even if merge fails
        print(f"⚠️  Merge step failed (adapter is still saved): {exc}", flush=True)
        print(f"    You can merge manually later:\n"
              f"    uv run python scripts/merge_adapter.py "
              f"--adapter_dir {output_dir} --merged_dir {merged_dir}", flush=True)

    print("\n💾  Final disk status:", flush=True)
    print_disk_usage()


if __name__ == "__main__":
    main()