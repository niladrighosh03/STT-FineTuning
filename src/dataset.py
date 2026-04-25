import io
import os
import numpy as np
from datasets import load_dataset, Audio, Dataset, load_from_disk


# ── Disk location written by download_dataset.py ─────────────────────────────
_DATASET_DISK_ROOT = "/workspace/datasets/librispeech_clean"

_SPLIT_FOLDER = {
    "train.360" : "train",
    "train.100" : "train_100",
    "validation": "validation",
    "test"      : "test",
}


def _decode_audio_safe(example):
    """
    Safely decode audio bytes → numpy float32 array at 16 kHz.

    Tries backends in order of preference:
      1. Already decoded (numpy array present)   — no-op
      2. torchaudio (always available; in pyproject.toml deps)
      3. soundfile   (fast, lightweight)
      4. librosa     (slower but universal)
      5. Silent fallback (1 s of zeros) — should never reach this
    """
    audio = example.get("audio", {})

    # ── Already decoded ─────────────────────────────────────────────────
    if isinstance(audio, dict) and isinstance(audio.get("array"), np.ndarray):
        return example

    raw_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    path      = audio.get("path")  if isinstance(audio, dict) else None

    arr, sr = None, 16000

    # ── 1. torchaudio ──────────────────────────────────────────────────
    try:
        import torchaudio

        if raw_bytes:
            waveform, sr = torchaudio.load(io.BytesIO(raw_bytes))
        elif path and os.path.exists(path):
            waveform, sr = torchaudio.load(path)
        else:
            raise ValueError("No audio source available")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform  = resampler(waveform)
            sr = 16000
        arr = waveform.squeeze(0).numpy().astype(np.float32)

    except Exception:
        pass

    # ── 2. soundfile ───────────────────────────────────────────────────
    if arr is None:
        try:
            import soundfile as sf
            if raw_bytes:
                arr, sr = sf.read(io.BytesIO(raw_bytes))
            elif path and os.path.exists(path):
                arr, sr = sf.read(path)
            if arr is not None:
                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
        except Exception:
            pass

    # ── 3. librosa ─────────────────────────────────────────────────────
    if arr is None:
        try:
            import librosa
            if raw_bytes:
                arr, sr = librosa.load(io.BytesIO(raw_bytes), sr=16000, mono=True)
            elif path and os.path.exists(path):
                arr, sr = librosa.load(path, sr=16000, mono=True)
        except Exception as exc:
            print(f"[WARN] All audio decoders failed: {exc}. Using silence.", flush=True)

    # ── 4. Silent fallback ──────────────────────────────────────────────
    if arr is None:
        arr, sr = np.zeros(16000, dtype=np.float32), 16000

    # ── Resample if needed ───────────────────────────────────────────────
    if sr != 16000:
        try:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
        except Exception:
            pass

    # Store flat decoded array — NOT an Audio feature struct.
    # Do NOT cast_column(Audio(...)) on the materialised dataset; that would
    # trigger torchcodec to re-encode. preprocess_function reads these keys directly.
    example["audio"] = {"array": arr.astype(np.float32), "sampling_rate": 16000}
    return example


def _load_split_from_disk(split_name: str, max_samples: int) -> Dataset | None:
    """
    Try to load a pre-downloaded split from /workspace/datasets/.
    Returns None if it doesn't exist on disk.
    """
    folder = _SPLIT_FOLDER.get(split_name, split_name.replace(".", "_"))
    path   = os.path.join(_DATASET_DISK_ROOT, folder)

    if not os.path.exists(path):
        return None

    print(f"  📂 Loading from disk: {path}", flush=True)
    dataset = load_from_disk(path)

    # Optionally cap the number of samples
    if max_samples and len(dataset) > max_samples:
        print(f"     Capping to {max_samples:,} / {len(dataset):,} samples", flush=True)
        dataset = dataset.select(range(max_samples))

    print(f"  ✅ Loaded {len(dataset):,} samples from disk", flush=True)
    return dataset


def _stream_split(dataset_name: str, subset: str | None, split_name: str, max_samples: int) -> Dataset:
    """
    Fall-back: stream from HuggingFace and decode audio on the fly.
    """
    print(f"  🌐 Streaming from HuggingFace (no local cache found)...", flush=True)
    load_kwargs: dict = dict(streaming=True)
    if subset:
        load_kwargs["name"] = subset

    stream = load_dataset(dataset_name, split=split_name, **load_kwargs)
    # Disable auto-decode to avoid torchcodec requirement
    stream = stream.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    samples = []
    for i, ex in enumerate(stream):
        samples.append(ex)  # NO DECODING HERE! Keep it massively compressed in RAM
        if (i + 1) % 1000 == 0:
            print(f"    ... {i+1:,} samples streamed", flush=True)
        if len(samples) >= max_samples:
            break

    return Dataset.from_list(samples)


def load_data(config):
    """
    Load dataset for fine-tuning.

    Priority:
      1. /workspace/datasets/librispeech_clean/<split>/  (save_to_disk cache)
         → Downloaded once by scripts/download_dataset.py, loaded instantly.
      2. HuggingFace streaming  (fallback when disk cache absent)

    Config keys (config["dataset"]):
        name              HuggingFace dataset id
        subset            subset / config name
        train_split       e.g. 'train.360'
        val_split         e.g. 'validation'
        max_train_samples cap on training examples (default 28 000)
        max_val_samples   cap on validation examples (default 2 760)
    """
    dataset_name = config["dataset"]["name"]
    subset       = config["dataset"].get("subset", None)
    train_split  = config["dataset"]["train_split"]
    val_split    = config["dataset"]["val_split"]
    max_train    = config["dataset"].get("max_train_samples", 28_000)
    max_val      = config["dataset"].get("max_val_samples",  2_760)

    print(
        f"Loading '{dataset_name}' (subset={subset}) — "
        f"train: '{train_split}' (≤{max_train:,}), "
        f"val: '{val_split}' (≤{max_val:,})",
        flush=True,
    )

    # ── Training split ───────────────────────────────────────────────────
    print(f"\n  ⬇ Train split ({train_split}):", flush=True)
    train_dataset = (
        _load_split_from_disk(train_split, max_train)
        or _stream_split(dataset_name, subset, train_split, max_train)
    )

    # ── Validation split ─────────────────────────────────────────────────
    print(f"\n  ⬇ Val split ({val_split}):", flush=True)
    val_dataset = (
        _load_split_from_disk(val_split, max_val)
        or _stream_split(dataset_name, subset, val_split, max_val)
    )

    print(
        f"\n✅ Dataset ready — {len(train_dataset):,} train / {len(val_dataset):,} val",
        flush=True,
    )
    return train_dataset, val_dataset
