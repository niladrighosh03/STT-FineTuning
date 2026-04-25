import io
import os
import numpy as np
from datasets import load_dataset, Audio, Dataset


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

    # ── Already decoded by datasets Audio feature ──────────────────────
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

        # Mix down to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform  = resampler(waveform)
            sr = 16000

        arr = waveform.squeeze(0).numpy().astype(np.float32)

    except Exception:
        pass  # fall through

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
                if arr.ndim > 1:          # stereo → mono
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
            print(f"[WARN] All audio decoders failed: {exc}. Using silence.")

    # ── 4. Silent fallback ──────────────────────────────────────────────
    if arr is None:
        arr, sr = np.zeros(16000, dtype=np.float32), 16000

    # ── Resample if a non-torchaudio path was taken ─────────────────────
    if sr != 16000:
        try:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16000)
        except Exception:
            pass   # best-effort

    example["audio"] = {"array": arr.astype(np.float32), "sampling_rate": 16000}
    return example


def load_data(config):
    """
    Load a sufficiently large streaming subset of LibriSpeech for fine-tuning
    on H200. All data is streamed so nothing is cached to disk up front.

    Config keys read from config["dataset"]:
        name                  HuggingFace dataset id  (librispeech_asr)
        subset                config name / subset     (clean)
        train_split           e.g. 'train.360'
        val_split             e.g. 'validation'
        max_train_samples     number of training examples  (default 28 000)
        max_val_samples       number of validation examples (default 2 760)

    RunPod storage note:
        HF_HOME / HF_DATASETS_CACHE is set to /workspace so large files land
        on the 60 GB volume disk, not the 20 GB container disk.
    """
    dataset_name = config["dataset"]["name"]
    subset       = config["dataset"].get("subset", None)
    train_split  = config["dataset"]["train_split"]
    val_split    = config["dataset"]["val_split"]
    max_train    = config["dataset"].get("max_train_samples", 28_000)
    max_val      = config["dataset"].get("max_val_samples",  2_760)

    print(
        f"Loading '{dataset_name}' (subset={subset}) — "
        f"train_split={train_split} (up to {max_train:,} samples), "
        f"val_split={val_split} (up to {max_val:,} samples)...",
        flush=True,
    )

    # ── Build load kwargs ────────────────────────────────────────────────
    # NOTE: trust_remote_code is no longer accepted by HF datasets ≥ 2.x
    # NOTE: decode=False prevents datasets from trying to use torchcodec;
    #       we handle audio decoding ourselves via _decode_audio_safe().
    load_kwargs = dict(streaming=True)
    if subset:
        load_kwargs["name"] = subset

    train_stream = load_dataset(dataset_name, split=train_split, **load_kwargs)
    val_stream   = load_dataset(dataset_name, split=val_split,   **load_kwargs)

    # Disable the datasets built-in audio decoder so it doesn't try to call
    # torchcodec.  We decode the raw bytes ourselves in _decode_audio_safe().
    # cast_column with decode=False makes streaming yield {"bytes": ..., "path": ...}
    # instead of trying to auto-decode.
    train_stream = train_stream.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    val_stream   = val_stream.cast_column("audio",   Audio(sampling_rate=16000, decode=False))

    # ── Materialise streaming subsets with safe audio decoding ──────────
    print(f"  Streaming {max_train:,} train samples...", flush=True)
    train_list = []
    for i, ex in enumerate(train_stream):
        train_list.append(_decode_audio_safe(ex))
        if (i + 1) % 1000 == 0:
            print(f"    ... {i + 1:,} train samples loaded", flush=True)
        if len(train_list) >= max_train:
            break

    print(f"  Streaming {max_val:,} val samples...", flush=True)
    val_list = []
    for i, ex in enumerate(val_stream):
        val_list.append(_decode_audio_safe(ex))
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1:,} val samples loaded", flush=True)
        if len(val_list) >= max_val:
            break

    train_dataset = Dataset.from_list(train_list)
    val_dataset   = Dataset.from_list(val_list)

    # Cast audio column so datasets knows the dtype/sr
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset   = val_dataset.cast_column("audio",   Audio(sampling_rate=16000))

    print(
        f"✅ Dataset ready — {len(train_dataset):,} train / {len(val_dataset):,} val",
        flush=True,
    )
    return train_dataset, val_dataset
