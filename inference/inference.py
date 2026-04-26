#!/usr/bin/env python3
"""
Inference script for the fine-tuned Whisper Medium model.
Merged model path: /workspace/whisper-medium-librispeech/outputs_merged

Usage examples
--------------
# Transcribe a local .wav / .flac / .mp3 file:
  python inference.py --audio path/to/audio.wav

# Transcribe a URL (requires soundfile + requests):
  python inference.py --audio https://example.com/speech.wav

# Pull one sample from LibriSpeech val and transcribe it:
  python inference.py --sample

# Compute WER over N validation samples:
  python inference.py --benchmark --num_samples 200
"""

import argparse, io, math, numpy as np, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_PATH = "/workspace/whisper-medium-librispeech/outputs_merged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32


# ──────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────

def load_model():
    print(f"📦  Loading model from : {MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    print(f"✅  Ready on {DEVICE} ({DTYPE})\n")
    return model, processor


# ──────────────────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────────────────

def transcribe(model, processor, audio: np.ndarray, language: str = "en") -> str:
    """
    audio : 1-D float32 numpy array sampled at 16 kHz.
    """
    inputs = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        ids = model.generate(inputs, language=language, task="transcribe")

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ──────────────────────────────────────────────────────────
# Audio loading helpers
# ──────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int = 16_000) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    from scipy.signal import resample_poly
    g = math.gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)


def load_audio_file(path_or_url: str) -> np.ndarray:
    """
    Returns mono float32 numpy array at 16 kHz.
    Accepts local paths (wav/flac/mp3) or HTTP URLs.
    """
    import soundfile as sf

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests
        data = requests.get(path_or_url, timeout=30).content
        audio, sr = sf.read(io.BytesIO(data))
    else:
        audio, sr = sf.read(path_or_url)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return _resample(audio, sr)


def load_librispeech_sample(index: int = 0) -> tuple:
    """
    Returns (audio_np, reference_text) for one LibriSpeech validation sample.
    Uses soundfile to decode — no torchcodec needed.
    """
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    for i, sample in enumerate(ds):
        if i == index:
            # The audio dict contains 'array' (decoded float) when
            # soundfile/torchcodec is available; fall back to 'bytes'.
            if "array" in sample["audio"] and sample["audio"]["array"] is not None:
                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                sr    = sample["audio"]["sampling_rate"]
            elif "bytes" in sample["audio"] and sample["audio"]["bytes"]:
                audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
                audio = audio.astype(np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
            elif "path" in sample["audio"] and sample["audio"]["path"]:
                audio, sr = sf.read(sample["audio"]["path"])
                audio = audio.astype(np.float32)
            else:
                raise RuntimeError("Cannot decode audio: no array, bytes, or path.")

            audio = _resample(audio, sr)
            return audio, sample["text"]

    raise IndexError(f"Dataset has fewer than {index+1} samples.")


# ──────────────────────────────────────────────────────────
# High-level actions
# ──────────────────────────────────────────────────────────

def run_sample(model, processor, index: int = 0):
    print("🎵  Fetching LibriSpeech validation sample …")
    audio, ref = load_librispeech_sample(index)
    print(f"📝  Reference  : {ref}")
    hyp = transcribe(model, processor, audio)
    print(f"🤖  Hypothesis : {hyp}\n")


def run_benchmark(model, processor, num_samples: int = 100):
    import evaluate
    from datasets import load_dataset
    import soundfile as sf

    print(f"📊  Benchmarking WER on {num_samples} LibriSpeech validation samples …\n")
    wer_metric = evaluate.load("wer")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    refs, hyps = [], []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break

        # Decode audio robustly
        ad = sample["audio"]
        if ad.get("array") is not None:
            audio = np.array(ad["array"], dtype=np.float32)
            sr    = ad["sampling_rate"]
        elif ad.get("bytes"):
            audio, sr = sf.read(io.BytesIO(ad["bytes"]))
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        else:
            print(f"  [skip {i}] cannot decode audio")
            continue

        audio = _resample(audio, sr)
        ref = sample["text"].lower()
        hyp = transcribe(model, processor, audio).lower()
        refs.append(ref)
        hyps.append(hyp)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>4}/{num_samples}]  ref: {ref[:55]}")
            print(f"             hyp: {hyp[:55]}")

    wer = wer_metric.compute(predictions=hyps, references=refs)
    print(f"\n🎯  WER = {wer * 100:.2f}%  (over {len(refs)} samples)")
    return wer


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Whisper Medium STT inference")
    p.add_argument("--audio",        type=str,  help="Path or URL to audio file")
    p.add_argument("--language",     type=str,  default="en")
    p.add_argument("--sample",       action="store_true",
                   help="Transcribe first LibriSpeech validation sample")
    p.add_argument("--sample_index", type=int,  default=0,
                   help="Which validation sample index to use with --sample")
    p.add_argument("--benchmark",    action="store_true",
                   help="Compute WER over validation samples")
    p.add_argument("--num_samples",  type=int,  default=100)
    args = p.parse_args()

    model, processor = load_model()

    if args.audio:
        print(f"🎵  Transcribing: {args.audio}")
        audio = load_audio_file(args.audio)
        text  = transcribe(model, processor, audio, language=args.language)
        print(f"\n📝  Transcription:\n{text}\n")

    elif args.benchmark:
        run_benchmark(model, processor, args.num_samples)

    else:
        # Default (also triggered by --sample)
        run_sample(model, processor, index=args.sample_index)


if __name__ == "__main__":
    main()
