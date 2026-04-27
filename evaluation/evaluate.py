#!/usr/bin/env python3
"""
Evaluate a fine-tuned Whisper model on LibriSpeech and save per-sample WER/CER.

Run:
  uv run python evaluation/evaluate_librispeech.py

Change the constants in EVALUATION CONFIG below if you need a different split,
sample count, model path, or output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import string
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Keep HuggingFace caches on the workspace volume, matching the training script.
os.environ.setdefault("HF_HOME", "/workspace/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/hf_home/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_home/hub")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf_home/hub")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset, load_from_disk
from jiwer import cer, wer
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - PEFT is optional for merged checkpoints.
    PeftModel = None

try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
except Exception:  # pragma: no cover - fallback is used if Transformers moves it.
    BasicTextNormalizer = None

from src.dataset import _decode_audio_safe


# ---------------------------
# EVALUATION CONFIG
# ---------------------------
MODEL_PATH = "/workspace/whisper-medium-librispeech/outputs_merged"
BASE_MODEL = "openai/whisper-medium"
DATASET_NAME = "librispeech_asr"
SUBSET = "clean"
SPLIT = "test"
MAX_SAMPLES = 200  # Set to None for the full split.
BATCH_SIZE = 8
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_NAME = None
LANGUAGE = "en"
TASK = "transcribe"
NORMALIZE_TEXT = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_DATASET_ROOT = Path("/workspace/datasets/librispeech_clean")
LOCAL_SPLIT_FOLDER = {
    "train.360": "train",
    "train.100": "train_100",
    "validation": "validation",
    "test": "test",
}


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model to evaluate.")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Name of the base model.")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--subset", type=str, default=SUBSET)
    parser.add_argument("--split", type=str, default=SPLIT)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--output_name", type=str, default=OUTPUT_NAME)
    parser.add_argument("--language", type=str, default=LANGUAGE)
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--normalize_text", type=bool, default=NORMALIZE_TEXT)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate the base model instead of the fine-tuned model.",
    )

    args = parser.parse_args()

    if args.base:
        args.model_path = args.base_model

    return args


def load_librispeech(args: argparse.Namespace) -> Dataset:
    """Load LibriSpeech from the local materialized cache when available."""
    if args.subset == "clean":
        local_folder = LOCAL_SPLIT_FOLDER.get(args.split, args.split.replace(".", "_"))
        local_path = LOCAL_DATASET_ROOT / local_folder
        if local_path.exists():
            print(f"Loading local LibriSpeech split: {local_path}")
            dataset = load_from_disk(str(local_path))
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            return dataset

    print(
        f"Loading HuggingFace dataset: {args.dataset_name}, "
        f"subset={args.subset}, split={args.split}"
    )
    dataset = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        streaming=args.max_samples is not None,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    if args.max_samples:
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) >= args.max_samples:
                break
        return Dataset.from_list(samples)

    return dataset


def load_model_and_processor(args: argparse.Namespace):
    """Load a merged Whisper checkpoint or a PEFT LoRA adapter checkpoint."""
    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model_path = Path(args.model_path)
    is_adapter = (model_path / "adapter_config.json").exists()

    processor_source = args.model_path if (model_path / "preprocessor_config.json").exists() else args.base_model
    processor = WhisperProcessor.from_pretrained(
        processor_source,
        language=args.language,
        task=args.task,
    )

    if is_adapter:
        if PeftModel is None:
            raise RuntimeError("This looks like a LoRA adapter, but peft is not installed.")
        print(f"Loading base model: {args.base_model}")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
        )
        print(f"Loading LoRA adapter: {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        print(f"Loading Whisper model: {args.model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
        )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device)
    model.eval()
    return model, processor, device, dtype


def normalize_text(text: str, enabled: bool) -> str:
    if not enabled:
        return text.strip()

    if BasicTextNormalizer is not None:
        return BasicTextNormalizer()(text).strip()

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def get_reference_text(sample: dict[str, Any]) -> str:
    return (
        sample.get("text")
        or sample.get("sentence")
        or sample.get("transcription")
        or ""
    )


def decode_audio(sample: dict[str, Any]) -> np.ndarray:
    decoded = _decode_audio_safe(sample)
    audio = decoded["audio"]
    return np.asarray(audio["array"], dtype=np.float32)


def batched(iterable: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield start, iterable[start : start + batch_size]


def transcribe_batch(
    model,
    processor: WhisperProcessor,
    device: torch.device,
    dtype: torch.dtype,
    audios: list[np.ndarray],
    language: str,
    task: str,
) -> list[str]:
    inputs = processor(
        audios,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    ).input_features.to(device=device, dtype=dtype)

    with torch.inference_mode():
        predicted_ids = model.generate(
            inputs,
            language=language,
            task=task,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


def safe_metric(metric_fn, reference: str, prediction: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return float(metric_fn(reference, prediction))


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_base" if getattr(args, "base", False) else ""
    stem = args.output_name or f"librispeech_{args.subset}_{args.split}{suffix}_wer_cer"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}_summary.json"


def main() -> None:
    args = get_config()
    normalize = args.normalize_text

    print("Evaluation config")
    print(f"Model path       : {args.model_path}")
    print(f"Dataset          : {args.dataset_name}/{args.subset}")
    print(f"Split            : {args.split}")
    print(f"Max samples      : {args.max_samples if args.max_samples else 'full split'}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Device           : {args.device}")

    dataset = load_librispeech(args)
    rows_source = list(dataset)
    if args.max_samples and len(rows_source) > args.max_samples:
        rows_source = rows_source[: args.max_samples]

    model, processor, device, dtype = load_model_and_processor(args)
    csv_path, summary_path = output_paths(args)

    rows = []
    references = []
    predictions = []

    progress = tqdm(
        total=len(rows_source),
        desc="Evaluating",
        unit="sample",
    )
    for batch_start, batch in batched(rows_source, args.batch_size):
        audios = [decode_audio(dict(sample)) for sample in batch]
        batch_predictions = transcribe_batch(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            audios=audios,
            language=args.language,
            task=args.task,
        )

        for offset, (sample, prediction_raw) in enumerate(zip(batch, batch_predictions)):
            index = batch_start + offset
            reference_raw = get_reference_text(sample)
            reference = normalize_text(reference_raw, normalize)
            prediction = normalize_text(prediction_raw, normalize)
            sample_wer = safe_metric(wer, reference, prediction)
            sample_cer = safe_metric(cer, reference, prediction)

            references.append(reference)
            predictions.append(prediction)
            rows.append(
                {
                    "index": index,
                    "id": sample.get("id", ""),
                    "subset": args.subset,
                    "split": args.split,
                    "reference": reference_raw,
                    "prediction": prediction_raw.strip(),
                    "normalized_reference": reference,
                    "normalized_prediction": prediction,
                    "wer": sample_wer,
                    "cer": sample_cer,
                    "wer_percent": sample_wer * 100,
                    "cer_percent": sample_cer * 100,
                }
            )
            progress.update(1)

    progress.close()

    mean_wer = float(wer(references, predictions)) if references else 0.0
    mean_cer = float(cer(references, predictions)) if references else 0.0
    per_sample_mean_wer = float(np.mean([row["wer"] for row in rows])) if rows else 0.0
    per_sample_mean_cer = float(np.mean([row["cer"] for row in rows])) if rows else 0.0

    fieldnames = [
        "index",
        "id",
        "subset",
        "split",
        "reference",
        "prediction",
        "normalized_reference",
        "normalized_prediction",
        "wer",
        "cer",
        "wer_percent",
        "cer_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "index": "__corpus__",
                "id": "",
                "subset": args.subset,
                "split": args.split,
                "reference": "",
                "prediction": "",
                "normalized_reference": "",
                "normalized_prediction": "",
                "wer": mean_wer,
                "cer": mean_cer,
                "wer_percent": mean_wer * 100,
                "cer_percent": mean_cer * 100,
            }
        )
        writer.writerow(
            {
                "index": "__mean_per_sample__",
                "id": "",
                "subset": args.subset,
                "split": args.split,
                "reference": "",
                "prediction": "",
                "normalized_reference": "",
                "normalized_prediction": "",
                "wer": per_sample_mean_wer,
                "cer": per_sample_mean_cer,
                "wer_percent": per_sample_mean_wer * 100,
                "cer_percent": per_sample_mean_cer * 100,
            }
        )

    summary = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "subset": args.subset,
        "split": args.split,
        "num_samples": len(rows),
        "text_normalized": normalize,
        "wer": mean_wer,
        "cer": mean_cer,
        "wer_percent": mean_wer * 100,
        "cer_percent": mean_cer * 100,
        "per_sample_mean_wer": per_sample_mean_wer,
        "per_sample_mean_cer": per_sample_mean_cer,
        "csv_path": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation complete")
    print(f"Samples          : {len(rows)}")
    print(f"WER              : {mean_wer * 100:.2f}%")
    print(f"CER              : {mean_cer * 100:.2f}%")
    print(f"Per-sample WER   : {per_sample_mean_wer * 100:.2f}%")
    print(f"Per-sample CER   : {per_sample_mean_cer * 100:.2f}%")
    print(f"Per-sample CSV   : {csv_path}")
    print(f"Summary JSON     : {summary_path}")


if __name__ == "__main__":
    main()
