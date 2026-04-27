import base64
import io
import math
import os
from threading import Thread
from threading import Lock

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_SR = 16000
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"


app = FastAPI(title="Whisper STT Azure-Like API", version="1.0.0")

_model = None
_processor = None
_model_ready = False
_load_error = None
_load_lock = Lock()


class TranscribeRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio bytes")
    language: str = Field(default="en", description="Whisper language code")


def _load_model_once():
    global _model, _processor, _model_ready, _load_error

    if _model_ready:
        return

    with _load_lock:
        if _model_ready:
            return

        try:
            if not MODEL_PATH:
                raise RuntimeError("MODEL_PATH environment variable is required")
            resolved_model_path = _resolve_model_path(MODEL_PATH)
            print(f"Loading model from {resolved_model_path}")
            _processor = WhisperProcessor.from_pretrained(resolved_model_path)
            _model = WhisperForConditionalGeneration.from_pretrained(
                resolved_model_path,
                torch_dtype=DTYPE,
            ).to(DEVICE)
            _model.eval()
            _model_ready = True
            print(f"Model ready on {DEVICE} with dtype={DTYPE}")
        except Exception as exc:
            _load_error = str(exc)
            print(f"Model load failed: {_load_error}")
            raise


def _resolve_model_path(model_path: str) -> str:
    if "/" not in model_path:
        return model_path

    org, name = model_path.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = sorted(
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        )
        if versions:
            return os.path.join(snapshots_dir, versions[-1])

    return model_path


def _decode_audio(audio_base64: str) -> np.ndarray:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc

    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode audio. Prefer wav/flac/mp3 supported by libsndfile/ffmpeg-compatible build. Error: {exc}",
        ) from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        gcd = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd).astype(np.float32)

    return audio


def _transcribe(audio: np.ndarray, language: str) -> str:
    inputs = _processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        generated_ids = _model.generate(
            inputs,
            language=language,
            task="transcribe",
        )

    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text


def _build_azure_like_response(text: str) -> dict:
    return {
        "RecognitionStatus": "Success",
        "DisplayText": text,
        "Offset": 0,
        "Duration": 0,
        "NBest": [
            {
                "Confidence": None,
                "Lexical": text.lower(),
                "ITN": text,
                "MaskedITN": text,
                "Display": text,
            }
        ],
    }


@app.on_event("startup")
def _startup():
    Thread(target=_load_model_once, daemon=True).start()


@app.get("/ping")
def ping():
    if _model_ready:
        return {"status": "healthy"}
    if _load_error:
        raise HTTPException(status_code=500, detail=_load_error)
    return Response(status_code=204)


@app.post("/transcribe")
def transcribe(request: TranscribeRequest):
    if not _model_ready:
        _load_model_once()

    audio = _decode_audio(request.audio_base64)
    text = _transcribe(audio, request.language)
    return _build_azure_like_response(text)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
