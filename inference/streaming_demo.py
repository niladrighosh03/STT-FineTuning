#!/usr/bin/env python3
"""
Real-time STREAMING microphone STT demo.

Why streaming can sound "bad" vs the record-and-send demo:
  - Gradio streaming sends raw int16 PCM at the browser's native rate (usually
    48 kHz), in tiny chunks of ~0.1–0.3 s.
  - Whisper needs ≥ 1 s of audio to work reliably; shorter clips cause
    hallucinations ("Thank you.", ".", music tokens, etc.).
  - Solution: accumulate audio in a growing buffer and only call Whisper once
    we have enough data. We keep the buffer bounded at 30 s (Whisper's max).

Strategy
--------
  1. Each stream chunk → convert to float32 mono @ 16 kHz, append to buffer.
  2. Gate: only run Whisper when the buffer contains ≥ MIN_SECONDS of audio.
  3. Re-transcribe the WHOLE buffer every time (simple & most accurate).
     The buffer is capped at 30 s so Whisper runtime stays bounded.
  4. On Clear, reset everything.
"""

import math
import numpy as np
import torch
import gradio as gr
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "/workspace/whisper-medium-librispeech/outputs_merged"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_SR  = 16_000

# How many seconds to accumulate before first inference fires.
# Whisper is unreliable below ~1 s; 1.5 s gives much cleaner first results.
MIN_SECONDS = 1.5
MIN_SAMPLES = int(MIN_SECONDS * TARGET_SR)  # in target-SR samples

# Hard cap on buffer length (Whisper's max window is 30 s)
MAX_SAMPLES = 30 * TARGET_SR
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    print(f"📦  Loading model from {MODEL_PATH} …")
    proc = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    print(f"✅  Model ready on {DEVICE} ({DTYPE})")
    return model, proc


MODEL, PROCESSOR = load_model()


# ── audio helpers ─────────────────────────────────────────────────────────────

def _to_float32_mono_16k(raw: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert a raw audio chunk to float32 mono at 16 kHz.

    Gradio streaming typically delivers int16 PCM at 48 kHz (browser default).
    We must:
      1. Cast to float32
      2. Normalise int16 → [-1, 1]   (max of int16 is 32768)
      3. Mix down to mono if stereo
      4. Resample to 16 kHz
    """
    y = raw.astype(np.float32)

    # Normalise: int16 range is ±32768; float audio should be ±1
    # Guard: if already in [-1,1] this is a no-op
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / 32768.0

    # Mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample to TARGET_SR
    if sr != TARGET_SR and sr > 0:
        g = math.gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)

    return y


def _run_whisper(audio_np: np.ndarray) -> str:
    """Transcribe float32 mono 16 kHz audio with Whisper."""
    inputs = PROCESSOR(
        audio_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        ids = MODEL.generate(
            inputs,
            language="en",
            task="transcribe",
        )

    return PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ── state ─────────────────────────────────────────────────────────────────────

def _init_state() -> dict:
    return {
        "buffer": np.array([], dtype=np.float32),  # accumulated audio @ 16 kHz
        "last_sr": None,                            # sr of the first chunk seen
    }


# ── main streaming callback ───────────────────────────────────────────────────

def transcribe_stream(new_chunk, state):
    """
    Called by Gradio on every microphone chunk.

    Parameters
    ----------
    new_chunk : tuple (sample_rate: int, audio: np.ndarray)  or  None
    state     : dict from _init_state()

    Returns
    -------
    (text: str, updated_state: dict)
    """
    if state is None:
        state = _init_state()

    if new_chunk is None:
        return "", state

    sr, raw = new_chunk

    # Safety: empty chunk
    if raw is None or len(raw) == 0:
        return "", state

    # Debug log (visible in terminal) — helpful to confirm chunks are arriving
    print(f"  chunk sr={sr}  shape={raw.shape}  dtype={raw.dtype}  "
          f"peak={np.abs(raw).max():.1f}  buf={len(state['buffer'])/TARGET_SR:.2f}s")

    # Convert and append
    y = _to_float32_mono_16k(raw, sr)
    state["buffer"] = np.concatenate([state["buffer"], y])

    # Cap at 30 s (drop oldest)
    if len(state["buffer"]) > MAX_SAMPLES:
        state["buffer"] = state["buffer"][-MAX_SAMPLES:]

    # Gate: wait until we have enough audio
    if len(state["buffer"]) < MIN_SAMPLES:
        return "", state

    # Transcribe the whole buffer
    text = _run_whisper(state["buffer"])
    return text, state


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#output-box textarea {
    font-size: 1.4rem !important;
    line-height: 1.7 !important;
    border-color: #ff4b4b !important;
    border-width: 2px !important;
    min-height: 200px !important;
}
"""

with gr.Blocks(title="Whisper Live Streaming") as demo:
    gr.Markdown("# 🔴 Live Streaming STT", elem_id="title")
    gr.Markdown(
        f"Audio accumulates in a rolling **30-second** buffer.  \n"
        f"Transcription fires after the first **{MIN_SECONDS} s** of speech."
    )

    audio_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="🎙️ Speak here",
            )
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Live Transcription",
                lines=10,
                elem_id="output-box",
                interactive=False,
                placeholder="Start speaking — text appears after ~1.5 s …",
            )

    audio_in.stream(
        fn=transcribe_stream,
        inputs=[audio_in, audio_state],
        outputs=[output, audio_state],
        show_progress="hidden",
    )

    clear_btn.click(
        fn=lambda: (_init_state(), ""),
        inputs=None,
        outputs=[audio_state, output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
