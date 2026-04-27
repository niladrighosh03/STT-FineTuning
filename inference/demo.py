#!/usr/bin/env python3
"""
Real-time microphone STT demo using the fine-tuned Whisper Medium model.

Run:
    .venv/bin/python demo.py

Then open the URL printed in the terminal in your browser.
Click the microphone button, speak, stop — and see the transcription.
"""

import math, io, numpy as np, torch, soundfile as sf, gradio as gr
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "/workspace/whisper-medium-librispeech/outputs_merged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_SR = 16_000
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    print(f"📦  Loading model from {MODEL_PATH} …")
    proc  = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    print(f"✅  Model ready on {DEVICE} ({DTYPE})")
    return model, proc


def resample(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(src_sr, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, src_sr // g).astype(np.float32)


# Load once at startup
MODEL, PROCESSOR = load_model()


def transcribe(audio_tuple):
    """
    Gradio passes audio as (sample_rate, numpy_array).
    Returns the transcription string.
    """
    if audio_tuple is None:
        return "⚠️  No audio received. Please record something first."

    sr, audio = audio_tuple

    # Ensure mono float32
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Normalize if int (Gradio sometimes gives int16)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    # Resample to 16 kHz
    audio = resample(audio, sr)

    if len(audio) < 400:
        return "⚠️  Recording too short. Please speak for at least a second."

    # Compute features and generate
    inputs = PROCESSOR(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        ids = MODEL.generate(
            inputs,
            language="en",
            task="transcribe",
        )

    text = PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text if text else "(no speech detected)"


# ──────────────────────────────────────────────────────────────────────────────
# Build the Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #888; margin-bottom: 1.5rem; }
#output-box textarea {
    font-size: 1.3rem !important;
    min-height: 120px !important;
    border-radius: 12px !important;
}
.record-btn { font-size: 1.1rem !important; }
"""

with gr.Blocks(title="Whisper Medium STT Demo") as demo:

    gr.Markdown("# 🎤 Whisper Medium — Fine-tuned STT Demo", elem_id="title")
    gr.Markdown(
        "Record your voice → model transcribes it in real time.  \n"
        "Model: **whisper-medium** fine-tuned on LibriSpeech `train.360` (20 k samples, 5 epochs, LoRA r=32).",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎙️  Your voice",
            )
            transcribe_btn = gr.Button(
                "🚀  Transcribe", variant="primary", elem_classes="record-btn"
            )

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📝  Transcription",
                placeholder="Transcription will appear here …",
                lines=5,
                elem_id="output-box",
            )

    # Wire up: button click OR audio submit both trigger transcription
    transcribe_btn.click(fn=transcribe, inputs=audio_in, outputs=output)
    audio_in.stop_recording(fn=transcribe, inputs=audio_in, outputs=output)

    gr.Markdown(
        "---\n"
        "**Tips:**  \n"
        "• Click the red ● button to record, click again (■) to stop — transcription runs automatically.  \n"
        "• For best accuracy speak clearly in English.  \n"
        "• You can also upload a `.wav` / `.flac` file using the upload icon.  \n"
        f"• Running on **{DEVICE.upper()}** with {DTYPE}.",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # listen on all interfaces (required for RunPod)
        server_port=None,
        share=True,              # set True to get a public gradio.live link
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS
    )
