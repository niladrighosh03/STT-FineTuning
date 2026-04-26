#!/usr/bin/env python3
"""
Real-time STREAMING microphone STT demo.
The text updates LIVE as you speak.
"""

import math, io, numpy as np, torch, gradio as gr
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
    print(f"✅  Model ready on {DEVICE}")
    return model, proc

MODEL, PROCESSOR = load_model()

def transcribe_stream(new_chunk, state):
    """
    new_chunk: (sampling_rate, numpy_array)
    state: The accumulated audio numpy array
    """
    if new_chunk is None:
        return "", state

    sr, y = new_chunk
    
    # Convert to mono float32
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Normalize
    if np.abs(y).max() > 1.0:
        y = y / 32768.0

    # Resample if needed
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)

    # Append to state (accumulated audio)
    if state is None:
        state = y
    else:
        state = np.concatenate([state, y])

    # Only transcribe if we have enough audio (at least 0.5s)
    if len(state) < 8000:
        return "", state

    # To keep it fast, we only transcribe the last 30 seconds
    # (Whisper's maximum window size)
    max_samples = 30 * TARGET_SR
    input_audio = state[-max_samples:]

    inputs = PROCESSOR(
        input_audio,
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
    return text, state


CSS = """
#title { text-align: center; }
#output-box textarea {
    font-size: 1.5rem !important;
    line-height: 1.6 !important;
    border-color: #ff4b4b !important;
    border-width: 2px !important;
}
"""

with gr.Blocks(title="Whisper Live Streaming", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔴 Live Streaming STT", elem_id="title")
    gr.Markdown("Speak continuously and watch the transcription update in real-time.")

    # State to keep track of the audio buffer
    audio_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                sources=["microphone"],
                streaming=True, # <--- ENABLE STREAMING
                label="Speak here",
            )
            clear_btn = gr.Button("🗑️ Clear Transcription")

        with gr.Column():
            output = gr.Textbox(
                label="Live Transcription (Last 30s)",
                lines=8,
                elem_id="output-box",
                interactive=False
            )

    # Trigger transcription on every stream chunk
    audio_in.stream(
        fn=transcribe_stream,
        inputs=[audio_in, audio_state],
        outputs=[output, audio_state],
        show_progress="hidden"
    )

    # Clear everything
    clear_btn.click(lambda: (None, ""), None, [audio_state, output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
        css=CSS
    )
