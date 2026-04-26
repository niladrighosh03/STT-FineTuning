import os, io, math, argparse, torch
import numpy as np
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = "/workspace/whisper-medium-librispeech/outputs_merged"
REPO_ID = "niladrighosh033/whisper-medium-librispeech" 
TEST_AUDIO_PATH = "inference/test_voice.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Audio loading helpers (from inference.py)
# ──────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int = 16_000) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    from scipy.signal import resample_poly
    g = math.gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)

def load_audio_file(path: str) -> np.ndarray:
    """Returns mono float32 numpy array at 16 kHz."""
    try:
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        return _resample(audio, sr)
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=16_000, mono=True)
        return audio

def upload_to_hf():
    print(f"📦 Loading local model from: {LOCAL_MODEL_PATH}")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Error: {LOCAL_MODEL_PATH} not found!")
        return

    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH)

    print(f"🚀 Uploading to Hugging Face: https://huggingface.co/{REPO_ID}")
    model.push_to_hub(REPO_ID)
    processor.push_to_hub(REPO_ID)
    print(f"✨ SUCCESS! Model is now live on HF.")



def test_inference_from_hf():
    print(f"🌐 Loading model DIRECTLY from Hugging Face: {REPO_ID}...")
    
    # Load from HF Hub
    processor = WhisperProcessor.from_pretrained(REPO_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        REPO_ID, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    print(f"✅ Model loaded from cloud!")
    print(f"🎵 Transcribing local test file: {TEST_AUDIO_PATH}")

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"❌ Error: {TEST_AUDIO_PATH} not found!")
        return

    audio = load_audio_file(TEST_AUDIO_PATH)

    # Process and Generate
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, dtype=model.dtype)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="en", task="transcribe")
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n" + "="*50)
    print(f"🤖 Prediction : {transcription.strip()}")
    print("="*50)
    print("✅ Remote inference test successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Model Manager")
    parser.add_argument("--upload", action="store_true", help="Upload local model to HF")
    parser.add_argument("--inference", action="store_true", help="Test inference by downloading from HF")
    
    args = parser.parse_args()

    if args.upload:
        upload_to_hf()
    elif args.inference:
        test_inference_from_hf()
    else:
        parser.print_help()
