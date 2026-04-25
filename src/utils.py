# Preprocessing the dataset: audio → log-mel features, text → token ids

def preprocess_function(batch, processor):
    """
    Convert a single dataset example into Whisper model inputs.

    Args:
        batch: dict with keys 'audio' (datasets Audio dict) and a text key.
        processor: WhisperProcessor (feature_extractor + tokenizer).

    Returns:
        batch with added 'input_features' and 'labels' keys.
    """
    audio = batch["audio"]

    # --- Feature extraction (audio → 80-channel log-mel spectrogram) ---
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    # Shape: (1, 80, 3000)  →  squeeze to (80, 3000) so we can batch later
    batch["input_features"] = inputs.input_features[0]

    # --- Tokenise transcript ---
    # librispeech uses "text"; other datasets may use "sentence" or "transcription"
    text = (
        batch.get("text")
        or batch.get("sentence")
        or batch.get("transcription")
        or ""
    )
    batch["labels"] = processor.tokenizer(text).input_ids

    return batch