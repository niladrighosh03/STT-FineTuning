#model loader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def load_model(config):
    processor = WhisperProcessor.from_pretrained(
        config["model_name"],
        language="en",
        task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained(config["model_name"])

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return model, processor