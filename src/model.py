import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_model(config):
    """
    Load Whisper model + processor with proper settings
    """

    model_name = config["model_name"]

    # Load processor (tokenizer + feature extractor)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="en",
        task="transcribe"
    )
    print("Loading model....")

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # IMPORTANT SETTINGS
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Gradient checkpointing (VERY IMPORTANT for medium/large)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("Model loaded successfully!") 

    return model, processor


# -----------------------------
# 🔥 LoRA (PEFT) SUPPORT
# -----------------------------

def apply_lora(model, config):
    """
    Apply LoRA to Whisper model
    """
    if not config.get("use_lora", False):
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    print("\n✅ LoRA applied successfully!")
    model.print_trainable_parameters()

    return model


# -----------------------------
# ❄️ Freeze base model (optional)
# -----------------------------

def freeze_base_model(model, config):
    """
    Freeze base Whisper weights (useful with LoRA)
    """
    if not config.get("freeze_base_model", False):
        return model

    for param in model.model.parameters():
        param.requires_grad = False

    print("\n❄️ Base model frozen (only LoRA layers will train)")
    return model


# -----------------------------
# ⚡ Optional: torch.compile
# -----------------------------

def compile_model(model, config):
    """
    Compile model for speed (PyTorch 2.x)
    """
    if not config.get("torch_compile", False):
        return model

    model = torch.compile(model)
    print("\n⚡ Model compiled with torch.compile()")

    return model


# -----------------------------
# 🎯 MAIN BUILDER FUNCTION
# -----------------------------

def build_model(config):
    """
    Full pipeline:
    load → LoRA → freeze → compile
    """

    model, processor = load_model(config)

    # Apply LoRA if enabled
    model = apply_lora(model, config)

    # Freeze base model if enabled
    model = freeze_base_model(model, config)

    # Compile (optional)
    model = compile_model(model, config)

    return model, processor