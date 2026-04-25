# Data collator for Whisper seq2seq training
import torch
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DataCollatorSpeechSeq2Seq:
    """
    Collates a list of preprocessed examples into a padded batch.

    • input_features are already fixed-length (80 × 3000) mel spectrograms
      — we just stack them into a tensor.
    • labels (token ids) are variable-length — we pad to the longest
      sequence in the batch and replace padding with -100 so they are
      ignored by the loss.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ---- 1. Stack log-mel features ----
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # ---- 2. Pad label sequences ----
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If BOS token sits at the front of every label sequence, strip it
        # (Whisper uses it as decoder_start_token_id and adds it automatically)
        if (
            labels.shape[1] > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch