#data collator
from dataclasses import dataclass

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: any

    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch