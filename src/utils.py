#preprocessing the dataset
def preprocess_function(batch, processor):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )

    labels = processor.tokenizer(batch["text"]).input_ids

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels

    return batch``