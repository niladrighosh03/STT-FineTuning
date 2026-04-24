from datasets import load_dataset, Audio

def load_data(config):
    dataset_name = config["dataset"]["name"]
    subset = config["dataset"].get("subset", "all")
    train_split = config["dataset"]["train_split"]
    val_split = config["dataset"]["val_split"]

    print(f"Loading '{dataset_name}' (subset: {subset}) - Train: {train_split}, Val: {val_split}...", flush=True)

    # Load datasets using streaming to avoid disk quota issues
    train_stream = load_dataset(dataset_name, subset, split=train_split, streaming=True)
    val_stream = load_dataset(dataset_name, subset, split=val_split, streaming=True)

    # Materialize a small subset since we are hitting disk quota limits for the full dataset 
    from datasets import Dataset
    train_dataset = Dataset.from_list(list(train_stream.take(150)))
    val_dataset = Dataset.from_list(list(val_stream.take(50)))

    # Ensure the sampling rate is 16kHz for Whisper
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset, val_dataset
