from datasets import load_dataset

def load_data(config):
    dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"]["subset"],
        split=config["dataset"]["train_split"]
    )

    val_dataset = load_dataset(
        config["dataset"]["name"],
        config["dataset"]["subset"],
        split=config["dataset"]["val_split"]
    )

    return dataset, val_dataset