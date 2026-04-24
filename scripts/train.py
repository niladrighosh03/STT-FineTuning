import yaml
from src.dataset import load_data
from src.model import load_model
from src.utils import preprocess_function
from src.collator import DataCollatorSpeechSeq2Seq
from src.metrics import compute_metrics
from src.trainer import get_trainer

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    dataset, val_dataset = load_data(config)

    model, processor = load_model(config)

    dataset = dataset.map(lambda x: preprocess_function(x, processor))
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, processor))

    data_collator = DataCollatorSpeechSeq2Seq(processor)

    trainer = get_trainer(
        config,
        model,
        processor,
        dataset,
        val_dataset,
        data_collator,
        compute_metrics
    )

    trainer.train()

    trainer.save_model(config["training"]["output_dir"])

if __name__ == "__main__":
    main()