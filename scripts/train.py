import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from src.dataset import load_data
from src.model import build_model   # 👈 CHANGED
from src.utils import preprocess_function
from src.collator import DataCollatorSpeechSeq2Seq
from src.metrics import compute_metrics
from src.trainer import get_trainer


def main():
    print("Loading config...", flush=True)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading datasets...", flush=True)
    dataset, val_dataset = load_data(config)

    print("Building model and processor...", flush=True)
    model, processor = build_model(config)   
    #test
    dataset = dataset.select(range(100))
    val_dataset = val_dataset.select(range(20))
    
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

    print("Starting training...", flush=True)
    trainer.train()

    trainer.save_model(config["training"]["output_dir"])
    print(f"Train samples: {len(dataset)}")
    print(f"Validation samples: {len(val_dataset)}")


if __name__ == "__main__":
    main()