import time
import logging
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def setup_logger(log_file="output.log"):
    """
    Setup logging to file + console
    """
    logger = logging.getLogger("whisper_training")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_trainer(config, model, processor, train_dataset, val_dataset, data_collator, compute_metrics):

    import time

    log_file = f"output_{int(time.time())}.log"
    logger = setup_logger(log_file)

    args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],

        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["eval_batch_size"],

        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_epochs"],
        warmup_steps=config["training"]["warmup_steps"],

        logging_steps=config["training"]["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],

        predict_with_generate=True,

        # 🔥 Precision (H200 optimized)
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        report_to="tensorboard",    
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, processor),
    )

    # -----------------------------
    # ⏱️ Wrap training with timer
    # -----------------------------
    original_train = trainer.train

    def timed_train(*args, **kwargs):
        logger.info("🚀 Training started...")

        start_time = time.time()

        result = original_train(*args, **kwargs)

        end_time = time.time()
        total_time = end_time - start_time

        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60

        logger.info(
            f"⏱️ Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )

        return result

    trainer.train = timed_train

    return trainer