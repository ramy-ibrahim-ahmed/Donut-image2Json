from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from .dataloader import get_processed_dataset
from .model import load_model
from .config import *


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom trainer to handle an extra key in the model inputs.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        # The 'datasets' library might add extra columns, which we remove here
        # before passing them to the model.
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def main():
    # 1. Load and process the data
    processed_dataset, processor = get_processed_dataset()

    # 2. Load and configure the model
    model = load_model(processor)
    model.config.decoder.max_length = len(
        max(processed_dataset["train"]["labels"], key=len)
    )

    # 3. Define training arguments
    num_train_steps_per_epoch = len(processed_dataset["train"]) / (
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    total_train_steps = int(num_train_steps_per_epoch * NUM_TRAIN_EPOCHS)

    training_args = Seq2SeqTrainingArguments(
        output_dir=TRAINED_MODEL_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch_fused",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=int(WARMUP_RATIO * total_train_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=False,
        overwrite_output_dir=True,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        fp16=True if DEVICE == "cuda" else False,
    )

    # 4. Initialize the Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    # 5. Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 6. Save the final model and processor
    print(f"Saving model and processor to {TRAINED_MODEL_DIR}")
    model.save_pretrained(TRAINED_MODEL_DIR)
    processor.save_pretrained(TRAINED_MODEL_DIR)


if __name__ == "__main__":
    main()
