"""
Encoder-based fine-tuning script for Stock TBSA (Target-Based Sentiment Analysis)

This script fine-tunes XLM-RoBERTa-Longformer on a 4-class TBSA dataset
(Thai/English/Multilingual) using the HuggingFace Trainer API.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import os
import json
import time
from typing import Dict, Any

# -----------------------------
# Third-party imports
# -----------------------------
import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# GPU Device (explicit placement)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Configuration
# -----------------------------

# Path to dataset folder (e.g., "./dataset/thai", "./dataset/english", etc.)
DATA_DIR = "./dataset/thai"

# Path to output folder where results and model will be saved
OUTPUT_BASE_DIR = "./output/thai_encoder"

# Model checkpoint
MODEL_NAME = "markussagen/xlm-roberta-longformer-base-4096"
NUM_EPOCHS = 10  # Number of training epochs
SEED = 42  # Random seed for reproducibility

# Sentiment label mappings
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2, "exclude": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# -----------------------------
# Utility functions
# -----------------------------
def load_dataset(path: str) -> Dataset:
    """Load JSON dataset into HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_pandas(pd.DataFrame(data))


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute classification metrics."""
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "precision": metric_precision.compute(
            predictions=preds, references=labels, average="micro"
        )["precision"],
        "recall": metric_recall.compute(
            predictions=preds, references=labels, average="micro"
        )["recall"],
        "f1_micro": metric_f1.compute(
            predictions=preds, references=labels, average="micro"
        )["f1"],
        "f1_macro": metric_f1.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"],
        "accuracy": metric_acc.compute(predictions=preds, references=labels)[
            "accuracy"
        ],
    }


# -----------------------------
# Main training function
# -----------------------------
def train_model(batch_size: int, learning_rate: float) -> None:
    """Train model with specified batch size and learning rate."""
    run_name = f"BS{batch_size}_LR{str(learning_rate).replace('.', '')}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer & collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=4096)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(examples):
        return tokenizer(examples["Text"], examples["TICKER"], truncation=True)

    def add_label(examples):
        return {"label": [LABEL2ID[label] for label in examples["Sentiment_class"]]}

    # NOTE:
    # The released dataset is provided as a single JSONL file in wide format.
    # You are expected to perform your own data splitting as needed.
    # For example, in our published experiments, we use a temporal split:
    #   - Train:      2018–2020
    #   - Validation: 2021
    #   - Test:       2022–2023
    # Below is just an example assuming you have already created split files.

    train_path = os.path.join(DATA_DIR, "Thai_train_4class.json")
    val_path = os.path.join(DATA_DIR, "Thai_val_4class.json")

    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)
    datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})

    datasets = datasets.map(preprocess, batched=True)
    datasets = datasets.map(
        add_label,
        batched=True,
        remove_columns=[
            "Article_ID",
            "Text",
            "TICKER",
            "Data-Source",
            "Date",
            "Year",
            "Sentiment_class",
        ],
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        report_to="none",  # Set to "wandb" if you integrate with Weights & Biases
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID)
    ).to(device)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"Starting training: {run_name}")
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    elapsed = end_time - start_time

    # Save training time
    result_path = os.path.join(output_dir, "training_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(
            f"Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n"
        )

    print(f" Training complete. Results saved to: {result_path}")


# NOTE:
# We track performance (loss/F1/etc.) using wandb during training.
# The best checkpoint is restored via `load_best_model_at_end=True` based on validation loss.
# No metrics are saved to disk directly in this script.
# You can manually select the best checkpoint (based on wandb logs) for downstream inference or analysis.

# -----------------------------
# Run all combinations
# -----------------------------
if __name__ == "__main__":
    batch_sizes = [8, 16, 32]
    learning_rates = [3e-4, 3e-5, 3e-6, 4e-4, 4e-5, 4e-6, 5e-4, 5e-5, 5e-6]

    for bs in batch_sizes:
        for lr in learning_rates:
            train_model(bs, lr)
