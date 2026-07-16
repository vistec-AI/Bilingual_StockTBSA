"""
Encoder-based fine-tuning code for Stock TBSA (Target-Based Sentiment Analysis).

This code fine-tunes XLM-RoBERTa-Longformer or mmBERT on a four-class
TBSA dataset using the Hugging Face Trainer API.
"""

# -----------------------------
# Imports
# -----------------------------
import json
import os
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# Configuration
# -----------------------------

# Select the model:
#   "xlm_roberta_longformer"
#   "mmbert"
MODEL_KEY = "xlm_roberta_longformer"

MODEL_CONFIGS = {
    "xlm_roberta_longformer": {
        "model_name": "markussagen/xlm-roberta-longformer-base-4096",
        "max_length": 4096,
        "output_dir": "./output/xlm_roberta_longformer",
        "wandb_project": "tbsa-xlm-roberta-longformer",
        "model_kwargs": {},
    },
    "mmbert": {
        "model_name": "jhu-clsp/mmBERT-base",
        "max_length": 8192,
        "output_dir": "./output/mmbert",
        "wandb_project": "tbsa-mmbert",
        "model_kwargs": {"reference_compile": False},
    },
}

MODEL_CONFIG = MODEL_CONFIGS[MODEL_KEY]
MODEL_NAME = MODEL_CONFIG["model_name"]
MAX_LENGTH = MODEL_CONFIG["max_length"]
OUTPUT_BASE_DIR = MODEL_CONFIG["output_dir"]
WANDB_PROJECT = MODEL_CONFIG["wandb_project"]
MODEL_KWARGS = MODEL_CONFIG["model_kwargs"]

# Path to dataset folder (e.g., "./dataset/thai", "./dataset/english", etc.)
DATA_DIR = "./dataset/thai"
TRAIN_FILE = "Thai_train_4class.json"  # Training dataset
VALIDATION_FILE = "Thai_validation_4class.json"  # Validation dataset

NUM_EPOCHS = 10
SEED = 42

# Sentiment label mappings
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2, "exclude": 3}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

REMOVE_COLUMNS = [
    "Article_ID",
    "Text",
    "TICKER",
    "Data-Source",
    "Date",
    "Year",
    "Sentiment_class",
]


# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Selected model: {MODEL_NAME}")

if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# -----------------------------
# Dataset loading
# -----------------------------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    data = pd.DataFrame(data)
    return Dataset.from_pandas(data)


# -----------------------------
# Evaluation metrics
# -----------------------------
def custom_metrics(eval_pred):
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")
    metric_accuracy = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "precision": metric_precision.compute(
            predictions=predictions,
            references=labels,
            average="micro",
        )["precision"],
        "recall": metric_recall.compute(
            predictions=predictions,
            references=labels,
            average="micro",
        )["recall"],
        "f1_micro": metric_f1.compute(
            predictions=predictions,
            references=labels,
            average="micro",
        )["f1"],
        "f1_macro": metric_f1.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )["f1"],
        "accuracy": metric_accuracy.compute(
            predictions=predictions,
            references=labels,
        )["accuracy"],
    }


# -----------------------------
# Model training
# -----------------------------
def train_model(batch_size, learning_rate):
    run_name = f"BS{batch_size}_LR{str(learning_rate).replace('.', '')}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, run_name)

    wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=MAX_LENGTH,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(
            examples["Text"],
            examples["TICKER"],
            truncation=True,
        )

    def create_label(examples):
        return [int(LABEL2ID[label]) for label in examples["Sentiment_class"]]

    train = load_dataset(os.path.join(DATA_DIR, TRAIN_FILE))
    validation = load_dataset(os.path.join(DATA_DIR, VALIDATION_FILE))

    datasets = DatasetDict(
        {
            "train": train,
            "validation": validation,
        }
    )

    datasets = datasets.map(
        preprocess_function,
        batched=True,
    )
    datasets = datasets.map(
        lambda examples: {"label": create_label(examples)},
        batched=True,
        remove_columns=REMOVE_COLUMNS,
    )

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
        report_to="wandb",
        seed=SEED,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        **MODEL_KWARGS,
    ).to(device)

    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=custom_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model and measure training time.
    train_start_time = time.time()
    train_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trainer.train()

    train_end_time = time.time()
    train_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_elapsed_time = train_end_time - train_start_time

    wandb.finish()

    # -----------------------------
    # Save training information
    # -----------------------------
    results_path = os.path.join(output_dir, "training_results.txt")

    with open(results_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {MODEL_NAME}\n")
        file.write("Training Time:\n")
        file.write(f"Start Time: {train_start_datetime}\n")
        file.write(f"End Time: {train_end_datetime}\n")
        file.write(
            f"Total Training Time: {train_elapsed_time:.2f} seconds "
            f"({train_elapsed_time / 60:.2f} minutes)\n\n"
        )

    print(f"Results saved in {results_path}")


# -----------------------------
# Hyperparameter search
# -----------------------------
if __name__ == "__main__":
    batch_sizes = [8, 16, 32]
    learning_rates = [3e-4, 3e-5, 3e-6, 4e-4, 4e-5, 4e-6, 5e-4, 5e-5, 5e-6]

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            train_model(batch_size, learning_rate)
