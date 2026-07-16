"""
Encoder-based inference code for Stock TBSA (Target-Based Sentiment Analysis).

This code performs inference using a fine-tuned XLM-RoBERTa-Longformer or mmBERT model on a four-class TBSA test set
and saves inference time, a confusion matrix, and a classification report.
"""

# -----------------------------
# Imports
# -----------------------------
import os
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------

# Select the model:
#   "xlm_roberta_longformer"
#   "mmbert"
MODEL_KEY = "xlm_roberta_longformer"

MODEL_CONFIGS = {
    "xlm_roberta_longformer": {
        "model_path": "./output/xlm_roberta_longformer/best_checkpoint",
        "max_length": 4096,
    },
    "mmbert": {
        "model_path": "./output/mmbert/best_checkpoint",
        "max_length": 8192,
    },
}

MODEL_CONFIG = MODEL_CONFIGS[MODEL_KEY]
MODEL_PATH = MODEL_CONFIG["model_path"]
MAX_LENGTH = MODEL_CONFIG["max_length"]

TEST_JSON_PATH = (
    "./dataset/thai/Thai_test_4class.json"  # Path for Testing set (Thai or English)
)
SAVE_PATH = os.path.join("./output/inference_results", MODEL_KEY)

BATCH_SIZE = 8
WARMUP_SAMPLES = 32

# Sentiment label mappings
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2, "exclude": 3}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
LABELS_ORDER = ["positive", "neutral", "negative", "exclude"]


# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Selected model: {MODEL_KEY}")

if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# -----------------------------
# Load model and dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    model_max_length=MAX_LENGTH,
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()

df_test = pd.read_json(TEST_JSON_PATH, lines=True)


# -----------------------------
# Batch tokenization
# -----------------------------
def tokenize_batch(start_index):
    texts = df_test["Text"].iloc[start_index : start_index + BATCH_SIZE].tolist()

    tickers = df_test["TICKER"].iloc[start_index : start_index + BATCH_SIZE].tolist()

    return tokenizer(
        texts,
        text_pair=tickers,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)


# -----------------------------
# Warm-up
# -----------------------------
print("Running warm-up...")

with torch.no_grad():
    warmup_size = min(len(df_test), WARMUP_SAMPLES)

    for index in range(0, warmup_size, BATCH_SIZE):
        inputs = tokenize_batch(index)
        _ = model(**inputs)

if device.type == "cuda":
    torch.cuda.synchronize()

print("Warm-up completed.")


# -----------------------------
# Inference
# -----------------------------
print("Starting timed inference...")

predictions = []
inference_start_time = time.time()
inference_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with torch.no_grad():
    for index in range(0, len(df_test), BATCH_SIZE):
        inputs = tokenize_batch(index)
        outputs = model(**inputs)

        probabilities = softmax(outputs.logits, dim=1)
        batch_predictions = (
            torch.argmax(
                probabilities,
                dim=1,
            )
            .cpu()
            .tolist()
        )

        predictions.extend(batch_predictions)

if device.type == "cuda":
    torch.cuda.synchronize()

inference_end_time = time.time()
inference_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
inference_elapsed_time = inference_end_time - inference_start_time

print(f"Total Inference Time: {inference_elapsed_time:.2f} seconds")
print(
    f"Average Time per Sample: "
    f"{inference_elapsed_time / len(df_test):.4f} sec/sample"
)


# -----------------------------
# Save inference information
# -----------------------------
os.makedirs(SAVE_PATH, exist_ok=True)

timing_path = os.path.join(SAVE_PATH, "inference_time.txt")

with open(timing_path, "w", encoding="utf-8") as file:
    file.write(f"Model: {MODEL_KEY}\n")
    file.write(f"Checkpoint: {MODEL_PATH}\n")
    file.write(f"Start Time: {inference_start_datetime}\n")
    file.write(f"End Time: {inference_end_datetime}\n")
    file.write(
        f"Total Inference Time: {inference_elapsed_time:.2f} seconds "
        f"({inference_elapsed_time / 60:.2f} minutes)\n"
    )
    file.write(
        f"Time per Sample: " f"{inference_elapsed_time / len(df_test):.4f} sec/sample\n"
    )


# -----------------------------
# Evaluation
# -----------------------------
df_test["predicted_label"] = [ID2LABEL[prediction] for prediction in predictions]

conf_matrix = confusion_matrix(
    df_test["Sentiment_class"],
    df_test["predicted_label"],
    labels=LABELS_ORDER,
)

report = classification_report(
    df_test["Sentiment_class"],
    df_test["predicted_label"],
    labels=LABELS_ORDER,
    digits=4,
)


# -----------------------------
# Save confusion matrix
# -----------------------------
confusion_matrix_path = os.path.join(
    SAVE_PATH,
    "confusion_matrix.png",
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS_ORDER,
    yticklabels=LABELS_ORDER,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(confusion_matrix_path)
plt.close()


# -----------------------------
# Save classification report
# -----------------------------
report_path = os.path.join(
    SAVE_PATH,
    "classification_report.txt",
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print(f"Inference information saved to: {timing_path}")
print(f"Confusion matrix saved to: {confusion_matrix_path}")
print(f"Classification report saved to: {report_path}")
