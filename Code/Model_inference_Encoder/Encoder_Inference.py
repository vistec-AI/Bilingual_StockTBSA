"""
Encoder-based inference script for Stock TBSA (Target-Based Sentiment Analysis)

This script performs inference using a fine-tuned XLM-RoBERTa-Longformer model
on a 4-class TBSA test set (Thai/English/Multilingual/Cross-lingual) and reports performance metrics.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import os
import time
import json
from typing import List

# -----------------------------
# Third-party imports
# -----------------------------
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# GPU Device (explicit placement)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Configuration
# -----------------------------
SAVE_PATH = "./output/inference_results/"
MODEL_PATH = "./output/thai_encoder/BS32_LR3e05/checkpoint-408"  # Update as needed (path of your best checkpoint model.)
TEST_JSON_PATH = "./dataset/thai/Thai_test_4class.json"  # Update as needed
BATCH_SIZE = 8

# Sentiment label mappings
LABEL2ID = {"neutral": 0, "positive": 1, "negative": 2, "exclude": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -----------------------------
# Load model and tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()

# -----------------------------
# Load test set
# -----------------------------
df_test = pd.read_json(TEST_JSON_PATH, lines=True)

# -----------------------------
# Warm-up
# -----------------------------
print("Running warm-up...")
with torch.no_grad():
    for i in range(0, min(len(df_test), 32), BATCH_SIZE):
        texts = df_test["Text"].iloc[i : i + BATCH_SIZE].tolist()
        tickers = df_test["TICKER"].iloc[i : i + BATCH_SIZE].tolist()
        inputs = tokenizer(
            texts, text_pair=tickers, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        _ = model(**inputs)
torch.cuda.synchronize()
print("Warm-up completed.")

# -----------------------------
# Inference
# -----------------------------
print("Starting timed inference...")
predictions: List[int] = []
start_time = time.time()

with torch.no_grad():
    for i in range(0, len(df_test), BATCH_SIZE):
        texts = df_test["Text"].iloc[i : i + BATCH_SIZE].tolist()
        tickers = df_test["TICKER"].iloc[i : i + BATCH_SIZE].tolist()
        inputs = tokenizer(
            texts, text_pair=tickers, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        batch_preds = torch.argmax(probs, dim=1).cpu().tolist()
        predictions.extend(batch_preds)

torch.cuda.synchronize()
end_time = time.time()

elapsed = end_time - start_time
print(f"Total Inference Time: {elapsed:.2f} seconds")
print(f"Avg Time per Sample: {elapsed / len(df_test):.4f} sec/sample")

# -----------------------------
# Save timing and predictions
# -----------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
with open(os.path.join(SAVE_PATH, "inference_time.txt"), "w") as f:
    f.write(f"Total Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
    f.write(f"Time per Sample: {elapsed / len(df_test):.4f} sec/sample\n")

# -----------------------------
# Confusion Matrix & Report
# -----------------------------
df_test["predicted_label"] = [ID2LABEL[p] for p in predictions]
labels_order = ["positive", "neutral", "negative", "exclude"]

conf_matrix = confusion_matrix(
    df_test["Sentiment_class"], df_test["predicted_label"], labels=labels_order
)

report = classification_report(
    df_test["Sentiment_class"],
    df_test["predicted_label"],
    labels=labels_order,
    digits=4,
)

# Save classification report only
os.makedirs(SAVE_PATH, exist_ok=True)
with open(os.path.join(SAVE_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# Optional: print confusion matrix as plain text (if needed)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification report saved.")
