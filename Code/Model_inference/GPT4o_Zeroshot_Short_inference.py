"""
LLM-based zero-shot inference script for Stock TBSA (Target-Based Sentiment Analysis)

This script performs zero-shot inference using a large language model (LLM)
(e.g., GPT-4o) on a TBSA test set and outputs predicted sentiment labels.
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum, EnumMeta
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI

# -----------------------------
# Prompt template and label enums
# -----------------------------
# 🔹NOTE:
# - This is the short prompt version in English used for inference.
# - A Thai version is available at: `Code/Examples_PromptTemplate/Zeroshot_ShortPrompt_Thai.txt`

PROMPT_TEMPLATE = """I want you to act as a financial expert and NLP researcher in the field of data-centric research.

I want you to annotate stock sentiment for each stock TICKER that is mentioned in an input document.
Read the entire input document and assign final stock sentiment to each target stock TICKER.
- Sentiment must be determined solely based on the content of the given document.
- Respond only in English.

These are definitions for stock sentiment classes:
- Positive = "The content of this news article has a positive impact on the target stock."
- Negative = "The content of this news article has a negative impact on the target stock."
- Neutral = "The content of this news article has neither a positive nor negative impact on the target stock."
- Exclude = "The content of this news article does not fall into the above three classes or is unrelated to the target stock in terms of investment."

TARGET_ARTICLE: {doc}
TICKER: {tags}
SENTIMENT_CLASS: 
"""

# -----------------------------
# Configuration (replace with your actual paths)
# -----------------------------
TEST_JSON_PATH = "./path/to/test_set.json"  # <-- Replace with your test set path
OUTPUT_JSON_PATH = "./path/to/output_predictions.json"  # <-- Output predictions
INFERENCE_TIME_LOG = "./path/to/inference_timing.txt"  # <-- Output timing info

MODEL_NAME = "gpt-4o-2024-08-06"
API_KEY = "EMPTY"  # <-- Replace with your actual API key
TEMPERATURE = 0.0


class EnumDirectValueMeta(EnumMeta):
    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class SentimentType(Enum, metaclass=EnumDirectValueMeta):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    EXCLUDE = "exclude"


class Sentiment(BaseModel):
    sentiment: List[SentimentType]
    # reason: Optional[str] = Field(...)  # Optional if explanation is desired


# -----------------------------
# Helper functions
# -----------------------------
def create_prompt(doc: str, tags: str) -> str:
    return PROMPT_TEMPLATE.format(doc=doc, tags=tags)


def create_binding(
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> RunnableBinding:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )
    return llm.with_structured_output(Sentiment)


def create_output(structured_llm: RunnableBinding, prompt: str) -> dict[str, str]:
    try:
        res = structured_llm.invoke(prompt)
        sentiment = res.sentiment[0].value
        return dict(sentiment=sentiment, reason=np.nan)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return np.nan


# -----------------------------
# Load test set
# -----------------------------
print("Loading test set...")
df = pd.read_json(TEST_JSON_PATH, lines=True)  # Import testing set

# -----------------------------
# Prepare LLM and Inference
# -----------------------------
print("Initializing GPT-4o model for inference...")
structured_llm = create_binding()
tqdm.pandas()


start_time = time.time()

df["AI"] = df.progress_apply(
    lambda x: create_output(structured_llm, create_prompt(x["Text"], x["TICKER"])),
    axis=1,
)

end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# Save timing info
# -----------------------------
with open(INFERENCE_TIME_LOG, "w") as f:
    f.write(
        f"Total Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

# -----------------------------
# Extract predicted sentiment
# -----------------------------
# Handle cases with NaN values in dictionaries
df["AI_sentiment"] = (
    df["AI"]
    .apply(lambda x: x.get("sentiment") if isinstance(x, dict) else np.nan)
    .str.capitalize()
)

# -----------------------------
# Save output predictions
# -----------------------------
df.to_json(OUTPUT_JSON_PATH, orient="records", lines=True)

print("LLM inference completed and results saved.")
