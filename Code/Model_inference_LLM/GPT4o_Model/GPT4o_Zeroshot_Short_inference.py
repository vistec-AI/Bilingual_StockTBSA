"""
LLM-based zero-shot inference script for Stock TBSA
(Target-Based Sentiment Analysis)

This script performs zero-shot inference using GPT-4o on a Stock TBSA
test set.

The script outputs predicted sentiment labels and records the total
inference time.
"""

# -----------------------------
# Library imports
# -----------------------------
import time
from typing import List
from enum import Enum, EnumMeta
import numpy as np
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from tqdm import tqdm

tqdm.pandas()


# -----------------------------
# Prompt template
# -----------------------------
# NOTE:
# - This is the short prompt version in English.
# - A Thai version is available at: `Code/Prompt_Template/Zeroshot_ShortPrompt_Thai.txt`
# - The model must classify each target stock into one of four classes:
#   Positive, Negative, Neutral, or Exclude.

PROMPT_TEMPLATE = """I want you to act as a financial expert and NLP researcher in the field of data-centric research.
 
I want you to annotate stock sentiment for each stock TICKER that is mentioned in an input document.
Read the entire input document and assign final stock sentiment to each target stock TICKER.
-Sentiment must be determined solely based on the content of the given document.
-Respond only in English.
 
These are definitions for stock sentiment classes.
- Positive class = "The content of this news article has a positive impact on the target stock."
- Negative class = "The content of this news article has a negative impact on the target stock."
- Neutral class = "The content of this news article has neither a positive nor negative impact on the target stock."
- Exclude class = "The content of this news article does not fall into the above three classes or is unrelated to the target stock in terms of investment."

TARGET_ARTICLE: {doc}
TICKER: {tags}
SENTIMENT_CLASS: 
"""

# -----------------------------
# Configuration
# -----------------------------
# Replace these example paths with the paths in your environment.

TEST_JSON_PATH = "./path/to/test_data.json"  # Replace with your test set path
SAVE_JSON_PATH = "./path/to/output_predictions.jsonl"  # Predicion output
SAVE_BEFORE_SPLIT_PATH = "./path/to/output_predictions_before_split.jsonl"
INFERENCE_TIME_LOG = "./path/to/inference_timing.txt"  # Inference time log

MODEL_NAME = "gpt-4o-2024-08-06"
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key

TEMPERATURE = 0.0


# -----------------------------
# Structured output schema
# -----------------------------
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


# -----------------------------
# Helper functions
# -----------------------------
def create_prompt(doc: str, tags: str) -> str:
    """Create a zero-shot TBSA prompt for one target instance."""

    return PROMPT_TEMPLATE.format(
        doc=doc,
        tags=tags,
    )


def create_binding(
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> RunnableBinding:
    """Create an LLM binding with structured sentiment output."""

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    return llm.with_structured_output(Sentiment)


def create_output(
    structured_llm: RunnableBinding,
    prompt: str,
):
    """Run zero-shot inference for one target instance."""

    try:
        res = structured_llm.invoke(prompt)

        sentiment = res.sentiment[0].value

        return {
            "sentiment": sentiment,
        }

    except Exception as e:
        print("Inference error:", e)

        return np.nan


# -----------------------------
# Load test set
# -----------------------------
print("Loading test set...")

df = pd.read_json(
    TEST_JSON_PATH,
    lines=True,
)


# -----------------------------
# Prepare LLM binding
# -----------------------------
print("Preparing LLM...")

structured_llm = create_binding()


# -----------------------------
# Run zero-shot inference
# -----------------------------
print("Starting zero-shot inference...")

inference_start_time = time.time()

df["AI"] = df.progress_apply(
    lambda row: create_output(
        structured_llm,
        create_prompt(
            row["Text"],
            row["TICKER"],
        ),
    ),
    axis=1,
)

inference_end_time = time.time()
inference_elapsed_time = inference_end_time - inference_start_time


# -----------------------------
# Save inference timing
# -----------------------------
with open(
    INFERENCE_TIME_LOG,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        f"Total Inference Time: "
        f"{inference_elapsed_time:.2f} seconds "
        f"({inference_elapsed_time / 60:.2f} minutes)\n\n"
    )


# -----------------------------
# Save intermediate output
# -----------------------------
df.to_json(
    SAVE_BEFORE_SPLIT_PATH,
    orient="records",
    lines=True,
)


# -----------------------------
# Extract output fields
# -----------------------------
df["AI_sentiment"] = (
    df["AI"]
    .apply(lambda x: (x.get("sentiment") if isinstance(x, dict) else np.nan))
    .str.capitalize()
)


# -----------------------------
# Save final output
# -----------------------------
df.to_json(
    SAVE_JSON_PATH,
    orient="records",
    lines=True,
)

# -----------------------------
# Finish
# -----------------------------
print("Zero-shot inference completed and results saved.")
