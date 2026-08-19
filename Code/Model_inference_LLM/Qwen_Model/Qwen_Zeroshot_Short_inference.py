"""
LLM-based zero-shot batch inference script for Stock TBSA
(Target-Based Sentiment Analysis)

This script performs zero-shot inference in batches using a large language
model (LLM), such as Qwen2.5-72B-Instruct, on a Stock TBSA test set.

The script outputs predicted sentiment labels and records the total
inference time.
"""

# -----------------------------
# Library imports
# -----------------------------
import time
from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI

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
INFERENCE_TIME_LOG = "./path/to/inference_timing.txt"  # Inference time log

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # you can change to other LLMs
API_KEY = "EMPTY"  # Replace with your actual API key
API_BASE = "http://your-vllm-server:8000/v1"  # Replace with your API base
TEMPERATURE = 0.0

BATCH_SIZE = 50
MAX_CONCURRENCY = 8


# -----------------------------
# Structured output schema
# -----------------------------
class Sentiment(BaseModel):
    sentiment: Literal[
        "negative",
        "neutral",
        "positive",
        "exclude",
    ]


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
    api_base: str = API_BASE,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> RunnableBinding:
    """Create an LLM binding with structured sentiment output."""

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model_name=model_name,
        temperature=temperature,
    )

    return llm.with_structured_output(Sentiment)


def parse_output(res):
    """Parse a structured LLM response or a returned exception."""

    try:
        if isinstance(res, Exception):
            return {
                "sentiment": np.nan,
                "reason": str(res),
            }

        sentiment = res.sentiment

        if sentiment not in {
            "negative",
            "neutral",
            "positive",
            "exclude",
        }:
            return {
                "sentiment": np.nan,
                "reason": f"invalid sentiment: {sentiment}",
            }

        return {
            "sentiment": sentiment,
            "reason": np.nan,
        }

    except Exception as e:
        print("Parse error:", e)

        return {
            "sentiment": np.nan,
            "reason": str(e),
        }


def run_batch_inference(
    df: pd.DataFrame,
    structured_llm: RunnableBinding,
    batch_size: int = BATCH_SIZE,
    max_concurrency: int = MAX_CONCURRENCY,
):
    """Run zero-shot LLM inference in batches."""

    outputs = []

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="Running batch inference",
    ):
        end = start + batch_size
        batch_df = df.iloc[start:end]

        prompts = [
            create_prompt(row["Text"], row["TICKER"]) for _, row in batch_df.iterrows()
        ]

        results = structured_llm.batch(
            prompts,
            config={"max_concurrency": max_concurrency},
            return_exceptions=True,
        )

        batch_outputs = [parse_output(res) for res in results]
        outputs.extend(batch_outputs)

    return outputs


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
# Run batch inference
# -----------------------------
print("Starting batch inference...")

inference_start_time = time.time()

df["AI"] = run_batch_inference(
    df=df,
    structured_llm=structured_llm,
    batch_size=BATCH_SIZE,
    max_concurrency=MAX_CONCURRENCY,
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
# Extract output fields
# -----------------------------
df[["AI_sentiment", "AI_reason"]] = df["AI"].apply(
    lambda x: (
        pd.Series(
            {
                "AI_sentiment": (
                    x["sentiment"].capitalize()
                    if isinstance(x, dict) and pd.notna(x["sentiment"])
                    else np.nan
                ),
                "AI_reason": (
                    x["reason"]
                    if isinstance(x, dict) and pd.notna(x["reason"])
                    else np.nan
                ),
            }
        )
        if isinstance(x, dict)
        else pd.Series(
            {
                "AI_sentiment": np.nan,
                "AI_reason": np.nan,
            }
        )
    )
)


# -----------------------------
# Save output file
# -----------------------------
df.to_json(
    SAVE_JSON_PATH,
    orient="records",
    lines=True,
)

print("Batch inference completed and results saved.")
