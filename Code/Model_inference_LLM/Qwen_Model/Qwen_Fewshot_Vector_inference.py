"""
LLM-based few-shot batch inference script with pre-retrieved vector examples
for Stock TBSA (Target-Based Sentiment Analysis)

This script performs few-shot batch inference using a large language model
(LLM), such as Qwen2.5-72B-Instruct.

Similar examples are assumed to have already been retrieved from a vector
database and saved in a CSV file. The script combines these examples with
each target article, performs sentiment classification, and saves the
predicted labels and total inference time.
"""

# -----------------------------
#  Library imports
# -----------------------------
import ast
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
# - This is the long English prompt for few-shot inference.
# - A Thai version is available at: `Code/Prompt_Template/Zeroshot_LongPrompt_Thai.txt`
# - The few-shot examples were retrieved beforehand from a vector database.
# - The model must classify each target stock into one of four classes:
#   Positive, Negative, Neutral, or Exclude.

PROMPT_TEMPLATE = """I want you to act as a financial expert and NLP researcher in the field of data-centric research.
 
I want you to annotate stock sentiment for each stock TICKER that is mentioned in an input document.
Read the entire input document and assign final stock sentiment to each target stock TICKER.
-Sentiment must be determined solely based on the content of the given document.
-Respond only in Thai.
 
These are definitions for stock sentiment classes.
- Positive class = "The content of this news article has a positive impact on the target stock."
- Negative class = "The content of this news article has a negative impact on the target stock."
- Neutral class = "The content of this news article has neither a positive nor negative impact on the target stock."
- Exclude class = "The content of this news article does not fall into the above three classes or is unrelated to the target stock in terms of investment."
 
Annotation rules:
- If the news article includes stock direction analysis by analysts, assign the sentiment based on the analyst's opinion.
- If the company associated with the target stock engages in business expansion, increases production capacity, acquires other businesses, or signs cooperation agreements with other companies, label that stock as Positive.
- If the company associated with the target stock adjusts its policies to enhance business opportunities, label that stock as Positive.
- If the company associated with the target stock is involved in a lawsuit but wins the case without incurring damages, label that stock as Positive.
- If investors make additional investments or purchase the target stock for their portfolio, indicating that the stock is performing well, label that stock as Positive.
- If the target stock is included in the SET50 or SET100 index, indicating business growth, label that stock as Positive.
- If the target stock is involved in a lawsuit and loses the case, requiring payment or damages, label that stock as Negative.
- If the target stock is subject to financial audits or required to submit financial statements for review, label that stock as Negative.
- If investors sell the target stock from their portfolios, indicating potential issues, label that stock as Negative.
- If the target stock is placed under a cash balance restriction or has an extension of such a restriction, label that stock as Negative.
- If the stock exchange issues a trading alert for the target stock, label that stock as Negative.
- If a court or a stock exchange orders the company associated with the stock to provide clarification, indicating no wrongdoing, label that stock as Neutral.
- If a court announces an extension of the case deliberation period, indicating no wrongdoing, label that stock as Neutral.
- If the company associated with the stock receives a warning but no penalty, label that stock as Neutral.
- If the company associated with the stock files a lawsuit against another company, label the suing company as Neutral.
- Reports on opening or closing stock prices for a single day should be labeled as Neutral because single-day price changes do not indicate long-term stock trends.
- If the article discusses a stock split or reverse stock split, label that stock as Neutral.
- If the article mentions "buying pressure" or "selling pressure," label that stock as Neutral since these are short-term fluctuations, and the market may normalize the next day.
- If the article discusses "bond issuance," label the company associated with the stock as Neutral, unless the article states that the bond issuance is funding a new project, in which case label that stock as Positive.
- For banks or companies underwriting bonds for others, label the underwriters as Neutral.
- Articles about loan approvals should label the company receiving the loan as Positive, while labeling the approving bank as Neutral.
- Articles discussing adjustments to gasoline or gasohol prices should be labeled as Neutral.
- If a company or its stock is mentioned briefly in an article without positive or negative elaboration, label that stock as Neutral.
- Articles about donations, promotions, campaigns, annual or extraordinary general meetings, receiving or giving awards, recreational activities, non-investment-related activities, changes in executive positions (appointments, resignations, retirements), IPO launches, meeting venue changes, product launches as advertisements, or factory/company visits should be labeled as Exclude.
- Participating in exhibitions, showcasing innovations, launching new products, or releasing new packages is considered public relations and should be labeled as Exclude.
- Articles discussing overall market economic conditions without specific stock references should be labeled as Exclude.
- Articles unrelated to stock investments should be labeled as Exclude.
- Articles about application system maintenance should be labeled as Exclude.
- If the target stock is mentioned in the context of its first trading day, as this does not reflect long-term performance, label that stock as Exclude.
- Articles mentioning Biglot transactions should be labeled as Exclude.
- Articles discussing dividend payouts should be labeled as Exclude.
- Articles about the biography of company executives, unrelated to stock investments, should be labeled as Exclude.
- If the company associated with the target stock is acting as a stock analyst or reporting news about other stocks, label that stock as Exclude.

"""


# -----------------------------
# Configuration
# -----------------------------
# Replace these example values with the paths and server configuration
# in your environment.

RETRIEVED_DOCS_PATH = (
    "./path/to/retrieved_documents.csv"  # Replace with path of retrieved ICL examples
)
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
# Prompt construction helpers
# -----------------------------
def to_list_if_string(x):
    """Convert a stored list representation into a Python list."""

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        return ast.literal_eval(x)

    return []


def build_final_prompt(row):
    """Combine retrieved examples with the target TBSA instance."""

    examples = to_list_if_string(row["Vector_Example"])

    examples_text = "\n\n".join(str(example).strip() for example in examples)

    target_text = (
        "TARGET_ARTICLE: "
        + str(row["Text"]).strip()
        + "\n"
        + "TICKER: "
        + str(row["TICKER"]).strip()
        + "\n"
        + "SENTIMENT_CLASS: "
    )

    return examples_text + "\n\n" + target_text


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
# LLM initialization
# -----------------------------
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


# -----------------------------
# Output parsing
# -----------------------------
def parse_output(res):
    """Parse a structured LLM response or returned exception."""

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
        return {
            "sentiment": np.nan,
            "reason": str(e),
        }


# -----------------------------
# Batch inference
# -----------------------------
def run_batch_inference(
    df: pd.DataFrame,
    structured_llm: RunnableBinding,
    prompt_col: str = "final_prompt",
    batch_size: int = BATCH_SIZE,
    max_concurrency: int = MAX_CONCURRENCY,
):
    """Run few-shot LLM inference in batches."""

    outputs = []

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="Running batch inference",
    ):
        end = start + batch_size
        batch_df = df.iloc[start:end]

        prompts = batch_df[prompt_col].tolist()

        results = structured_llm.batch(
            prompts,
            config={"max_concurrency": max_concurrency},
            return_exceptions=True,
        )

        batch_outputs = [parse_output(res) for res in results]

        outputs.extend(batch_outputs)

    return outputs


# -----------------------------
# Load test data and retrieved examples
# -----------------------------
# Vector retrieval was performed beforehand using:
# "Code/Prepare_RetrievedDocuments/Prepare_RetrievedDocument_FromVectorRetriever.py"
#
# This script loads the pre-retrieved examples from the generated CSV file.

print("Loading retrieved examples...")

vector_retrieved_docs = pd.read_csv(
    RETRIEVED_DOCS_PATH,
)

print("Loading test set...")

df_test = pd.read_json(
    TEST_JSON_PATH,
    lines=True,
)


# -----------------------------
# Align retrieved examples with test instances
# -----------------------------
df_test = df_test.reset_index(drop=True)
vector_retrieved_docs = vector_retrieved_docs.reset_index(drop=True)

df_test["Vector_Example"] = vector_retrieved_docs["retrieved_document"]


# -----------------------------
# Construct inference prompts
# -----------------------------
print("Constructing few-shot prompts...")

df_test = df_test.copy()

df_test["input_prompt"] = df_test.apply(
    build_final_prompt,
    axis=1,
)

df_test["final_prompt"] = PROMPT_TEMPLATE + df_test["input_prompt"].astype(str)


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

df_test["AI"] = run_batch_inference(
    df=df_test,
    structured_llm=structured_llm,
    prompt_col="final_prompt",
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
    "a",
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
df_test["AI_sentiment"] = df_test["AI"].apply(
    lambda x: (x.get("sentiment") if isinstance(x, dict) else np.nan)
)

df_test["AI_reason"] = df_test["AI"].apply(
    lambda x: (x.get("reason") if isinstance(x, dict) else np.nan)
)


# -----------------------------
# Save inference results
# -----------------------------
df_test.to_json(
    SAVE_JSON_PATH,
    orient="records",
    lines=True,
    force_ascii=False,
)

print("Few-shot batch inference completed and results saved.")
