"""
LLM-based zero-shot inference script for Stock TBSA
(Target-Based Sentiment Analysis)

This script performs zero-shot inference using GPT-4o on a Stock TBSA
test set.

Few-shot examples are randomly selected and prepared beforehand. The generated
prompts are saved in a JSONL file. This script loads the prepared prompts,
performs sentiment classification, and saves the predicted labels and total
inference time.
"""

# -----------------------------
# Library imports
# -----------------------------
import json
import time
from enum import Enum, EnumMeta
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBinding

# -----------------------------
# Prompt template
# -----------------------------
# NOTE:
# - This is the long English prompt for few-shot inference.
# - A Thai version is available at: `Code/Prompt_Template/Zeroshot_LongPrompt_Thai.txt`
# - The few-shot examples were retrieved beforehand using BM25 .
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
# Replace these example paths with the paths in your environment.

RANDOM_PROMPTS_PATH = "./path/to/random_fewshot_prompts.jsonl"  # Replace with path of retrieved ICL examples
SAVE_JSON_PATH = "./path/to/output_predictions.jsonl"  # Predicion output
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
# LLM initialization
# -----------------------------
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


# -----------------------------
# Output generation
# -----------------------------
def create_output(
    structured_llm: RunnableBinding,
    prompt: str,
) -> dict[str, str]:
    """
    Run one prompt and return the predicted sentiment.

    Inference is intentionally performed one row at a time
    to preserve the original experimental procedure.
    """

    try:
        res = structured_llm.invoke(prompt)

        sentiment = res.sentiment[0].value

        return dict(
            sentiment=sentiment,
        )

    except Exception as e:
        print(e)

        return np.nan


# -----------------------------
# Load pre-generated random prompts
# -----------------------------
# Random example selection and prompt preparation were performed beforehand.
#
# This script loads the generated CSV file, which contains the prepared
# few-shot prompts in the "input_prompt" column.

print("Loading pre-generated random few-shot prompts...")

df_retrieved_docs = pd.read_csv(
    RANDOM_PROMPTS_PATH,
)


# -----------------------------
# Construct inference prompts
# -----------------------------
print("Constructing few-shot prompts...")

prompts = df_retrieved_docs["input_prompt"].tolist()

final_prompt = [PROMPT_TEMPLATE + item for item in prompts]


# -----------------------------
# Prepare LLM binding
# -----------------------------
print("Preparing GPT-4o...")

structured_llm = create_binding()


# -----------------------------
# Run inference
# -----------------------------
# NOTE:
# Inference is performed one prompt at a time.
# This intentionally does NOT use batch inference.

print("Starting inference...")

inference_start_time = time.time()

for prompt in tqdm(
    final_prompt,
    desc="Running inference",
):
    try:
        response = create_output(
            structured_llm,
            prompt,
        )

        with open(
            SAVE_JSON_PATH,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    response,
                    ensure_ascii=False,
                )
                + "\n"
            )

    except Exception as e:
        print(f"Error occurred while processing prompt: {prompt}")
        print(f"Error details: {e}")


# -----------------------------
# Save inference timing
# -----------------------------
inference_end_time = time.time()

inference_elapsed_time = inference_end_time - inference_start_time

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
# Finish
# -----------------------------
print("GPT-4o random few-shot inference " "completed and results saved.")
