"""
LLM-based inference script with Random Retrieval for Stock TBSA (Target-Based Sentiment Analysis)

This script performs few-shot inference using a large language model (LLM)
(e.g., Qwen2.5-72B-Instruct) by randomly selecting few-shot examples from a sample pool
and constructing prompts for sentiment classification.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import os
import time
import json
import random
from typing import List

# -----------------------------
# Third-party imports
# -----------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum, EnumMeta
from pythainlp.tokenize import word_tokenize
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI

# -----------------------------
# Prompt template
# -----------------------------
# 🔹NOTE:
# - This is the 3-shot English prompt with vector retrieval used for inference.
# - A corresponding Thai version is available at:
#   `Code/Examples_PromptTemplate/3-shot_LongPrompt_Thai.txt`

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
# Configuration (replace with your actual paths)
# -----------------------------
TEST_PATH = "./path/to/test_set.json"  # <-- Replace with your test set
FEWSHOT_POOL_PATH = "./path/to/fewshot_pool.json"  # <-- Replace with your few-shot pool (we use Train + Validation set)
OUTPUT_JSON_PATH = "./path/to/output_predictions.jsonl"
RANDOM_EXAMPLES_PATH = "./path/to/retrieved_random_examples.csv"
INFERENCE_TIME_LOG = "./path/to/inference_time.txt"

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # <-- Can be changed to other LLMs
API_KEY = "EMPTY"
API_BASE = "https://your-inference-endpoint.com/v1"
TEMPERATURE = 0.0
TOP_K = 3  # <-- Number of random examples


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
    # reason: Optional[str] = Field(...)  # Uncomment if reasoning is required


# -----------------------------
# Prompt Construction Helpers
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


# -----------------------------
# Few-shot Prompt Construction
# -----------------------------
def add_random_examples(
    df_A: pd.DataFrame, df_B: pd.DataFrame, n: int = 3
) -> pd.DataFrame:
    random_examples_list = []
    for _ in range(len(df_A)):
        examples = df_B.sample(n=n, random_state=random.randint(0, 10000))
        examples_dict = [
            {
                "text": row["Text"],
                "ticker": row["TICKER"],
                "sentiment_class": row["Sentiment_class"],
            }
            for _, row in examples.iterrows()
        ]
        random_examples_list.append(examples_dict)
    df_A = df_A.copy()
    df_A["Random_example"] = random_examples_list
    return df_A


def format_few_shot(examples: List[dict]) -> str:
    return "\n".join([FEW_SHOT_TEMPLATE.format(**ex) for ex in examples])


# -----------------------------
# Output Inference Wrapper
# -----------------------------
def create_output(structured_llm: RunnableBinding, prompt: str) -> dict[str, str]:
    try:
        res = structured_llm.invoke(prompt)
        sentiment = res.sentiment[0].value
        return dict(sentiment=sentiment, reason=np.nan)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return np.nan


# -----------------------------
# Load Data + Prepare Prompts
# -----------------------------
print("Loading and preparing test and few-shot pool...")
df_test = pd.read_json(TEST_PATH, lines=True)
df_fewshot = pd.read_json(FEWSHOT_POOL_PATH, lines=True)

# Attach random examples to each row in test set
df_with_examples = add_random_examples(df_test, df_fewshot, n=TOP_K)

# Format few-shot and input prompts
df_with_examples["Few_shot_prompt"] = df_with_examples["Random_example"].apply(
    format_few_shot
)
df_with_examples["prompt"] = (
    "TARGET_ARTICLE: "
    + df_with_examples["Text"]
    + "\n"
    + "TICKER: "
    + df_with_examples["TICKER"]
    + "\n"
    + "SENTIMENT_CLASS: "
)
df_with_examples["Input_Prompt"] = (
    df_with_examples["Few_shot_prompt"] + "\n" + df_with_examples["prompt"]
)

# Save retrieved examples (optional)
df_with_examples.to_csv(RANDOM_EXAMPLES_PATH, index=False)

# Construct final prompt
final_prompt = [
    PROMPT_TEMPLATE + item for item in df_with_examples["Input_Prompt"].tolist()
]

# -----------------------------
# Initialize LLM
# -----------------------------
structured_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
).with_structured_output(Sentiment)

# -----------------------------
# Inference + Save
# -----------------------------
print("Starting LLM inference with random few-shot prompts...")
start_time = time.time()

for prompt in tqdm(final_prompt):
    try:
        response = create_output(structured_llm, prompt)
        with open(OUTPUT_JSON_PATH, "a") as f:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error occurred while processing prompt:\n{prompt}\nError details: {e}")

end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# Log inference time
# -----------------------------
with open(INFERENCE_TIME_LOG, "w") as f:
    f.write(
        f"Total Random_Retrieval Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

print("LLM inference completed and results saved.")
