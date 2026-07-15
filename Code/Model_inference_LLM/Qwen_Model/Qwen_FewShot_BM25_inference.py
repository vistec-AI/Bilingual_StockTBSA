"""
LLM-based inference script with BM25 retrieval for Stock TBSA (Target-Based Sentiment Analysis)

This script performs few-shot inference using a large language model (LLM)
(e.g., Qwen2.5-72B-Instruct) by retrieving similar examples from a BM25 retriever
and constructing prompts for sentiment classification.
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
from pythainlp.tokenize import word_tokenize
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever

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
DATASET_PATH = "./path/to/test_set.json"  # <-- Replace with your test set
FEWSHOT_SET_PATH = (
    "./path/to/fewshot_dataset.json"  # <-- Replace with your few-shot pool
)
OUTPUT_JSON_PATH = "./path/to/output_predictions.jsonl"
RETRIEVED_DOCS_PATH = (
    "./path/to/retrieved_docs.csv"  # <-- Save retrieved few-shot examples
)
INFERENCE_TIME_LOG = "./path/to/inference_time.txt"  # <-- Save timing information

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # <-- Can be changed to other LLMs
API_KEY = "EMPTY"
API_BASE = "https://your-inference-endpoint.com/v1"
TEMPERATURE = 0.0
TOP_K = 3  # <-- For 3-shot BM25 retrieval


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
    # reason: Optional[str] = Field(...)  # Uncomment if you want reasoning


# -----------------------------
# Prompt Construction Helpers
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


def create_documents(df: pd.DataFrame) -> List[Document]:
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row["Text"],
            metadata={
                "ticker": str(row["TICKER"]),
                "data-source": str(row["Data-Source"]),
                "date": str(row["Date"]),
                "year": str(row["Year"]),
                "sentiment-class": str(row["Sentiment_class"]),
            },
            id=str(row["Article_ID"]),
        )
        documents.append(doc)
    return documents


def prepare_queries(file_path: str) -> list[str]:
    df = pd.read_json(file_path, lines=True)
    df = pd.concat([df], ignore_index=True)
    df["queries"] = (
        "TARGET_ARTICLE: "
        + df["Text"]
        + "\n"
        + "TICKER: "
        + df["TICKER"]
        + "\n"
        + "SENTIMENT_CLASS: "
    )
    return df["queries"].tolist()


def process_documents(
    queries: List[str], retriever: BM25Retriever, template: str
) -> List[List[str]]:
    searched_docs = [retriever.invoke(query) for query in tqdm(queries)]
    results = []
    for docs in searched_docs:
        sub_prompts = [
            template.format(
                text=doc.page_content,
                ticker=doc.metadata["ticker"],
                sentiment_class=doc.metadata["sentiment-class"],
            )
            for doc in docs
        ]
        results.append(sub_prompts)
    return results


def create_prompts(queries: List[str], searched_docs: List[List[str]]) -> List[str]:
    prompts = []
    for i, doc in enumerate(searched_docs):
        joined_examples = "\n".join(doc)
        prompts.append(joined_examples + "\n" + queries[i])
    return prompts


def create_output(structured_llm: RunnableBinding, prompt: str) -> dict[str, str]:
    try:
        res = structured_llm.invoke(prompt)
        sentiment = res.sentiment[0].value
        return dict(sentiment=sentiment, reason=np.nan)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return np.nan


# -----------------------------
# Prepare Few-shot Retrieval Docs
# -----------------------------
print("Preparing documents for BM25 retriever...")
fewshot_df = pd.read_json(FEWSHOT_SET_PATH, lines=True)
documents = create_documents(fewshot_df)

print("Preparing queries from test set...")
queries = prepare_queries(DATASET_PATH)  # Testing set

# -----------------------------
# Initialize BM25 Retriever
# -----------------------------
print("Initializing BM25 retriever...")
retriever = BM25Retriever.from_documents(
    documents, k=TOP_K, preprocess_func=word_tokenize
)

# -----------------------------
# BM25 Retrieval + Prompt Construction
# -----------------------------
print("Retrieving few-shot examples using BM25...")
start_time = time.time()

searched_docs = process_documents(queries, retriever, FEW_SHOT_TEMPLATE)
prompts = create_prompts(queries, searched_docs)
final_prompt = [PROMPT_TEMPLATE + item for item in prompts]

end_time = time.time()
elapsed = end_time - start_time

# Save retrieved docs
pd.DataFrame({"retrieved_document": searched_docs}).to_csv(RETRIEVED_DOCS_PATH)

# Save retrieval time
with open(INFERENCE_TIME_LOG, "w") as f:
    f.write(
        f"Total Retrieve_BM25 Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

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
print("Starting LLM inference with BM25 prompts...")
start_time = time.time()

for prompt in tqdm(final_prompt):
    try:
        response = create_output(structured_llm, prompt)  # generate structured output
        with open(OUTPUT_JSON_PATH, "a") as f:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error occurred while processing prompt:\n{prompt}\nError details: {e}")

end_time = time.time()
elapsed = end_time - start_time

# Save inference time
with open(INFERENCE_TIME_LOG, "a") as f:
    f.write(
        f"Total Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

print("LLM inference completed and results saved.")
