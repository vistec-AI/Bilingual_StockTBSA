"""
LLM-based inference script with vector retrieval for Stock TBSA (Target-Based Sentiment Analysis)

This script performs few-shot inference using a large language model (LLM)
(e.g., Qwen2.5-72B-Instruct) by retrieving similar examples from a vector store and constructing prompts for sentiment classification.
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
import chromadb
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableBinding
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
VECTOR_DB_PATH = "./path/to/vector_db"  # <-- Replace with "Vector Database" path
OUTPUT_JSON_PATH = "./path/to/output_predictions.json"
RETRIEVED_DOCS_PATH = "./path/to/retrieved_docs.csv"  # <-- Save retrieved documents
INFERENCE_TIME_LOG = "./path/to/inference_time.txt"  # <-- Save timing information

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # can change to other LLM such as "meta-llama/Llama-3.1-70B-Instruct"
API_KEY = "EMPTY"  # <-- Replace with your actual API key
API_BASE = "https://your-inference-endpoint.com/v1"  # <-- Replace with your API base
TEMPERATURE = 0.0
TOP_K = 3  # <-- For 3-shot (change a number for n-shot)


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
    # reason: Optional[str] = Field(...)


# -----------------------------
# Vector Retrieval Initialization
# -----------------------------
# IMPORTANT:
# Before running this inference script, you must first build the vector database
# using a representative sample pool (e.g., training + validation sets).
# These samples will later serve as the source for few-shot example retrieval during inference.
#
# To create the vector database, please refer to:
#     "Prepare_VectorDatabase/Prepare_VectorDatabase.py"
#
# Once the vector database has been created and persisted,
# you can re-initialize the vector store using the function below
# to enable document retrieval for constructing few-shot prompts.


def initialize_vector_store(
    model_name: str = "BAAI/bge-m3",
    db_path: str = VECTOR_DB_PATH,
    collection_name: str = "financial_collection",
    collection_metadata={"hnsw:space": "cosine"},
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
    persistent_client = chromadb.PersistentClient(path=db_path)
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata=collection_metadata,
    )
    return vector_store


# -----------------------------
# Prompt Construction Helpers
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


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
    queries: List[str], retriever: VectorStoreRetriever, template: str
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
        docs_text = "\n".join(d for d in doc)
        prompts.append(docs_text + "\n" + queries[i])
    return prompts


def create_output(structured_llm: RunnableBinding, prompt: str) -> dict[str, str]:
    try:
        res = structured_llm.invoke(prompt)
        sentiment = res.sentiment[0].value
        return dict(sentiment=sentiment, reason=np.nan)
    except Exception as e:
        print(e)
        return np.nan


# -----------------------------
# Load test data and prepare prompts
# -----------------------------
print("Initializing vector store and preparing prompts...")
vector_store = initialize_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

queries = prepare_queries(DATASET_PATH)  # Testing set
searched_docs = process_documents(queries, retriever, FEW_SHOT_TEMPLATE)

# Save retrieved examples for reproducibility
pd.DataFrame({"retrieved_document": searched_docs}).to_csv(RETRIEVED_DOCS_PATH)

# Final prompts with retrieved examples
prompts = create_prompts(queries, searched_docs)
final_prompt = [PROMPT_TEMPLATE + item for item in prompts]

# -----------------------------
# Initialize LLM with structured output
# -----------------------------
structured_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
).with_structured_output(Sentiment)

# -----------------------------
# Inference + Timing
# -----------------------------
print("Starting LLM inference with vector retrieval...")
start_time = time.time()

for prompt in tqdm(final_prompt):
    try:
        response = create_output(structured_llm, prompt)  # generate structured output
        with open(OUTPUT_JSON_PATH, "a") as f:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error occurred while processing prompt:\n{prompt}\nError: {e}")


end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# Save timing info
# -----------------------------
with open(INFERENCE_TIME_LOG, "w") as f:
    f.write(
        f"Total Inference Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n\n"
    )

print("LLM inference completed and results saved.")
