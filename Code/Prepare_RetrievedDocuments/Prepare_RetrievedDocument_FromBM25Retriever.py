# -----------------------------
# Prepare BM25-Retrieved Documents for Financial TBSA
# -----------------------------
"""
Retrieve few-shot examples from an ICL data pool using BM25
and combine the retrieved documents with the test dataset.
"""

# -----------------------------
# Imports Libraries
# -----------------------------
import pandas as pd
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
ICL_DATA_PATH = (
    "./path/to/icl_data_pool.json"  # Train-validation data pool used for retrieval
)
DATASET_PATH = "./path/to/test_dataset.json"  # Thai or English test dataset
OUTPUT_PATH = (
    "./path/to/bm25_retrieved_data.csv"  # Output dataset with retrieved documents
)

NUM_RETRIEVED_DOCUMENTS = 3  # Number of retrieved examples (k) used for k-shot ICL


# -----------------------------
# Few-Shot Example Template
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


# -----------------------------
# Create LangChain Documents
# -----------------------------
def create_documents(df: pd.DataFrame):
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


# -----------------------------
# Prepare Retrieval Queries
# -----------------------------
def prepare_queries(file_path: str) -> list[str]:
    df = pd.read_json(file_path, lines=True)

    df["queries"] = (
        "TARGET_ARTICLE: "
        + df["Text"]
        + "\nTICKER: "
        + df["TICKER"]
        + "\nSENTIMENT_CLASS: "
    )

    return df["queries"].tolist()


# -----------------------------
# Retrieve and Format Documents
# -----------------------------
def process_documents(
    queries: list[str],
    retriever: BM25Retriever,
    template: str,
) -> list[list[str]]:
    searched_docs = [retriever.invoke(query) for query in tqdm(queries)]

    documents = []

    for docs in tqdm(searched_docs):
        formatted_docs = [
            template.format(
                text=doc.page_content,
                ticker=doc.metadata["ticker"],
                sentiment_class=doc.metadata["sentiment-class"],
            )
            for doc in docs
        ]
        documents.append(formatted_docs)

    return documents


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load the ICL data pool
    icl_data = pd.read_json(ICL_DATA_PATH, lines=True)
    documents = create_documents(icl_data)

    # Initialize the BM25 retriever
    retriever = BM25Retriever.from_documents(
        documents,
        k=NUM_RETRIEVED_DOCUMENTS,
        preprocess_func=word_tokenize,
    )

    # Prepare test queries and retrieve few-shot examples
    queries = prepare_queries(DATASET_PATH)
    retrieved_documents = process_documents(
        queries,
        retriever,
        FEW_SHOT_TEMPLATE,
    )

    # -----------------------------
    # Optional: Merge Retrieved Documents with the Test Dataset
    # -----------------------------
    # Skip this section to save retrieved documents separately; merging is useful for later inference.
    test_data = pd.read_json(DATASET_PATH, lines=True)
    retrieve_df = pd.DataFrame({"retrieved_document": retrieved_documents})

    # Reset indices before combining the datasets
    test_data = test_data.reset_index(drop=True)
    retrieve_df = retrieve_df.reset_index(drop=True)

    # Ensure that both datasets have the same number of rows
    assert len(test_data) == len(retrieve_df), (
        f"Row mismatch: test_data={len(test_data)}, " f"retrieve_df={len(retrieve_df)}"
    )

    # Add retrieved documents to the test dataset
    test_data["retrieved_document"] = retrieve_df["retrieved_document"]
    test_data.to_csv(OUTPUT_PATH)

    print("BM25-retrieved documents saved successfully.")
