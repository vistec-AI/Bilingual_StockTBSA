# -----------------------------
# Prepare Vector-Retrieved Documents for Financial TBSA
# -----------------------------
"""
Retrieve few-shot examples from a persistent Chroma vector database
and save the retrieved documents for use in LLM inference.
"""

# -----------------------------
# Imports Libraries
# -----------------------------
import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
VECTOR_DB_PATH = "./path/to/chroma_langchain_DB"  # Path to the Chroma vector database created by Prepare_VectorDatabase.py
DATASET_PATH = "./path/to/test_dataset.json"  # Path to the Thai or English test dataset
OUTPUT_PATH = "./path/to/vector_retrieved_documents.csv"  # Path for saving the retrieved documents

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "financial_collection"
NUM_RETRIEVED_DOCUMENTS = 3  # Number of retrieved examples (k) used for k-shot ICL


# -----------------------------
# Few-Shot Example Template
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


# -----------------------------
# Initialize Persistent Chroma Vector Store
# -----------------------------
def initialize_vector_store(
    model_name: str = EMBEDDING_MODEL,
    db_path: str = VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    collection_metadata={"hnsw:space": "cosine"},
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
    persistent_client = chromadb.PersistentClient(path=db_path)

    return Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata=collection_metadata,
    )


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
    retriever: VectorStoreRetriever,
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
    vector_store = initialize_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCUMENTS})

    queries = prepare_queries(DATASET_PATH)
    retrieved_documents = process_documents(queries, retriever, FEW_SHOT_TEMPLATE)

    retrieve_df = pd.DataFrame({"retrieved_document": retrieved_documents})
    retrieve_df.to_csv(OUTPUT_PATH)

    print("Retrieved documents saved successfully.")
