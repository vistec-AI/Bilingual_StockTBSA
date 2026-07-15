# -----------------------------
# Vector Database Preparation for Financial TBSA
# -----------------------------
"""
This script prepares a vector database using Chroma + HuggingFace embeddings
to be used for vector-based retrieval in LLM inference pipelines (e.g., few-shot retrieval).
"""

# -----------------------------
# Standard library imports
# -----------------------------
from uuid import uuid4

# -----------------------------
# Third-party imports
# -----------------------------
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# Create LangChain-compatible Document objects
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
# Initialize Chroma vector store
# -----------------------------
def create_vector_store(
    model_name="BAAI/bge-m3",
    collection_name="financial_collection",
    persist_directory="./path/to/chroma_langchain_DB",  # Save path for you Vector database
    collection_metadata={"hnsw:space": "cosine"},
):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, show_progress=True)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata=collection_metadata,
    )
    return vector_store


# -----------------------------
# Full vector store pipeline: document → vector → persist
# -----------------------------
def initialize_vector_store(df: pd.DataFrame):
    documents = create_documents(df)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store = create_vector_store()
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    dataset_path = "./path/to/Data_Pool.json"  # Path for input data (we use train + validation set)
    df_train = pd.read_json(dataset_path, lines=True)
    df = pd.concat([df_train], ignore_index=True)
    vector_store = initialize_vector_store(df)
    print("Vector store initialized and persisted.")
