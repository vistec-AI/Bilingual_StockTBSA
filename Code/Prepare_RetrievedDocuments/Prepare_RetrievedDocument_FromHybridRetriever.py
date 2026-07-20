# -----------------------------
# Prepare Hybrid-Retrieved Documents for Financial TBSA
# -----------------------------
"""
Retrieve few-shot examples using a hybrid dense-sparse retriever
that combines Chroma vector retrieval and BM25 retrieval.
"""

# -----------------------------
# Imports Libraries
# -----------------------------
import chromadb
import pandas as pd
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
TRAIN_PATH = (
    "./path/to/icl_data_pool.json"  # Train-validation data pool used for BM25 retrieval
)
TEST_PATH = "./path/to/test_dataset.json"  # Thai or English test dataset
CHROMA_DB_PATH = "./path/to/chroma_langchain_DB"  # Chroma database created by Prepare_VectorDatabase.py
RETRIEVED_OUTPUT_CSV = "./path/to/hybrid_retrieved_data.csv"

EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "financial_collection"

FINAL_K = 3  # Final number of retrieved examples used for k-shot ICL
DENSE_K = 10  # Number of candidates retrieved by the dense retriever
BM25_K = 10  # Number of candidates retrieved by the BM25 retriever

DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5


# -----------------------------
# Few-Shot Example Template
# -----------------------------
FEW_SHOT_TEMPLATE = """EXAMPLE: {text}
TICKER: {ticker}
SENTIMENT_CLASS: {sentiment_class}
"""


# -----------------------------
# Create LangChain Documents for BM25
# -----------------------------
def create_documents(df: pd.DataFrame) -> list[Document]:
    documents = []

    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row["Text"]),
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
# Initialize Persistent Chroma Vector Store
# -----------------------------
def initialize_vector_store(
    model_name: str = EMBEDDING_MODEL,
    db_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    collection_metadata: dict = {"hnsw:space": "cosine"},
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        show_progress=False,
    )

    persistent_client = chromadb.PersistentClient(path=db_path)

    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata=collection_metadata,
    )

    return vector_store


# -----------------------------
# Prepare Retrieval Queries
# -----------------------------
def prepare_query_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_json(file_path, lines=True)

    df["queries"] = (
        "TARGET_ARTICLE: "
        + df["Text"].astype(str)
        + "\n"
        + "TICKER: "
        + df["TICKER"].astype(str)
        + "\n"
        + "SENTIMENT_CLASS: "
    )

    return df


# -----------------------------
# Format Retrieved Documents
# -----------------------------
def format_retrieved_docs(
    docs: list[Document],
    template: str = FEW_SHOT_TEMPLATE,
) -> list[str]:
    formatted_docs = []

    for doc in docs:
        formatted_docs.append(
            template.format(
                text=doc.page_content,
                ticker=doc.metadata["ticker"],
                sentiment_class=doc.metadata["sentiment-class"],
            )
        )

    return formatted_docs


# -----------------------------
# Build Hybrid Ensemble Retriever
# -----------------------------
def build_hybrid_retriever(
    train_df: pd.DataFrame,
    dense_k: int = DENSE_K,
    bm25_k: int = BM25_K,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    dense_model_name: str = EMBEDDING_MODEL,
    chroma_db_path: str = CHROMA_DB_PATH,
    chroma_collection_name: str = COLLECTION_NAME,
):
    """
    Build a hybrid retriever that combines dense vector retrieval
    and BM25 retrieval using EnsembleRetriever.

    dense_k and bm25_k control the number of candidates retrieved
    by each retriever before the ensemble ranking.

    dense_weight and bm25_weight control the contribution of each
    retriever to the final ranking.
    """

    # Dense/HNSW retriever
    vector_store = initialize_vector_store(
        model_name=dense_model_name,
        db_path=chroma_db_path,
        collection_name=chroma_collection_name,
    )

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": dense_k})

    # Sparse/BM25 retriever
    documents = create_documents(train_df)

    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=bm25_k,
        preprocess_func=word_tokenize,
    )

    # Hybrid ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, bm25_weight],
    )

    return hybrid_retriever, dense_retriever, bm25_retriever


# -----------------------------
# Retrieve Documents into DataFrame
# -----------------------------
def retrieve_into_dataframe(
    query_df: pd.DataFrame,
    retriever,
    final_k: int = 3,
    template: str = FEW_SHOT_TEMPLATE,
) -> pd.DataFrame:
    retrieved_docs_all = []

    for query in tqdm(
        query_df["queries"].tolist(),
        desc="Hybrid retrieving",
    ):
        docs = retriever.invoke(query)

        # EnsembleRetriever may return more documents than final_k
        docs = docs[:final_k]

        formatted_docs = format_retrieved_docs(
            docs=docs,
            template=template,
        )

        retrieved_docs_all.append(formatted_docs)

    output_df = query_df.copy()
    output_df["retrieved_document"] = retrieved_docs_all

    return output_df


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load the train-validation data pool for BM25
    train_df = pd.read_json(TRAIN_PATH, lines=True)

    # Load and prepare the test queries
    query_df = prepare_query_dataframe(TEST_PATH)

    # Build the hybrid retriever
    hybrid_retriever, dense_retriever, bm25_retriever = build_hybrid_retriever(
        train_df=train_df,
        dense_k=DENSE_K,
        bm25_k=BM25_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        dense_model_name=EMBEDDING_MODEL,
        chroma_db_path=CHROMA_DB_PATH,
        chroma_collection_name=COLLECTION_NAME,
    )

    # Retrieve and format few-shot examples
    retrieve_df = retrieve_into_dataframe(
        query_df=query_df,
        retriever=hybrid_retriever,
        final_k=FINAL_K,
        template=FEW_SHOT_TEMPLATE,
    )

    # Save the test dataset with retrieved documents
    retrieve_df.to_csv(RETRIEVED_OUTPUT_CSV, index=False)

    print("Hybrid-retrieved documents saved successfully.")
