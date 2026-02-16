"""Document loading, chunking, and vector-store ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app import config


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.LLAMA_CPP_EMBED_MODEL,
        base_url=config.LLAMA_CPP_EMBED_BASE_URL,
        api_key=SecretStr(config.LLAMA_CPP_API_KEY),
    )


def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="documents",
        embedding_function=_get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def ingest_file(file_path: Path) -> int:
    """Load a file, split it into chunks, and store embeddings.

    Returns the number of chunks created.
    """
    suffix = file_path.suffix.lower()
    loader: PyPDFLoader | TextLoader
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        # Fallback: treat as plain text (.txt, .md, .csv, …)
        loader = TextLoader(str(file_path), encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)


def get_retriever(k: int = 4) -> Any:
    """Return a retriever backed by the ChromaDB vector store."""
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(
    query: str,
    k: int = 4,
    threshold: float | None = None,
) -> list[Document]:
    """Retrieve documents filtered by a relevance score threshold.

    ChromaDB's ``similarity_search_with_relevance_scores`` returns
    ``(Document, score)`` pairs where *score* is a similarity value
    (higher = more relevant).  Documents below *threshold* are dropped.

    Returns an ordinary list of :class:`Document` objects.
    """
    from app import config  # local import to avoid circular ref at module level

    if threshold is None:
        threshold = config.RETRIEVAL_SCORE_THRESHOLD

    vectorstore = _get_vectorstore()

    try:
        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    except Exception:
        # Fallback: if relevance-score search is unavailable (e.g. empty
        # collection), return an empty list rather than crash.
        return []

    # Keep only documents whose similarity score meets the threshold
    return [doc for doc, score in scored_docs if score >= threshold]
