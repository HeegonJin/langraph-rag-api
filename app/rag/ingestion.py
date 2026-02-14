"""Document loading, chunking, and vector-store ingestion."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app import config


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.LLAMA_CPP_EMBED_MODEL,
        base_url=config.LLAMA_CPP_EMBED_BASE_URL,
        api_key=config.LLAMA_CPP_API_KEY,
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


def get_retriever(k: int = 4):
    """Return a retriever backed by the ChromaDB vector store."""
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
