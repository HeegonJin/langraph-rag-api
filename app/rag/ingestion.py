"""Document loading, chunking, and Elasticsearch ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.doc_chunk import DocMeta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app import config

logger = logging.getLogger(__name__)

# ── Docling converter (module-level lazy singleton) ───────────────────────────


def _build_converter() -> DocumentConverter:
    table_mode = (
        TableFormerMode.ACCURATE
        if config.DOCLING_TABLE_MODE == "accurate"
        else TableFormerMode.FAST
    )
    pipeline_options = PdfPipelineOptions(
        do_ocr=config.DOCLING_OCR_ENABLED,
        ocr_options=EasyOcrOptions(lang=["en", "ko"]),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=table_mode),
        generate_picture_images=True,
        document_timeout=config.DOCLING_TIMEOUT,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        logger.info("Initializing Docling DocumentConverter (first use)...")
        _converter = _build_converter()
    return _converter


# ── Embeddings & vector store ─────────────────────────────────────────────────


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.LLAMA_CPP_EMBED_MODEL,
        base_url=config.LLAMA_CPP_EMBED_BASE_URL,
        api_key=SecretStr(config.LLAMA_CPP_API_KEY),
    )


def _get_vectorstore() -> ElasticsearchStore:
    return ElasticsearchStore(
        index_name=config.ELASTICSEARCH_INDEX,
        embedding=_get_embeddings(),
        es_url=config.ELASTICSEARCH_URL,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────


def _ingest_pdf(file_path: Path) -> list[Document]:
    converter = _get_converter()
    result = converter.convert(str(file_path))
    dl_doc = result.document

    chunker = HybridChunker(merge_peers=True)
    chunks = list(chunker.chunk(dl_doc))

    documents: list[Document] = []
    for chunk in chunks:
        contextualized = chunker.contextualize(chunk)
        metadata: dict[str, object] = {
            "source": str(file_path),
            "chunk_type": "text",
        }
        meta = chunk.meta
        if isinstance(meta, DocMeta):
            if meta.headings:
                metadata["headings"] = meta.headings
            # Extract page number from first doc_item's provenance
            if meta.doc_items:
                first_item = meta.doc_items[0]
                if first_item.prov:
                    metadata["page"] = first_item.prov[0].page_no
                # Tag chunk type based on doc item label
                label_val = (
                    first_item.label.value
                    if hasattr(first_item.label, "value")
                    else str(first_item.label)
                )
                if label_val == "table":
                    metadata["chunk_type"] = "table"
                elif label_val == "picture":
                    metadata["chunk_type"] = "picture"

        documents.append(
            Document(
                page_content=contextualized,
                metadata=metadata,
            )
        )

    return documents


def _ingest_text(file_path: Path) -> list[Document]:
    loader = TextLoader(str(file_path), encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)


def ingest_file(file_path: Path) -> int:
    """Load a file, split it into chunks, and store embeddings.

    Returns the number of chunks created.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        chunks = _ingest_pdf(file_path)
    else:
        chunks = _ingest_text(file_path)

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────


def retrieve_with_scores(
    query: str,
    k: int = 4,
    threshold: float | None = None,
) -> list[Document]:
    """Retrieve documents using Elasticsearch hybrid search (BM25 + dense vector via RRF)."""
    from app import config as _config  # local import to avoid circular ref

    if threshold is None:
        threshold = _config.RETRIEVAL_SCORE_THRESHOLD

    vectorstore = _get_vectorstore()

    try:
        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    except Exception:
        return []

    if threshold > 0.0:
        return [doc for doc, score in scored_docs if score >= threshold]

    return [doc for doc, _score in scored_docs]
