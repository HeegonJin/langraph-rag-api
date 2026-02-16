"""Tests for the ingestion module (Docling PDF + Elasticsearch pipeline).

Mocks the vector store and Docling converter to avoid needing running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document

from app.rag.ingestion import ingest_file, retrieve_with_scores


def _make_docling_chunk(
    text: str,
    headings: list[str] | None = None,
    label: DocItemLabel = DocItemLabel.TEXT,
    page_no: int = 1,
) -> MagicMock:
    """Build a mock DocChunk with realistic meta structure."""
    prov = MagicMock()
    prov.page_no = page_no

    doc_item = MagicMock()
    doc_item.prov = [prov]
    doc_item.label = label

    meta = MagicMock(spec=DocMeta)
    meta.headings = headings
    meta.doc_items = [doc_item]

    chunk = MagicMock()
    chunk.text = text
    chunk.meta = meta
    return chunk


class TestIngestFile:
    # ── Text file tests (preserved behavior) ──────────────────────────────

    @patch("app.rag.ingestion._get_vectorstore")
    def test_ingest_txt_file(self, mock_vs, tmp_path):
        """Create tmp .txt -> ingest -> verify chunks stored via mock ES."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world. " * 200, encoding="utf-8")

        num_chunks = ingest_file(txt_file)
        assert num_chunks > 0
        mock_store.add_documents.assert_called_once()
        chunks = mock_store.add_documents.call_args[0][0]
        assert len(chunks) == num_chunks

    @patch("app.rag.ingestion._get_vectorstore")
    def test_ingest_md_file(self, mock_vs, tmp_path):
        """Markdown files should be treated as plain text."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\nSome content.\n" * 100, encoding="utf-8")

        num_chunks = ingest_file(md_file)
        assert num_chunks > 0

    @patch("app.rag.ingestion._get_vectorstore")
    def test_ingest_small_file_single_chunk(self, mock_vs, tmp_path):
        """A very small file should produce exactly one chunk."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        txt_file = tmp_path / "small.txt"
        txt_file.write_text("Short text.", encoding="utf-8")

        num_chunks = ingest_file(txt_file)
        assert num_chunks == 1

    # ── PDF tests (NEW — Docling pipeline) ────────────────────────────────

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_uses_docling(self, mock_conv, mock_chunker_cls, mock_vs, tmp_path):
        """Mock DocumentConverter -> verify called for .pdf files."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        # Set up converter mock
        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        # Set up chunker mock
        chunk = _make_docling_chunk("Some PDF content")
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunker.contextualize.return_value = "Some PDF content"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        num_chunks = ingest_file(pdf_file)
        assert num_chunks == 1
        mock_conv.return_value.convert.assert_called_once_with(str(pdf_file))

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_extracts_tables_as_markdown(
        self, mock_conv, mock_chunker_cls, mock_vs, tmp_path
    ):
        """Mock Docling -> doc with table items -> verify table chunk_type."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        table_chunk = _make_docling_chunk(
            "| Col A | Col B |\n|---|---|\n| 1 | 2 |",
            label=DocItemLabel.TABLE,
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [table_chunk]
        mock_chunker.contextualize.return_value = "| Col A | Col B |\n|---|---|\n| 1 | 2 |"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "tables.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        stored_docs = mock_store.add_documents.call_args[0][0]
        assert stored_docs[0].metadata["chunk_type"] == "table"
        assert "|" in stored_docs[0].page_content

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_extracts_image_metadata(
        self, mock_conv, mock_chunker_cls, mock_vs, tmp_path
    ):
        """Mock Docling -> doc with PictureItem -> verify metadata."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        picture_chunk = _make_docling_chunk(
            "Figure 1: Architecture diagram",
            label=DocItemLabel.PICTURE,
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [picture_chunk]
        mock_chunker.contextualize.return_value = "Figure 1: Architecture diagram"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "images.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        stored_docs = mock_store.add_documents.call_args[0][0]
        assert stored_docs[0].metadata["chunk_type"] == "picture"

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_preserves_section_headings(
        self, mock_conv, mock_chunker_cls, mock_vs, tmp_path
    ):
        """Mock Docling -> doc with headings -> verify in metadata."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        chunk = _make_docling_chunk(
            "Content under heading",
            headings=["Chapter 1", "Section 1.1"],
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunker.contextualize.return_value = "Chapter 1 > Section 1.1\nContent under heading"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "headings.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        stored_docs = mock_store.add_documents.call_args[0][0]
        assert stored_docs[0].metadata["headings"] == ["Chapter 1", "Section 1.1"]

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_uses_hybrid_chunker(self, mock_conv, mock_chunker_cls, mock_vs, tmp_path):
        """Mock HybridChunker -> verify it is instantiated and called."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        chunk = _make_docling_chunk("Text chunk")
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunker.contextualize.return_value = "Text chunk"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "chunker.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        mock_chunker_cls.assert_called_once_with(merge_peers=True)
        mock_chunker.chunk.assert_called_once_with(mock_doc)

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_ingest_pdf_contextualizes_chunks(
        self, mock_conv, mock_chunker_cls, mock_vs, tmp_path
    ):
        """Mock contextualize() -> verify heading context in text."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        chunk = _make_docling_chunk("Raw text", headings=["Intro"])
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunker.contextualize.return_value = "Intro\nRaw text"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "ctx.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        mock_chunker.contextualize.assert_called_once_with(chunk)
        stored_docs = mock_store.add_documents.call_args[0][0]
        assert stored_docs[0].page_content == "Intro\nRaw text"

    # ── Vector store tests ────────────────────────────────────────────────

    @patch("app.rag.ingestion._get_vectorstore")
    def test_chunks_stored_in_elasticsearch(self, mock_vs, tmp_path):
        """Mock ES store -> verify add_documents called."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        txt_file = tmp_path / "store.txt"
        txt_file.write_text("Test content for storing.", encoding="utf-8")

        ingest_file(txt_file)
        mock_store.add_documents.assert_called_once()

    @patch("app.rag.ingestion._get_vectorstore")
    @patch("app.rag.ingestion.HybridChunker")
    @patch("app.rag.ingestion._get_converter")
    def test_chunk_metadata_includes_source(self, mock_conv, mock_chunker_cls, mock_vs, tmp_path):
        """Verify metadata dict has 'source' key for PDF chunks."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_conv.return_value.convert.return_value = mock_result

        chunk = _make_docling_chunk("Content with source")
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]
        mock_chunker.contextualize.return_value = "Content with source"
        mock_chunker_cls.return_value = mock_chunker

        pdf_file = tmp_path / "source.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ingest_file(pdf_file)

        stored_docs = mock_store.add_documents.call_args[0][0]
        assert "source" in stored_docs[0].metadata
        assert stored_docs[0].metadata["source"] == str(pdf_file)

    # ── Error handling ────────────────────────────────────────────────────

    def test_ingest_nonexistent_file_raises(self, tmp_path):
        """Path doesn't exist -> error."""
        missing = tmp_path / "nonexistent.txt"
        try:
            ingest_file(missing)
            raise AssertionError("Expected an exception")
        except Exception:
            pass

    @patch("app.rag.ingestion._get_vectorstore")
    def test_ingest_unsupported_extension_falls_back_to_text(self, mock_vs, tmp_path):
        """.csv -> TextLoader used (text fallback)."""
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        num_chunks = ingest_file(csv_file)
        assert num_chunks > 0
        mock_store.add_documents.assert_called_once()


class TestRetrieveWithScores:
    @patch("app.rag.ingestion._get_vectorstore")
    def test_returns_docs_above_threshold(self, mock_vs):
        """Only documents above the threshold are returned."""
        mock_store = MagicMock()
        doc1 = Document(page_content="relevant", metadata={})
        doc2 = Document(page_content="irrelevant", metadata={})
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.9),
            (doc2, 0.3),
        ]
        mock_vs.return_value = mock_store

        results = retrieve_with_scores("test query", threshold=0.5)
        assert len(results) == 1
        assert results[0].page_content == "relevant"

    @patch("app.rag.ingestion._get_vectorstore")
    def test_returns_all_docs_when_threshold_zero(self, mock_vs):
        """Threshold of 0.0 returns all documents (no filtering)."""
        mock_store = MagicMock()
        doc1 = Document(page_content="a", metadata={})
        doc2 = Document(page_content="b", metadata={})
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.9),
            (doc2, 0.1),
        ]
        mock_vs.return_value = mock_store

        results = retrieve_with_scores("test query", threshold=0.0)
        assert len(results) == 2

    @patch("app.rag.ingestion._get_vectorstore")
    def test_raises_on_exception(self, mock_vs):
        """Exceptions from vectorstore propagate instead of being silently swallowed."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.side_effect = RuntimeError("fail")
        mock_vs.return_value = mock_store

        with pytest.raises(RuntimeError, match="fail"):
            retrieve_with_scores("test query")

    @patch("app.config.RETRIEVAL_SCORE_THRESHOLD", 0.6)
    @patch("app.rag.ingestion._get_vectorstore")
    def test_default_threshold_from_config(self, mock_vs):
        """Uses config threshold when none is provided."""
        mock_store = MagicMock()
        doc1 = Document(page_content="above", metadata={})
        doc2 = Document(page_content="below", metadata={})
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.7),
            (doc2, 0.5),
        ]
        mock_vs.return_value = mock_store

        results = retrieve_with_scores("test query")
        assert len(results) == 1
        assert results[0].page_content == "above"

    @patch("app.rag.ingestion._get_vectorstore")
    def test_respects_k_parameter(self, mock_vs):
        """The k parameter is forwarded to the vectorstore search."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mock_vs.return_value = mock_store

        retrieve_with_scores("test query", k=10, threshold=0.0)
        mock_store.similarity_search_with_relevance_scores.assert_called_once_with(
            "test query", k=10
        )

    @patch("app.rag.ingestion._get_vectorstore")
    def test_returns_empty_for_no_matches(self, mock_vs):
        """Returns empty list when search returns no results."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mock_vs.return_value = mock_store

        results = retrieve_with_scores("test query", threshold=0.5)
        assert results == []
