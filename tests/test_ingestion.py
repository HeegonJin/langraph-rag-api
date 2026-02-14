"""Tests for the ingestion module (document loading & splitting).

Mocks the vector store to avoid needing a running embedding server.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.rag.ingestion import ingest_file, get_retriever


class TestIngestFile:
    @patch("app.rag.ingestion._get_vectorstore")
    def test_ingest_txt_file(self, mock_vs, tmp_path):
        """Ingest a plain text file and verify chunks are created."""
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


class TestGetRetriever:
    @patch("app.rag.ingestion._get_vectorstore")
    def test_returns_retriever(self, mock_vs):
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_vs.return_value = mock_store

        retriever = get_retriever(k=3)
        assert retriever is mock_retriever
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})

    @patch("app.rag.ingestion._get_vectorstore")
    def test_default_k(self, mock_vs):
        mock_store = MagicMock()
        mock_vs.return_value = mock_store

        get_retriever()
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
