"""Tests for FastAPI endpoints.

Uses httpx + pytest-asyncio with mocked graph invocations so no LLM
or vector store is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


# We disable startup/shutdown events to avoid sample_data ingestion & Langfuse
# flushes during tests.  The on_event decorator captures the original function
# reference, so patching the module-level name is not enough – we must clear the
# handler lists on the router.
@pytest.fixture()
def client():
    """Create a TestClient with lifespan disabled to avoid sample_data ingestion."""
    from app.main import app

    # Override the lifespan to a no-op so tests don't require Redis or sample data
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    finally:
        app.router.lifespan_context = original_lifespan


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Root redirect ─────────────────────────────────────────────────────────────


class TestRoot:
    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")


# ── POST /ask (single-turn) ──────────────────────────────────────────────────


class TestAskEndpoint:
    @patch("app.main.invoke_graph_with_tracing")
    def test_ask_success(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "Test answer",
            "documents": [Document(page_content="snippet content here")],
            "grounded": True,
        }
        resp = client.post("/ask", json={"question": "What is RAG?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert len(data["source_documents"]) == 1

    @patch("app.main.invoke_graph_with_tracing")
    def test_ask_no_documents(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "I don't know.",
            "documents": [],
            "grounded": False,
        }
        resp = client.post("/ask", json={"question": "Unknown?"})
        assert resp.status_code == 200
        assert resp.json()["source_documents"] == []

    def test_ask_empty_question(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_missing_question(self, client):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422


# ── POST /chat (multi-turn) ──────────────────────────────────────────────────


class TestChatEndpoint:
    @patch("app.main.invoke_graph_with_tracing")
    def test_chat_new_session(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "Hello!",
            "intent": "new_topic",
            "documents": [],
            "grounded": True,
        }
        resp = client.post("/chat", json={"question": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Hello!"
        assert "session_id" in data
        assert data["intent"] == "new_topic"
        assert isinstance(data["turn_number"], int)

    @patch("app.main.invoke_graph_with_tracing")
    def test_chat_with_session_id(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "Follow up!",
            "intent": "follow_up",
            "documents": [Document(page_content="doc")],
            "grounded": True,
        }
        resp = client.post(
            "/chat",
            json={"question": "more info", "session_id": "my-session"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "my-session"

    def test_chat_empty_question(self, client):
        resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 422


# ── POST /chat/clear ─────────────────────────────────────────────────────────


class TestClearContext:
    @patch("app.main.invoke_graph_with_tracing")
    def test_clear_existing_session(self, mock_invoke, client):
        # First create a session via /chat
        mock_invoke.return_value = {
            "answer": "ok",
            "intent": "new_topic",
            "documents": [],
            "grounded": True,
        }
        chat_resp = client.post(
            "/chat", json={"question": "hi", "session_id": "clear-me"}
        )
        session_id = chat_resp.json()["session_id"]

        # Now clear it
        resp = client.post("/chat/clear", json={"session_id": session_id})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    def test_clear_nonexistent_session(self, client):
        resp = client.post("/chat/clear", json={"session_id": "nope"})
        assert resp.status_code == 404


# ── GET /chat/sessions ────────────────────────────────────────────────────────


class TestListSessions:
    @patch("app.main.invoke_graph_with_tracing")
    def test_list_after_creating(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "ok",
            "intent": "new_topic",
            "documents": [],
            "grounded": True,
        }
        client.post("/chat", json={"question": "hi", "session_id": "list-test"})

        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "list-test" in data["sessions"]
        assert data["count"] >= 1


# ── DELETE /chat/sessions/{id} ────────────────────────────────────────────────


class TestDeleteSession:
    @patch("app.main.invoke_graph_with_tracing")
    def test_delete_existing(self, mock_invoke, client):
        mock_invoke.return_value = {
            "answer": "ok",
            "intent": "new_topic",
            "documents": [],
            "grounded": True,
        }
        client.post("/chat", json={"question": "hi", "session_id": "del-me"})

        resp = client.delete("/chat/sessions/del-me")
        assert resp.status_code == 200

        # Should be gone now
        resp2 = client.delete("/chat/sessions/del-me")
        assert resp2.status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/chat/sessions/nope")
        assert resp.status_code == 404


# ── POST /ingest ──────────────────────────────────────────────────────────────


class TestIngestEndpoint:
    @patch("app.main.ingest_file", return_value=5)
    def test_ingest_txt(self, mock_ingest, client):
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["num_chunks"] == 5

    @patch("app.main.ingest_file", side_effect=ValueError("bad file"))
    def test_ingest_failure(self, mock_ingest, client):
        resp = client.post(
            "/ingest",
            files={"file": ("bad.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 422
