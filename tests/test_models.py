"""Tests for Pydantic request/response models."""

import pytest
from pydantic import ValidationError

from app.models import (
    AnswerResponse,
    ChatRequest,
    ChatResponse,
    ClearContextRequest,
    ClearContextResponse,
    IngestResponse,
    QuestionRequest,
    SessionListResponse,
)

# ── QuestionRequest ───────────────────────────────────────────────────────────


class TestQuestionRequest:
    def test_valid(self):
        q = QuestionRequest(question="What is RAG?")
        assert q.question == "What is RAG?"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRequest(question="")

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRequest()


# ── ChatRequest ───────────────────────────────────────────────────────────────


class TestChatRequest:
    def test_valid_without_session(self):
        r = ChatRequest(question="hello")
        assert r.question == "hello"
        assert r.session_id is None

    def test_valid_with_session(self):
        r = ChatRequest(question="hello", session_id="s1")
        assert r.session_id == "s1"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="")


# ── ClearContextRequest ──────────────────────────────────────────────────────


class TestClearContextRequest:
    def test_valid(self):
        r = ClearContextRequest(session_id="abc")
        assert r.session_id == "abc"

    def test_empty_session_rejected(self):
        with pytest.raises(ValidationError):
            ClearContextRequest(session_id="")


# ── IngestResponse ────────────────────────────────────────────────────────────


class TestIngestResponse:
    def test_defaults(self):
        r = IngestResponse(filename="test.pdf", num_chunks=5)
        assert r.message == "Document ingested successfully"

    def test_custom_message(self):
        r = IngestResponse(filename="x.txt", num_chunks=1, message="ok")
        assert r.message == "ok"


# ── AnswerResponse ────────────────────────────────────────────────────────────


class TestAnswerResponse:
    def test_minimal(self):
        r = AnswerResponse(answer="42")
        assert r.answer == "42"
        assert r.source_documents == []

    def test_with_sources(self):
        r = AnswerResponse(answer="ok", source_documents=["snippet1"])
        assert len(r.source_documents) == 1


# ── ChatResponse ──────────────────────────────────────────────────────────────


class TestChatResponse:
    def test_full(self):
        r = ChatResponse(
            answer="hi",
            session_id="s1",
            intent="follow_up",
            source_documents=["a"],
            turn_number=3,
        )
        assert r.session_id == "s1"
        assert r.intent == "follow_up"
        assert r.turn_number == 3

    def test_minimal(self):
        r = ChatResponse(
            answer="hi",
            session_id="s1",
            intent="new_topic",
            turn_number=0,
        )
        assert r.source_documents == []


# ── ClearContextResponse ─────────────────────────────────────────────────────


class TestClearContextResponse:
    def test_default_message(self):
        r = ClearContextResponse(session_id="s1")
        assert "cleared" in r.message.lower()


# ── SessionListResponse ──────────────────────────────────────────────────────


class TestSessionListResponse:
    def test_basic(self):
        r = SessionListResponse(sessions=["a", "b"], count=2)
        assert r.count == 2
        assert len(r.sessions) == 2

    def test_empty(self):
        r = SessionListResponse(sessions=[], count=0)
        assert r.count == 0
