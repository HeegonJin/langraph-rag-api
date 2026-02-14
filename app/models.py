"""Pydantic models for request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────


class QuestionRequest(BaseModel):
    """Single-turn question (legacy endpoint)."""

    question: str = Field(..., min_length=1, examples=["What is this document about?"])


class ChatRequest(BaseModel):
    """Multi-turn chat request.

    If *session_id* is omitted a new session is created automatically.
    """

    question: str = Field(..., min_length=1, examples=["이번 주말에 상영하는 액션 영화 목록을 알려줘"])
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Omit to start a new conversation.",
        examples=["abc123"],
    )


class ClearContextRequest(BaseModel):
    """Request to clear conversation context (대화 맥락 초기화)."""

    session_id: str = Field(..., min_length=1, examples=["abc123"])


# ── Responses ─────────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    filename: str
    num_chunks: int
    message: str = "Document ingested successfully"


class AnswerResponse(BaseModel):
    """Single-turn answer (legacy endpoint)."""

    answer: str
    source_documents: list[str] = Field(
        default_factory=list,
        description="Relevant text snippets used to generate the answer",
    )


class ChatResponse(BaseModel):
    """Multi-turn chat response."""

    answer: str
    session_id: str = Field(description="Session ID to use for follow-up questions")
    intent: str = Field(
        description="Detected intent: follow_up | new_topic | clear_context",
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="Relevant text snippets used to generate the answer",
    )
    turn_number: int = Field(description="Current turn number in the conversation")


class ClearContextResponse(BaseModel):
    """Response after clearing conversation context."""

    session_id: str
    message: str = "Conversation context cleared successfully"


class SessionListResponse(BaseModel):
    """List of active session IDs."""

    sessions: list[str]
    count: int
