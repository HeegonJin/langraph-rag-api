"""FastAPI application – RAG over your documents with llama.cpp.

Supports both single-turn (/ask) and multi-turn (/chat) conversation modes.
Multi-turn features include:
  • Session-based conversation memory via Redis (대화 맥락 유지)
  • Intent classification (인텐트 분류) – follow_up / new_topic / clear_context
  • Query contextualization using chat history (질문 의도 파악)
  • Document deduplication across turns (정보 중복 처리 최소화)
  • Chain-of-Thought prompting for better reasoning
  • Explicit context clearing (Clear Context / 대화 맥락 초기화)
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from app.config import UPLOAD_DIR
from app.models import (
    AnswerResponse,
    ChatRequest,
    ChatResponse,
    ClearContextRequest,
    ClearContextResponse,
    ClearDocumentsResponse,
    IngestResponse,
    QuestionRequest,
    SessionListResponse,
)
from app.rag.conversation import conversation_store
from app.rag.graph import rag_graph
from app.rag.ingestion import clear_all_documents, ingest_file
from app.rag.multiturn_graph import multiturn_rag_graph
from app.rag.sample_ingest import auto_ingest_sample_data
from app.rag.tracing import flush as langfuse_flush
from app.rag.tracing import invoke_graph_with_tracing

logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated on_event) ──────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup / shutdown lifecycle for the FastAPI application."""
    # Startup: auto-ingest sample data (run in thread to avoid blocking)
    await asyncio.to_thread(auto_ingest_sample_data)
    yield
    # Shutdown: flush Langfuse
    langfuse_flush()


app = FastAPI(
    title="LangGraph Multi-Turn RAG API",
    description=(
        "Upload documents, then ask questions in single-turn or multi-turn mode.\n\n"
        "**Multi-turn features:**\n"
        "- 🔄 Session-based conversation memory (Redis)\n"
        "- 🎯 Intent classification (follow-up / new topic / clear context)\n"
        "- 📝 Query contextualization using chat history\n"
        "- 📚 Document deduplication across turns\n"
        "- 🧠 Chain-of-Thought prompting\n"
        "- 🗑️ Explicit context clearing\n\n"
        "Uses llama.cpp (local LLM), Elasticsearch, Docling, LangChain & LangGraph."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ── Document Ingestion ────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest(file: UploadFile) -> IngestResponse:
    """Upload a document (.txt, .md, .pdf) and index it for RAG."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        num_chunks = ingest_file(dest)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(filename=file.filename, num_chunks=num_chunks)


@app.delete("/documents", response_model=ClearDocumentsResponse, tags=["documents"])
async def clear_documents() -> ClearDocumentsResponse:
    """Delete **all** ingested documents from Elasticsearch and uploaded files.

    After clearing, ``auto_ingest_sample_data()`` is re-run so the sample
    documents are available immediately.
    """
    await asyncio.to_thread(clear_all_documents)
    await asyncio.to_thread(auto_ingest_sample_data)
    return ClearDocumentsResponse()


# ── Single-turn (legacy) ─────────────────────────────────────────────────────


@app.post("/ask", response_model=AnswerResponse, tags=["single-turn"])
async def ask(body: QuestionRequest) -> AnswerResponse:
    """Ask a **single-turn** question about the ingested documents.

    This endpoint does *not* maintain conversation history.
    For multi-turn conversations, use ``POST /chat`` instead.
    """
    result = await asyncio.to_thread(
        invoke_graph_with_tracing,
        rag_graph,
        {
            "question": body.question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grounded": False,
            "retries": 0,
        },
        trace_name="rag-single-turn",
        tags=["single-turn"],
        metadata={"question": body.question},
    )

    source_snippets = [doc.page_content[:300] for doc in result.get("documents", [])]

    return AnswerResponse(
        answer=result.get("answer", "No answer generated."),
        source_documents=source_snippets,
    )


# ── Multi-turn Chat ──────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse, tags=["multi-turn"])
async def chat(body: ChatRequest) -> ChatResponse:
    """Ask a question in a **multi-turn** conversation.

    **How it works:**

    1. If ``session_id`` is omitted, a new session is created.
    2. The system classifies the user's **intent** (인텐트 분류):
       - ``follow_up`` – continues the current topic
       - ``new_topic`` – starts a fresh topic (but keeps session)
       - ``clear_context`` – resets the conversation memory
    3. For follow-ups, the question is **rewritten** using conversation history
       so retrieval captures the user's true intent (질문 의도 파악).
    4. Retrieved documents are **deduplicated** against the session's cache
       to minimize redundant searches (정보 중복 처리 최소화).
    5. The answer is generated using **Chain-of-Thought** prompting with full
       conversation context.

    **Example multi-turn flow:**

    ```
    Turn 1: "이번 주말에 상영하는 액션 영화 목록을 알려줘"
    Turn 2: "어벤져스 상영 시간이 궁금해"          (follow-up)
    Turn 3: "다른 주제로 넘어가자"                 (clear_context)
    ```
    """
    # Get or create session
    session = conversation_store.get_or_create(body.session_id)

    result = await asyncio.to_thread(
        invoke_graph_with_tracing,
        multiturn_rag_graph,
        {
            "question": body.question,
            "session_id": session.session_id,
            "intent": "",
            "contextualized_query": "",
            "rewritten_question": "",
            "documents": [],
            "chat_history": "",
            "answer": "",
            "grounded": False,
            "retries": 0,
        },
        trace_name="rag-multi-turn",
        session_id=session.session_id,
        tags=["multi-turn"],
        metadata={"question": body.question, "session_id": session.session_id},
    )

    source_snippets = [doc.page_content[:300] for doc in result.get("documents", [])]

    # Re-load session from Redis to get the updated turn count after save_turn
    updated_session = conversation_store.get(session.session_id)
    turn_count = len(updated_session.turns) // 2 if updated_session else 0

    return ChatResponse(
        answer=result.get("answer", "No answer generated."),
        session_id=session.session_id,
        intent=result.get("intent", "new_topic"),
        source_documents=source_snippets,
        turn_number=turn_count,
    )


@app.post(
    "/chat/clear",
    response_model=ClearContextResponse,
    tags=["multi-turn"],
)
async def clear_chat_context(body: ClearContextRequest) -> ClearContextResponse:
    """Clear the conversation context for a session (대화 맥락 초기화).

    After clearing, the next question will be treated as a brand-new topic.
    The session ID itself remains valid.
    """
    found = conversation_store.clear_session(body.session_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found",
        )
    return ClearContextResponse(session_id=body.session_id)


@app.get("/chat/sessions", response_model=SessionListResponse, tags=["multi-turn"])
async def list_sessions() -> SessionListResponse:
    """List all active conversation sessions."""
    sessions = conversation_store.list_sessions()
    return SessionListResponse(sessions=sessions, count=len(sessions))


@app.delete("/chat/sessions/{session_id}", tags=["multi-turn"])
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a conversation session entirely."""
    found = conversation_store.delete_session(session_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        )
    return {"message": f"Session '{session_id}' deleted", "session_id": session_id}


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
