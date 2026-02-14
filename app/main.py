"""FastAPI application – RAG over your documents with llama.cpp.

Supports both single-turn (/ask) and multi-turn (/chat) conversation modes.
Multi-turn features include:
  • Session-based conversation memory (대화 맥락 유지)
  • Intent classification (인텐트 분류) – follow_up / new_topic / clear_context
  • Query contextualization using chat history (질문 의도 파악)
  • Document deduplication across turns (정보 중복 처리 최소화)
  • Chain-of-Thought prompting for better reasoning
  • Explicit context clearing (Clear Context / 대화 맥락 초기화)
"""

import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import RedirectResponse

from app.config import SAMPLE_DATA_DIR, UPLOAD_DIR
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
from app.rag.conversation import conversation_store
from app.rag.graph import rag_graph
from app.rag.ingestion import ingest_file
from app.rag.multiturn_graph import multiturn_rag_graph
from app.rag.tracing import flush as langfuse_flush, invoke_graph_with_tracing

app = FastAPI(
    title="LangGraph Multi-Turn RAG API",
    description=(
        "Upload documents, then ask questions in single-turn or multi-turn mode.\n\n"
        "**Multi-turn features:**\n"
        "- 🔄 Session-based conversation memory\n"
        "- 🎯 Intent classification (follow-up / new topic / clear context)\n"
        "- 📝 Query contextualization using chat history\n"
        "- 📚 Document deduplication across turns\n"
        "- 🧠 Chain-of-Thought prompting\n"
        "- 🗑️ Explicit context clearing\n\n"
        "Uses llama.cpp (local LLM), ChromaDB, LangChain & LangGraph."
    ),
    version="0.2.0",
)


@app.on_event("shutdown")
async def shutdown_event():
    """Flush buffered Langfuse events on server shutdown."""
    langfuse_flush()


@app.on_event("startup")
async def ingest_sample_data():
    """Auto-ingest files from sample_data/ on first startup.

    Skips files that have already been ingested (tracked via a marker file
    in chroma_data/).
    """
    logger = logging.getLogger("app.startup")
    marker = Path("chroma_data/.sample_ingested")

    if marker.exists():
        logger.info("Sample data already ingested – skipping")
        return

    if not SAMPLE_DATA_DIR.is_dir():
        logger.info("No sample_data/ directory found – skipping")
        return

    supported = {".pdf", ".txt", ".md", ".csv"}
    files = [f for f in SAMPLE_DATA_DIR.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.info("No supported files in sample_data/ – skipping")
        return

    total_chunks = 0
    for filepath in files:
        try:
            n = ingest_file(filepath)
            total_chunks += n
            logger.info("Ingested %s → %d chunks", filepath.name, n)
        except Exception:
            logger.exception("Failed to ingest sample file %s", filepath.name)

    # Write marker so we don't re-ingest on next restart
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"Ingested {len(files)} files, {total_chunks} chunks\n")
    logger.info(
        "Sample data ingestion complete: %d files, %d chunks", len(files), total_chunks
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ── Document Ingestion ────────────────────────────────────────────────────────


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest(file: UploadFile):
    """Upload a document (.txt, .md, .pdf) and index it for RAG."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        num_chunks = ingest_file(dest)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return IngestResponse(filename=file.filename, num_chunks=num_chunks)


# ── Single-turn (legacy) ─────────────────────────────────────────────────────


@app.post("/ask", response_model=AnswerResponse, tags=["single-turn"])
async def ask(body: QuestionRequest):
    """Ask a **single-turn** question about the ingested documents.

    This endpoint does *not* maintain conversation history.
    For multi-turn conversations, use ``POST /chat`` instead.
    """
    result = invoke_graph_with_tracing(
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

    source_snippets = [
        doc.page_content[:300] for doc in result.get("documents", [])
    ]

    return AnswerResponse(
        answer=result.get("answer", "No answer generated."),
        source_documents=source_snippets,
    )


# ── Multi-turn Chat ──────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse, tags=["multi-turn"])
async def chat(body: ChatRequest):
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

    result = invoke_graph_with_tracing(
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

    source_snippets = [
        doc.page_content[:300] for doc in result.get("documents", [])
    ]

    return ChatResponse(
        answer=result.get("answer", "No answer generated."),
        session_id=session.session_id,
        intent=result.get("intent", "new_topic"),
        source_documents=source_snippets,
        turn_number=len(session.turns) // 2,  # each Q+A = 1 turn
    )


@app.post(
    "/chat/clear",
    response_model=ClearContextResponse,
    tags=["multi-turn"],
)
async def clear_chat_context(body: ClearContextRequest):
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
async def list_sessions():
    """List all active conversation sessions."""
    sessions = conversation_store.list_sessions()
    return SessionListResponse(sessions=sessions, count=len(sessions))


@app.delete("/chat/sessions/{session_id}", tags=["multi-turn"])
async def delete_session(session_id: str):
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
async def health():
    return {"status": "ok"}
