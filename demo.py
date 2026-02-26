"""Streamlit demo – full-featured RAG chat with your documents via the LangGraph RAG pipeline.

Supports:
  • Document upload & ingestion (+ auto-ingest sample_data/)
  • Multi-turn conversation with intent classification & CoT reasoning
  • Single-turn (one-shot) question mode
  • Session management (list / switch / clear / delete)
  • Langfuse tracing (transparent if keys are set)
"""

import logging

import streamlit as st

from app.config import UPLOAD_DIR
from app.rag.conversation import conversation_store
from app.rag.graph import rag_graph
from app.rag.ingestion import clear_all_documents, ingest_file
from app.rag.multiturn_graph import multiturn_rag_graph
from app.rag.sample_ingest import auto_ingest_sample_data
from app.rag.tracing import invoke_graph_with_tracing

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="LangGraph RAG Demo", page_icon="📚", layout="centered")
st.title("📚 LangGraph RAG Demo")
st.caption(
    "Upload documents, then ask questions — supports **single-turn** and "
    "**multi-turn** conversation with context tracking, intent classification "
    "& chain-of-thought reasoning"
)

# ── Session management ────────────────────────────────────────────────────────

if "rag_session_id" not in st.session_state:
    session = conversation_store.get_or_create()
    st.session_state.rag_session_id = session.session_id

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "multi-turn"

if "sample_ingested" not in st.session_state:
    st.session_state.sample_ingested = False


# ── Auto-ingest sample_data/ on first run ────────────────────────────────────

if not st.session_state.sample_ingested:
    with st.spinner("Auto-ingesting sample data..."):
        auto_ingest_sample_data()
    st.session_state.sample_ingested = True

# ── Sidebar: document upload & session controls ──────────────────────────────

with st.sidebar:
    # ── Chat mode selector ────────────────────────────────────────────────────
    st.header("⚙️ Mode")
    mode = st.radio(
        "Chat mode",
        ["multi-turn", "single-turn"],
        index=0 if st.session_state.chat_mode == "multi-turn" else 1,
        help="**Multi-turn**: maintains conversation history & intent classification.\n"
        "**Single-turn**: no history, each question is independent.",
    )
    if mode != st.session_state.chat_mode:
        st.session_state.chat_mode = mode
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Document upload ───────────────────────────────────────────────────────
    st.header("📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest", type="primary"):
        for uploaded_file in uploaded_files:
            dest = UPLOAD_DIR / uploaded_file.name
            dest.write_bytes(uploaded_file.getvalue())
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    num_chunks = ingest_file(dest)
                    st.success(f"**{uploaded_file.name}** → {num_chunks} chunks")
                except Exception as exc:
                    st.error(f"Failed to ingest {uploaded_file.name}: {exc}")

    st.divider()

    # ── Session controls (multi-turn only) ────────────────────────────────────
    st.header("💬 Session")

    if st.session_state.chat_mode == "multi-turn":
        # Show current session
        st.text(f"Active: {st.session_state.rag_session_id[:12]}…")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", help="대화 맥락 초기화 – Reset conversation memory"):
                conversation_store.clear_session(st.session_state.rag_session_id)
                st.session_state.messages = []
                st.success("Context cleared!")
                st.rerun()
        with col2:
            if st.button("🆕 New", help="Start a completely new session"):
                with st.spinner("Clearing documents & starting fresh..."):
                    clear_all_documents()
                    auto_ingest_sample_data()
                session = conversation_store.get_or_create()
                st.session_state.rag_session_id = session.session_id
                st.session_state.messages = []
                st.info(f"New: {session.session_id[:12]}…")
                st.rerun()

        # ── List / switch / delete sessions ───────────────────────────────────
        all_sessions = conversation_store.list_sessions()
        if all_sessions:
            st.markdown(f"**Active sessions** ({len(all_sessions)})")
            for sid in all_sessions:
                label = sid[:12] + "…"
                is_current = sid == st.session_state.rag_session_id
                c1, c2, c3 = st.columns([5, 2, 2])
                with c1:
                    if is_current:
                        st.markdown(f"▸ **{label}**")
                    else:
                        st.text(f"  {label}")
                with c2:
                    if not is_current and st.button("Switch", key=f"sw_{sid}"):
                        st.session_state.rag_session_id = sid
                        st.session_state.messages = []
                        st.rerun()
                with c3:
                    if st.button("🗑", key=f"del_{sid}"):
                        conversation_store.delete_session(sid)
                        if is_current:
                            new_s = conversation_store.get_or_create()
                            st.session_state.rag_session_id = new_s.session_id
                            st.session_state.messages = []
                        st.rerun()
    else:
        st.info("Single-turn mode — no session history.")

    st.divider()
    st.markdown(
        "**How it works (멀티턴 RAG):**\n"
        "1. Upload & ingest documents\n"
        "2. Pick **single-turn** or **multi-turn** mode\n"
        "3. Multi-turn detects **intent** (follow-up / new topic)\n"
        "4. Follow-up questions are **rewritten** with context\n"
        "5. Documents are **deduplicated** across turns\n"
        "6. **Chain-of-Thought** reasoning generates the answer\n"
        "7. Use *Clear* to reset conversation context"
    )

# ── Chat interface ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mode badge
if st.session_state.chat_mode == "multi-turn":
    st.info(f"🔄 **Multi-turn** mode | Session: `{st.session_state.rag_session_id[:12]}…`")
else:
    st.info("⚡ **Single-turn** mode — each question is independent (no history)")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("intent"):
            st.caption(f"🎯 Intent: {msg['intent']}")
        if msg.get("mode"):
            st.caption(f"Mode: {msg['mode']}")
        if msg.get("sources"):
            with st.expander("📎 Source snippets"):
                for i, snippet in enumerate(msg["sources"], 1):
                    st.text(f"[{i}] {snippet}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.chat_mode == "multi-turn":
            # ── Multi-turn RAG ────────────────────────────────────────────
            with st.spinner("Thinking (intent → contextualize → retrieve → generate)..."):
                result = invoke_graph_with_tracing(
                    multiturn_rag_graph,
                    {
                        "question": prompt,
                        "session_id": st.session_state.rag_session_id,
                        "intent": "",
                        "contextualized_query": "",
                        "rewritten_question": "",
                        "documents": [],
                        "chat_history": "",
                        "answer": "",
                        "grounded": False,
                        "retries": 0,
                    },
                    trace_name="rag-multi-turn-streamlit",
                    session_id=st.session_state.rag_session_id,
                    tags=["multi-turn", "streamlit"],
                    metadata={"question": prompt},
                )

                answer = result.get("answer", "No answer generated.")
                intent = result.get("intent", "new_topic")
                sources = [doc.page_content[:300] for doc in result.get("documents", [])]
                grounded = result.get("grounded", False)

            st.markdown(answer)
            st.caption(f"🎯 Intent: {intent}")

            if not grounded:
                st.warning("⚠️ The answer may not be fully grounded in the documents.")

            if sources:
                with st.expander("📎 Source snippets"):
                    for i, snippet in enumerate(sources, 1):
                        st.text(f"[{i}] {snippet}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "intent": intent,
                    "mode": "multi-turn",
                }
            )

        else:
            # ── Single-turn RAG ───────────────────────────────────────────
            with st.spinner("Thinking (retrieve → generate → grade)..."):
                result = invoke_graph_with_tracing(
                    rag_graph,
                    {
                        "question": prompt,
                        "rewritten_question": "",
                        "documents": [],
                        "answer": "",
                        "grounded": False,
                        "retries": 0,
                    },
                    trace_name="rag-single-turn-streamlit",
                    tags=["single-turn", "streamlit"],
                    metadata={"question": prompt},
                )

                answer = result.get("answer", "No answer generated.")
                sources = [doc.page_content[:300] for doc in result.get("documents", [])]
                grounded = result.get("grounded", False)

            st.markdown(answer)

            if not grounded:
                st.warning("⚠️ The answer may not be fully grounded in the documents.")

            if sources:
                with st.expander("📎 Source snippets"):
                    for i, snippet in enumerate(sources, 1):
                        st.text(f"[{i}] {snippet}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "mode": "single-turn",
                }
            )
