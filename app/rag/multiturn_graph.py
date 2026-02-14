"""Multi-turn LangGraph RAG workflow.

Extends the basic RAG graph with multi-turn conversation capabilities:

  1. **classify_intent**       – Determine if the query is a follow-up,
                                  a new topic, or a context-clear request
                                  (인텐트 분류)
  2. **contextualize_query**   – Rewrite the user's query by incorporating
                                  conversation history so retrieval works on
                                  the *real* meaning (대화 맥락 유지 / 질문 의도 파악)
  3. **retrieve**              – Fetch relevant chunks; merge with the session's
                                  document cache to avoid redundant searches
                                  (정보 중복 처리 최소화)
  4. **generate**              – Chain-of-thought generation that considers the
                                  full conversation history
  5. **grade**                 – Groundedness check (same as single-turn)
  6. **rewrite**               – Fallback query rewriter when grading fails
  7. **save_turn**             – Persist the turn into session memory
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app import config
from app.rag.constants import NO_DOCS_ANSWER
from app.rag.conversation import Session, conversation_store
from app.rag.ingestion import retrieve_with_scores
from app.rag.tracing import trace_node

# ── Graph state ───────────────────────────────────────────────────────────────


class MultiTurnRAGState(TypedDict, total=False):
    """State that flows through every node in the multi-turn graph."""

    # Input
    question: str
    session_id: str

    # Intermediate
    intent: str  # "follow_up" | "new_topic" | "clear_context"
    contextualized_query: str
    rewritten_question: str
    documents: list[Document]
    chat_history: str

    # Output
    answer: str
    grounded: bool
    retries: int

    # Tracing
    _langfuse_trace: object  # injected by invoke_graph_with_tracing


# ── Helpers ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLAMA_CPP_MODEL,
        base_url=config.LLAMA_CPP_BASE_URL,
        api_key=config.LLAMA_CPP_API_KEY,
        temperature=temperature,
        timeout=config.LLM_TIMEOUT,
    )


# Heuristic patterns that short-circuit the LLM intent classifier
_CLEAR_CONTEXT_PATTERNS = re.compile(
    r"\b(초기화|clear|reset|start\s*over|다시\s*시작|리셋|없던\s*것으로)",
    re.IGNORECASE,
)


def _get_session(state: MultiTurnRAGState) -> Session:
    return conversation_store.get_or_create(state.get("session_id"))


# ── Node: Intent Classification (인텐트 분류) ────────────────────────────────


@trace_node("classify_intent")
def classify_intent(state: MultiTurnRAGState) -> dict:
    """Classify the user's intent given conversation history.

    Categories:
      • follow_up    – The question continues / refines the current topic.
      • new_topic    – The question is about a completely new subject.
      • clear_context – The user wants to reset / start over.
    """
    session = _get_session(state)
    history = session.get_history_text(max_turns=5)
    question = state["question"]

    # If there's no history, it's always a new topic
    if not history.strip():
        return {
            "intent": "new_topic",
            "chat_history": "",
        }

    # Heuristic fast-path: skip LLM call for obvious clear_context requests
    if _CLEAR_CONTEXT_PATTERNS.search(question):
        session.last_intent = "clear_context"
        return {
            "intent": "clear_context",
            "chat_history": history,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intent classifier for a multi-turn conversation.\n"
                "Given the conversation history and the user's latest message, "
                "classify the intent as exactly one of:\n"
                "  follow_up    – continues or refines the current topic\n"
                "  new_topic    – completely new subject\n"
                "  clear_context – user explicitly wants to reset / start over\n\n"
                "## Examples\n"
                "History: User: 이번 주말 액션 영화 알려줘 / Assistant: 어벤져스, ...\n"
                "User: 어벤져스 상영 시간 알려줘\n"
                "→ follow_up\n\n"
                "History: User: What is RAG? / Assistant: RAG stands for...\n"
                "User: 오늘 날씨 어때?\n"
                "→ new_topic\n\n"
                "History: User: Tell me about Python / Assistant: Python is...\n"
                "User: 초기화해줘\n"
                "→ clear_context\n\n"
                "Reply with ONLY one of: follow_up, new_topic, clear_context\n\n"
                "Conversation history:\n{history}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    result = chain.invoke({"history": history, "question": state["question"]})
    intent_raw = result.content.strip().lower().replace(" ", "_")

    # Normalise
    if "clear" in intent_raw:
        intent = "clear_context"
    elif "follow" in intent_raw:
        intent = "follow_up"
    else:
        intent = "new_topic"

    session.last_intent = intent
    return {
        "intent": intent,
        "chat_history": history,
    }


# ── Node: Contextualize Query (대화 맥락 유지 + 질문 재작성) ─────────────────


@trace_node("contextualize_query")
def contextualize_query(state: MultiTurnRAGState) -> dict:
    """Rewrite the user question to be *self-contained* using chat history.

    For a follow-up like "그 영화 상영 시간은?" after talking about 어벤져스,
    this node produces: "어벤져스 상영 시간은 언제인가요?"
    """
    history = state.get("chat_history", "")
    question = state["question"]
    intent = state.get("intent", "new_topic")

    # New topic → no rewriting needed
    if intent == "new_topic" or not history.strip():
        return {"contextualized_query": question}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that rewrites user questions to be "
                "self-contained. Given the conversation history and the latest "
                "question, rewrite the question so that it can be understood "
                "WITHOUT the conversation history. Keep it concise.\n"
                "Do NOT answer the question — only rewrite it.\n\n"
                "Conversation history:\n{history}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    result = chain.invoke({"history": history, "question": question})
    return {"contextualized_query": result.content.strip()}


# ── Node: Clear Context (대화 맥락 초기화) ───────────────────────────────────


@trace_node("clear_context")
def clear_context(state: MultiTurnRAGState) -> dict:
    """Reset the session's memory and return a confirmation message."""
    session = _get_session(state)
    session.clear()
    return {
        "answer": "대화 맥락이 초기화되었습니다. 새로운 주제로 질문해 주세요! "
        "(Conversation history and document cache have been cleared. "
        "Please start a new topic.)",
        "documents": [],
        "grounded": True,
    }


# ── Node: Retrieve (문서 검색 + 중복 최소화) ─────────────────────────────────


@trace_node("retrieve")
def retrieve(state: MultiTurnRAGState) -> dict:
    """Fetch relevant documents, merging with session cache for dedup.

    Uses score-threshold filtering so that irrelevant chunks are dropped.
    """
    query = (
        state.get("rewritten_question")
        or state.get("contextualized_query")
        or state["question"]
    )

    new_docs = retrieve_with_scores(query)

    # Merge with session's document cache (정보 중복 처리 최소화)
    session = _get_session(state)
    intent = state.get("intent", "new_topic")

    if intent == "follow_up":
        merged = session.update_cached_documents(new_docs)
    else:
        session.cached_documents = new_docs
        merged = new_docs

    return {"documents": merged}


# ── Node: Generate with Chain-of-Thought ─────────────────────────────────────


@trace_node("generate")
def generate(state: MultiTurnRAGState) -> dict:
    """Generate an answer using Chain-of-Thought prompting.

    The prompt is designed so the LLM:
      1. Thinks step-by-step about what the user needs (Chain-of-Thought)
      2. Considers the conversation history for context
      3. Uses ONLY the retrieved documents as its knowledge source

    If no relevant documents were retrieved, returns a canned answer
    immediately without calling the LLM.
    """
    # No documents → nothing to generate from; skip LLM call
    if not state.get("documents"):
        return {"answer": NO_DOCS_ANSWER}

    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    history = state.get("chat_history", "")
    question = state["question"]

    # Escape curly braces so ChatPromptTemplate doesn't treat them as variables
    context_escaped = context.replace("{", "{{").replace("}", "}}")
    history_escaped = history.replace("{", "{{").replace("}", "}}")

    # Build system prompt with Chain-of-Thought instructions
    system_msg = (
        "You are a helpful assistant engaged in a multi-turn conversation.\n\n"
        "## Instructions\n"
        "1. First, review the conversation history to understand the context.\n"
        "2. Then, think step-by-step about the user's question and what they "
        "really need (consider their underlying intent).\n"
        "3. Use ONLY the retrieved documents below to formulate your answer.\n"
        "4. If the documents do not contain enough information, say so honestly.\n"
        "5. Be concise but thorough.\n"
        "6. **Always respond in the same language the user used in their question.**\n\n"
    )

    if history_escaped.strip():
        system_msg += f"## Conversation History\n{history_escaped}\n\n"

    system_msg += f"## Retrieved Documents\n{context_escaped}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    result = chain.invoke({"question": question})
    return {"answer": result.content}


# ── Node: Grade (groundedness check) ─────────────────────────────────────────


@trace_node("grade")
def grade(state: MultiTurnRAGState) -> dict:
    """Decide whether the answer is grounded in the retrieved documents.

    If no documents were retrieved:
      • First attempt → trigger a rewrite & retry cycle (the rephrased
        query may find relevant chunks the original missed).
      • After retry → accept the canned “no docs” answer and stop.
    """
    if not state.get("documents"):
        retries = state.get("retries", 0)
        # Allow one rewrite+retry before giving up
        if retries == 0:
            return {"grounded": False, "retries": retries + 1}
        return {"grounded": True, "retries": retries + 1}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grading assistant. Given some context documents and an "
                "answer, determine if the answer is grounded in the documents.\n\n"
                "A grounded answer means:\n"
                "- The main claims are supported by the document content\n"
                "- No significant facts are fabricated beyond what the documents state\n\n"
                "Reply with ONLY 'yes' or 'no'.\n\n"
                "Documents:\n{documents}\n\nAnswer:\n{answer}",
            ),
            ("human", "Is the answer grounded in the documents?"),
        ]
    )

    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    chain = prompt | _llm()
    result = chain.invoke({"documents": context, "answer": state["answer"]})

    grounded = "yes" in result.content.strip().lower()
    return {"grounded": grounded, "retries": state.get("retries", 0) + 1}


# ── Node: Rewrite query (retrieval fallback) ─────────────────────────────────


@trace_node("rewrite")
def rewrite(state: MultiTurnRAGState) -> dict:
    """Rewrite the question to improve retrieval on retry."""
    # Gather previously retrieved documents for context
    prev_docs = state.get("documents", [])
    doc_summary = ""
    if prev_docs:
        doc_summary = (
            "\n\nThe following documents were retrieved but were NOT sufficient "
            "to answer the question:\n"
            + "\n---\n".join(d.page_content[:200] for d in prev_docs[:3])
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question rewriter. Given the original question and "
                "conversation context, rewrite the question to be more specific "
                "so better documents can be found. Output ONLY the rewritten question."
                + doc_summary,
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    q = state.get("contextualized_query") or state["question"]
    result = chain.invoke({"question": q})
    return {"rewritten_question": result.content}


# ── Node: Save turn to memory ────────────────────────────────────────────────


@trace_node("save_turn")
def save_turn(state: MultiTurnRAGState) -> dict:
    """Persist the Q&A turn into session memory."""
    session = _get_session(state)
    session.add_human_turn(state["question"])
    session.add_assistant_turn(state.get("answer", ""))
    return {}


# ── Conditional edges ─────────────────────────────────────────────────────────


def route_by_intent(state: MultiTurnRAGState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "new_topic")
    if intent == "clear_context":
        return "clear_context"
    return "contextualize"


def should_retry(state: MultiTurnRAGState) -> str:
    """Route to 'rewrite' if the answer was not grounded and retries remain."""
    if state.get("grounded"):
        return "save"
    if state.get("retries", 0) >= 2:
        return "save"  # give up after max retries
    return "retry"


# ── Build the multi-turn graph ────────────────────────────────────────────────


def build_multiturn_rag_graph() -> StateGraph:
    """
    Graph topology:

        classify_intent
            ├─ clear_context ─→ END
            └─ contextualize_query
                    ↓
                retrieve
                    ↓
                generate  (no-docs → canned answer, skip LLM)
                    ↓
                  grade    (no-docs → auto-grounded, skip LLM)
                  ├─ save_turn ─→ END  (grounded / max retries)
                  └─ rewrite ─→ retrieve  (loop back)
    """
    graph = StateGraph(MultiTurnRAGState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("contextualize", contextualize_query)
    graph.add_node("clear_context", clear_context)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("save_turn", save_turn)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Intent routing
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "clear_context": "clear_context",
            "contextualize": "contextualize",
        },
    )

    # Clear context exits immediately
    graph.add_edge("clear_context", END)

    # Main flow
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")

    # Grading conditional
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {
            "save": "save_turn",
            "retry": "rewrite",
        },
    )
    graph.add_edge("rewrite", "retrieve")  # loop back
    graph.add_edge("save_turn", END)

    return graph.compile()


# Module-level compiled graph
multiturn_rag_graph = build_multiturn_rag_graph()
