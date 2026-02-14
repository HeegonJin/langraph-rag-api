"""LangGraph RAG workflow.

The graph has three nodes:

  1. **retrieve**  – fetch relevant document chunks from ChromaDB.
  2. **generate**  – pass the retrieved context + user question to the LLM.
  3. **grade**     – let the LLM decide if the answer is grounded in the
                     documents.  If not, loop back to retrieve with a
                     rewritten query (max 1 retry).

This gives you a taste of conditional edges and cycles in LangGraph while
keeping the project small enough to study.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app import config
from app.rag.ingestion import get_retriever, retrieve_with_scores
from app.rag.tracing import trace_node

_NO_DOCS_ANSWER = (
    "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다. "
    "다른 질문을 하시거나, 관련 문서를 먼저 업로드해 주세요.\n"
    "(No relevant documents were found for your question. "
    "Please try a different question or upload a related document first.)"
)

# ── Graph state ───────────────────────────────────────────────────────────────


class RAGState(TypedDict, total=False):
    """State that flows through every node in the graph."""

    question: str
    rewritten_question: str
    documents: list[Document]
    answer: str
    grounded: bool
    retries: int
    _langfuse_trace: object  # injected by invoke_graph_with_tracing


# ── Helpers ───────────────────────────────────────────────────────────────────


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLAMA_CPP_MODEL,
        base_url=config.LLAMA_CPP_BASE_URL,
        api_key=config.LLAMA_CPP_API_KEY,
        temperature=0,
    )


# ── Node functions ────────────────────────────────────────────────────────────


@trace_node("retrieve")
def retrieve(state: RAGState) -> dict:
    """Fetch relevant documents from the vector store.

    Uses score-threshold filtering so irrelevant chunks are dropped.
    """
    query = state.get("rewritten_question") or state["question"]
    docs = retrieve_with_scores(query)
    return {"documents": docs}


@trace_node("generate")
def generate(state: RAGState) -> dict:
    """Generate an answer grounded in the retrieved documents.

    If no relevant documents were retrieved, returns a canned answer
    immediately without calling the LLM.
    """
    # No documents → nothing to generate from; skip LLM call
    if not state.get("documents"):
        return {"answer": _NO_DOCS_ANSWER}

    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the context below. If the context does not contain the answer, say "
                '"I don\'t have enough information to answer that."\n\n'
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    result = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": result.content}


@trace_node("grade")
def grade(state: RAGState) -> dict:
    """Decide whether the answer is grounded in the retrieved documents.

    If no documents were retrieved, the answer is automatically marked
    as grounded (nothing to contradict) and no retry is triggered.
    """
    # No documents → nothing to grade against; accept the answer as-is
    if not state.get("documents"):
        return {"grounded": True, "retries": state.get("retries", 0) + 1}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grading assistant. Given some context documents and an "
                "answer, reply with ONLY 'yes' or 'no' — is the answer grounded in "
                "the documents?\n\nDocuments:\n{documents}\n\nAnswer:\n{answer}",
            ),
            ("human", "Is the answer grounded?"),
        ]
    )

    context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
    chain = prompt | _llm()
    result = chain.invoke({"documents": context, "answer": state["answer"]})

    grounded = "yes" in result.content.strip().lower()
    return {"grounded": grounded, "retries": state.get("retries", 0) + 1}


@trace_node("rewrite")
def rewrite(state: RAGState) -> dict:
    """Rewrite the question to improve retrieval on the next attempt."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question rewriter. Given the original question and some "
                "context that was NOT useful, rewrite the question to be more specific "
                "so better documents can be found. Output ONLY the rewritten question.",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | _llm()
    result = chain.invoke({"question": state["question"]})
    return {"rewritten_question": result.content}


# ── Conditional edge ──────────────────────────────────────────────────────────


def should_retry(state: RAGState) -> str:
    """Route to 'rewrite' if the answer was not grounded and retries remain."""
    if state.get("grounded"):
        return "finish"
    if state.get("retries", 0) >= 2:
        return "finish"  # give up after max retries
    return "retry"


# ── Build the graph ───────────────────────────────────────────────────────────


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {
            "retry": "rewrite",
            "finish": END,
        },
    )
    graph.add_edge("rewrite", "retrieve")  # loop back

    return graph.compile()


# Module-level compiled graph, ready to invoke
rag_graph = build_rag_graph()
