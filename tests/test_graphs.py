"""Tests for LangGraph graph structure (single-turn & multi-turn).

These tests verify the graph topology (nodes, edges) without invoking
any LLM calls, plus unit tests for individual node functions.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.graph import build_rag_graph, rag_graph, should_retry
from app.rag.graph import generate as st_generate
from app.rag.multiturn_graph import (
    build_multiturn_rag_graph,
    multiturn_rag_graph,
    route_by_intent,
)
from app.rag.multiturn_graph import (
    generate as mt_generate,
)
from app.rag.multiturn_graph import (
    should_retry as mt_should_retry,
)

# ── Single-turn graph ────────────────────────────────────────────────────────


class TestSingleTurnGraph:
    def test_compiles(self):
        """The graph compiles without error."""
        g = build_rag_graph()
        assert g is not None

    def test_has_expected_nodes(self):
        nodes = set(rag_graph.nodes.keys()) - {"__start__"}
        assert nodes == {"retrieve", "generate", "grade", "rewrite"}

    def test_should_retry_grounded(self):
        assert should_retry({"grounded": True, "retries": 1}) == "finish"

    def test_should_retry_max_retries(self):
        assert should_retry({"grounded": False, "retries": 2}) == "finish"

    def test_should_retry_needs_retry(self):
        assert should_retry({"grounded": False, "retries": 0}) == "retry"

    def test_should_retry_default_retries(self):
        assert should_retry({"grounded": False}) == "retry"


# ── Multi-turn graph ─────────────────────────────────────────────────────────


class TestMultiTurnGraph:
    def test_compiles(self):
        g = build_multiturn_rag_graph()
        assert g is not None

    def test_has_expected_nodes(self):
        nodes = set(multiturn_rag_graph.nodes.keys()) - {"__start__"}
        expected = {
            "classify_intent",
            "contextualize",
            "clear_context",
            "retrieve",
            "generate",
            "grade",
            "rewrite",
            "save_turn",
        }
        assert nodes == expected

    def test_route_by_intent_clear(self):
        assert route_by_intent({"intent": "clear_context"}) == "clear_context"

    def test_route_by_intent_follow_up(self):
        assert route_by_intent({"intent": "follow_up"}) == "contextualize"

    def test_route_by_intent_new_topic(self):
        assert route_by_intent({"intent": "new_topic"}) == "contextualize"

    def test_route_by_intent_default(self):
        assert route_by_intent({}) == "contextualize"

    def test_mt_should_retry_grounded(self):
        assert mt_should_retry({"grounded": True}) == "save"

    def test_mt_should_retry_max(self):
        assert mt_should_retry({"grounded": False, "retries": 2}) == "save"

    def test_mt_should_retry_needs_retry(self):
        assert mt_should_retry({"grounded": False, "retries": 0}) == "retry"


# ── Node function tests (curly-brace escaping) ───────────────────────────────


def _mock_llm_response(content: str):
    """Return a mock LLM that always produces the given content."""
    mock_result = MagicMock()
    mock_result.content = content
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result
    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: mock_chain  # prompt | llm
    # ChatPromptTemplate | llm  → need to patch __or__ on prompt side
    return mock_llm, mock_chain


class TestGenerateCurlyBraceEscaping:
    """Regression tests: documents/history containing { } must not crash."""

    @patch("app.rag.multiturn_graph._llm")
    def test_mt_generate_with_curly_braces_in_docs(self, mock_llm_factory):
        """Multi-turn generate should not raise KeyError for {name} in docs."""
        mock_result = MagicMock()
        mock_result.content = "The team members are listed in the document."
        mock_chain = MagicMock()
        mock_chain.return_value = mock_result
        mock_chain.invoke.return_value = mock_result
        mock_llm_factory.return_value = mock_chain

        state = {
            "question": "Who are the team members?",
            "documents": [
                Document(page_content="Team: {heegon, deftson, hbk5844}"),
            ],
            "chat_history": "",
        }

        result = mt_generate(state)
        assert result["answer"] == "The team members are listed in the document."

    @patch("app.rag.multiturn_graph._llm")
    def test_mt_generate_with_curly_braces_in_history(self, mock_llm_factory):
        """Curly braces in chat history should also be escaped."""
        mock_result = MagicMock()
        mock_result.content = "Yes, those are the members."
        mock_chain = MagicMock()
        mock_chain.return_value = mock_result
        mock_chain.invoke.return_value = mock_result
        mock_llm_factory.return_value = mock_chain

        state = {
            "question": "Can you confirm?",
            "documents": [Document(page_content="Some content")],
            "chat_history": "User: List members\nAssistant: {alice, bob}",
        }

        result = mt_generate(state)
        assert result["answer"] == "Yes, those are the members."

    @patch("app.rag.multiturn_graph._llm")
    def test_mt_generate_normal_docs_still_works(self, mock_llm_factory):
        """Ensure normal documents without braces still work correctly."""
        mock_result = MagicMock()
        mock_result.content = "Answer from docs."
        mock_chain = MagicMock()
        mock_chain.return_value = mock_result
        mock_chain.invoke.return_value = mock_result
        mock_llm_factory.return_value = mock_chain

        state = {
            "question": "What is RAG?",
            "documents": [
                Document(page_content="RAG is retrieval augmented generation."),
            ],
            "chat_history": "",
        }

        result = mt_generate(state)
        assert result["answer"] == "Answer from docs."

    @patch("app.rag.graph._llm")
    def test_st_generate_with_curly_braces_in_docs(self, mock_llm_factory):
        """Single-turn generate passes context via template vars — should be safe."""
        mock_result = MagicMock()
        mock_result.content = "The set is described."
        mock_chain = MagicMock()
        mock_chain.return_value = mock_result
        mock_chain.invoke.return_value = mock_result
        mock_llm_factory.return_value = mock_chain

        state = {
            "question": "Describe the set",
            "documents": [
                Document(page_content="Members: {alice, bob, charlie}"),
            ],
        }

        result = st_generate(state)
        assert result["answer"] == "The set is described."
