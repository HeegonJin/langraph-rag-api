"""Tests for LangGraph graph structure (single-turn & multi-turn).

These tests verify the graph topology (nodes, edges) without invoking
any LLM calls.
"""

from langgraph.graph import END

from app.rag.graph import build_rag_graph, rag_graph, should_retry
from app.rag.multiturn_graph import (
    build_multiturn_rag_graph,
    multiturn_rag_graph,
    route_by_intent,
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
