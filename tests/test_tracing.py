"""Tests for Langfuse tracing helpers.

All tests run with Langfuse DISABLED (no network calls) to verify
graceful degradation and the @trace_node decorator logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.tracing import (
    _summarise_output,
    _summarise_state,
    trace_node,
)

# ── _summarise_state ──────────────────────────────────────────────────────────


class TestSummariseState:
    def test_skips_internal_keys(self):
        state = {"_langfuse_trace": "secret", "question": "hi"}
        result = _summarise_state(state, for_input=True)
        assert "_langfuse_trace" not in result
        assert result["question"] == "hi"

    def test_documents_summarised(self):
        from langchain_core.documents import Document

        content = "hello world " * 50
        docs = [Document(page_content=content)]
        state = {"documents": docs, "question": "q"}
        result = _summarise_state(state, for_input=True)
        assert result["documents_count"] == 1
        assert "documents_preview" in result
        # Full content preserved – no truncation
        assert result["documents_preview"][0] == content

    def test_empty_documents(self):
        state = {"documents": [], "question": "q"}
        result = _summarise_state(state, for_input=True)
        assert result["documents_count"] == 0

    def test_long_string_fully_preserved(self):
        """No truncation – even very long strings are kept intact."""
        long = "x" * 50_000
        state = {"answer": long}
        result = _summarise_state(state, for_input=True)
        assert result["answer"] == long

    def test_primitives_preserved(self):
        state = {"grounded": True, "retries": 2, "question": "hi"}
        result = _summarise_state(state, for_input=True)
        assert result["grounded"] is True
        assert result["retries"] == 2
        assert result["question"] == "hi"

    def test_none_preserved(self):
        state = {"answer": None}
        result = _summarise_state(state, for_input=True)
        assert result["answer"] is None

    def test_list_summarised(self):
        state = {"tags": ["a", "b", "c"]}
        result = _summarise_state(state, for_input=True)
        assert result["tags"] == "list[3]"


# ── _summarise_output ─────────────────────────────────────────────────────────


class TestSummariseOutput:
    def test_non_dict_returns_raw(self):
        result = _summarise_output("just a string")
        assert "raw" in result

    def test_documents_in_output(self):
        from langchain_core.documents import Document

        output = {"documents": [Document(page_content="abc")]}
        result = _summarise_output(output)
        assert result["documents_count"] == 1

    def test_long_string_fully_preserved(self):
        """No truncation on output either."""
        long = "y" * 50_000
        output = {"answer": long}
        result = _summarise_output(output)
        assert result["answer"] == long

    def test_primitives(self):
        output = {"grounded": False, "retries": 1}
        result = _summarise_output(output)
        assert result == {"grounded": False, "retries": 1}


# ── @trace_node decorator ────────────────────────────────────────────────────


class TestTraceNodeDecorator:
    def test_no_trace_runs_normally(self):
        """When no _langfuse_trace in state, the node runs as-is."""

        @trace_node("my_node")
        def my_node(state: dict) -> dict:
            return {"answer": state["question"] + "!"}

        result = my_node({"question": "hi"})
        assert result == {"answer": "hi!"}

    def test_with_trace_creates_span(self):
        """When _langfuse_trace is present, a span is created."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        @trace_node("my_node")
        def my_node(state: dict) -> dict:
            return {"answer": "done"}

        result = my_node({"question": "hi", "_langfuse_trace": mock_trace})

        assert result == {"answer": "done"}
        mock_trace.span.assert_called_once()
        call_kwargs = mock_trace.span.call_args
        assert call_kwargs.kwargs["name"] == "my_node"
        mock_span.end.assert_called_once()

    def test_with_trace_records_error(self):
        """When the node raises, the span records the error."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        @trace_node("bad_node")
        def bad_node(state: dict) -> dict:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bad_node({"_langfuse_trace": mock_trace})

        mock_span.end.assert_called_once()
        end_kwargs = mock_span.end.call_args.kwargs
        assert end_kwargs["level"] == "ERROR"
        assert "boom" in end_kwargs["output"]["error"]

    def test_span_creation_failure_still_runs_node(self):
        """If span creation fails, the node still runs."""
        mock_trace = MagicMock()
        mock_trace.span.side_effect = RuntimeError("langfuse down")

        @trace_node("my_node")
        def my_node(state: dict) -> dict:
            return {"answer": "ok"}

        result = my_node({"_langfuse_trace": mock_trace})
        assert result == {"answer": "ok"}

    def test_preserves_function_metadata(self):
        @trace_node("my_node")
        def my_node(state: dict) -> dict:
            """My docstring."""
            return {}

        assert my_node.__name__ == "my_node"
        assert my_node.__doc__ == "My docstring."


# ── Disabled Langfuse helpers ─────────────────────────────────────────────────


class TestDisabledLangfuse:
    """Verify that all helpers return None when Langfuse is not configured."""

    def test_get_langfuse_handler_returns_none(self):
        with patch("app.rag.tracing._is_configured", return_value=False):
            from app.rag.tracing import get_langfuse_handler

            assert get_langfuse_handler() is None

    def test_create_trace_returns_none(self):
        with patch("app.rag.tracing.get_langfuse_client", return_value=None):
            from app.rag.tracing import create_trace

            assert create_trace(name="test") is None

    def test_flush_does_not_crash(self):
        with patch("app.rag.tracing.get_langfuse_client", return_value=None):
            from app.rag.tracing import flush

            flush()  # Should not raise


class TestInvokeGraphWithTracing:
    def test_invokes_graph_when_disabled(self):
        """When Langfuse is disabled, the graph is still invoked normally."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "ok", "grounded": True}

        with (
            patch("app.rag.tracing.create_trace", return_value=None),
            patch("app.rag.tracing.get_langfuse_handler", return_value=None),
            patch("app.rag.tracing.flush"),
        ):
            from app.rag.tracing import invoke_graph_with_tracing

            result = invoke_graph_with_tracing(
                mock_graph,
                {"question": "hello"},
                trace_name="test",
            )

        assert result["answer"] == "ok"
        mock_graph.invoke.assert_called_once()

    def test_injects_trace_into_state(self):
        """The trace object is injected as _langfuse_trace in state."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"answer": "ok"}
        mock_trace = MagicMock()
        mock_trace.id = "trace-123"

        with (
            patch("app.rag.tracing.create_trace", return_value=mock_trace),
            patch("app.rag.tracing.get_langfuse_handler", return_value=None),
            patch("app.rag.tracing.flush"),
        ):
            from app.rag.tracing import invoke_graph_with_tracing

            invoke_graph_with_tracing(
                mock_graph,
                {"question": "hello"},
                trace_name="test",
            )

        call_args = mock_graph.invoke.call_args
        state_passed = call_args[0][0]
        assert state_passed["_langfuse_trace"] is mock_trace
