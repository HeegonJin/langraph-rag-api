"""Langfuse observability integration for RAG pipelines.

Provides:
  • ``get_langfuse_handler()``    – LangChain CallbackHandler that auto-traces
                                    every LLM / chain / retriever call.
  • ``get_langfuse_client()``     – low-level Langfuse client for custom spans.
  • ``trace_node()``              – decorator that wraps a LangGraph node function
                                    in a Langfuse *span* so you can see per-node
                                    input / output in the dashboard.
  • ``create_trace()``            – start a new top-level trace (one per API
                                    request / graph invocation).

When ``LANGFUSE_ENABLED`` is ``False`` (or keys are missing), every helper
returns no-op / ``None`` so the rest of the code doesn't need to branch.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from app import config

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────

_langfuse_client = None
_initialised = False


def _is_configured() -> bool:
    """Return True when Langfuse is both enabled AND has credentials."""
    return bool(
        config.LANGFUSE_ENABLED and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY
    )


def get_langfuse_client() -> Any:
    """Return a shared ``Langfuse`` client, or ``None`` when disabled."""
    global _langfuse_client, _initialised
    if _initialised:
        return _langfuse_client
    _initialised = True

    if not _is_configured():
        logger.info("Langfuse disabled (missing keys or LANGFUSE_ENABLED=false)")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        logger.info("Langfuse client initialised (%s)", config.LANGFUSE_HOST)
    except Exception:
        logger.exception("Failed to initialise Langfuse client")
        _langfuse_client = None

    return _langfuse_client


def get_langfuse_handler(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Return a LangChain ``CallbackHandler`` wired to Langfuse.

    This handler auto-captures every LLM call, chain run, and retriever call
    made by LangChain / LangGraph and sends them to Langfuse as *observations*
    inside a trace.

    Returns ``None`` when Langfuse is disabled so callers can simply do::

        callbacks = [h for h in [get_langfuse_handler()] if h]
    """
    if not _is_configured():
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            metadata=metadata or {},
            tags=tags or [],
        )
        return handler
    except Exception:
        logger.exception("Failed to create Langfuse CallbackHandler")
        return None


def create_trace(
    *,
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    input: Any = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Create a new top-level Langfuse trace and return it.

    Returns ``None`` when Langfuse is disabled.
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace = client.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            input=input,
            metadata=metadata or {},
            tags=tags or [],
        )
        return trace
    except Exception:
        logger.exception("Failed to create Langfuse trace")
        return None


def flush() -> None:
    """Flush any buffered Langfuse events.

    Uses a background thread so the calling request is not blocked.
    """
    client = get_langfuse_client()
    if client:
        import threading

        def _do_flush() -> None:
            try:
                client.flush()
            except Exception:
                logger.exception("Failed to flush Langfuse")

        threading.Thread(target=_do_flush, daemon=True).start()


# ── Node-level tracing decorator ─────────────────────────────────────────────


def trace_node(
    node_name: str,
    *,
    trace_key: str = "_langfuse_trace",
) -> Callable:
    """Decorator that wraps a LangGraph node function in a Langfuse span.

    Usage::

        @trace_node("retrieve")
        def retrieve(state: RAGState) -> dict:
            ...

    The decorator:
      1. Opens a child span on the current trace (stored in ``state[trace_key]``).
      2. Records the node's *input* (relevant state keys) and *output*.
      3. Handles the case where tracing is disabled gracefully.

    The trace object is expected to be injected into the graph state before
    invocation (see ``invoke_with_tracing``).
    """

    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            trace = state.get(trace_key)
            if trace is None:
                # No tracing – just run the node
                return fn(state)

            # Build a summary of input (skip large fields like documents)
            input_summary = _summarise_state(state, for_input=True)

            try:
                span = trace.span(
                    name=node_name,
                    input=input_summary,
                )
            except Exception:
                logger.debug("Could not create span for %s", node_name)
                return fn(state)

            try:
                result = fn(state)
            except Exception as exc:
                span.end(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
                raise

            # Record output
            output_summary = _summarise_output(result)
            span.end(output=output_summary)
            return result

        return wrapper

    return decorator


def _summarise_state(state: dict[str, Any], *, for_input: bool = True) -> dict[str, Any]:
    """Create a JSON-safe summary of graph state for Langfuse.

    All string values are sent in full (no truncation) so that
    chat_history, answer, etc. are fully visible in the console.
    """
    summary: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("_"):
            continue  # skip internal keys like _langfuse_trace
        if key == "documents":
            docs = value or []
            summary["documents_count"] = len(docs)
            if docs:
                summary["documents_preview"] = [
                    d.page_content if hasattr(d, "page_content") else str(d) for d in docs
                ]
        elif isinstance(value, (str, int, float, bool)):
            summary[key] = value
        elif isinstance(value, list):
            summary[key] = f"list[{len(value)}]"
        elif value is None:
            summary[key] = None
        else:
            summary[key] = str(value)
    return summary


def _summarise_output(result: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-safe summary of node output for Langfuse.

    No truncation is applied — full content is preserved.
    """
    if not isinstance(result, dict):
        return {"raw": str(result)}
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if key == "documents":
            docs = value or []
            summary["documents_count"] = len(docs)
            if docs:
                summary["documents_preview"] = [
                    d.page_content if hasattr(d, "page_content") else str(d) for d in docs
                ]
        elif isinstance(value, (str, int, float, bool)):
            summary[key] = value
        elif value is None:
            summary[key] = None
        else:
            summary[key] = str(value)
    return summary


# ── High-level helpers for graph invocation ───────────────────────────────────


def invoke_graph_with_tracing(
    graph: Any,
    state: dict[str, Any],
    *,
    trace_name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a LangGraph compiled graph with full Langfuse tracing.

    This function:
      1. Creates a top-level Langfuse trace.
      2. Injects it into the graph state under ``_langfuse_trace``.
      3. Creates a LangChain CallbackHandler linked to the same trace.
      4. Invokes the graph with that callback.
      5. Updates the trace with the final output.
      6. Flushes buffered events.

    When Langfuse is disabled, the graph is invoked normally.
    """
    trace = create_trace(
        name=trace_name,
        session_id=session_id,
        input=_summarise_state(state, for_input=True),
        metadata=metadata or {},
        tags=tags or [],
    )

    # Inject trace into state for node-level spans
    state["_langfuse_trace"] = trace

    # Build LangChain callback handler linked to same trace
    callbacks = []
    if trace is not None:
        handler = get_langfuse_handler(
            trace_id=trace.id,
            session_id=session_id,
            trace_name=trace_name,
            tags=tags,
        )
        if handler:
            callbacks.append(handler)

    # Invoke graph
    config_dict = {}
    if callbacks:
        config_dict["callbacks"] = callbacks

    result: dict[str, Any] = graph.invoke(state, config=config_dict if config_dict else None)

    # Update trace with final output
    if trace is not None:
        try:
            output_summary = _summarise_state(result, for_input=False)
            trace.update(output=output_summary)
        except Exception:
            logger.debug("Failed to update trace output")

    # Flush async – don't block the response
    flush()

    return result
