# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-16
**Commit:** 6e15a4f
**Branch:** main

## OVERVIEW

Multi-turn RAG API using FastAPI + LangGraph + llama.cpp (local LLM) + ChromaDB + Redis sessions + Langfuse tracing. Bilingual (Korean/English) prompts throughout.

## STRUCTURE

```
langraph-rag-api/
├── app/                # FastAPI application package
│   ├── main.py         # HTTP routes, lifespan, app factory
│   ├── config.py       # All env-based config (single source of truth)
│   ├── models.py       # Pydantic request/response schemas
│   └── rag/            # Core RAG pipeline (graphs, retrieval, sessions)
├── tests/              # pytest suite (unit + e2e with LLM judge)
├── demo.py             # Streamlit interactive demo (shares app/ logic)
├── start.sh            # Orchestrates llama.cpp servers + Streamlit
├── sample_data/        # Auto-ingested on startup (idempotent)
├── uploads/            # Runtime: user-uploaded docs (gitignored)
└── chroma_data/        # Runtime: ChromaDB persistence (gitignored)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify API endpoint | `app/main.py` | Routes call `invoke_graph_with_tracing` via `asyncio.to_thread` |
| Change LLM prompts | `app/rag/graph.py`, `app/rag/multiturn_graph.py` | Prompts inline in node functions |
| Modify retrieval/ingestion | `app/rag/ingestion.py` | Score threshold in `config.py` |
| Session/conversation logic | `app/rag/conversation.py` | Redis-backed, TTL 1h, max 20 cached docs |
| Tracing/observability | `app/rag/tracing.py` | No-op when Langfuse disabled |
| Add env config | `app/config.py` + `.env.example` | Always update both |
| Request/response shapes | `app/models.py` | Pydantic v2 |
| Run tests (unit) | `pytest -m "not e2e"` | Mocks LLM/vectorstore |
| Run tests (e2e) | `pytest -m e2e` | Requires llama.cpp + Redis + ChromaDB |

## CONVENTIONS

- **Python 3.12** (.python-version), pyproject allows >=3.11
- **uv** for dependency management (uv.lock present)
- **ruff** for linting + formatting, **mypy** for static type checking — both configured in `pyproject.toml` and must pass with zero errors before committing
- LangGraph nodes are plain functions decorated with `@trace_node("name")`
- LLM accessed via `ChatOpenAI` wrapper pointing at llama.cpp server (not OpenAI)
- All graph invocations go through `invoke_graph_with_tracing()` — never call `graph.invoke()` directly
- `asyncio.to_thread` wraps synchronous graph calls in async FastAPI handlers
- Curly braces in document/history content **must be escaped** (`{{`/`}}`) before passing to `ChatPromptTemplate`

## ANTI-PATTERNS (THIS PROJECT)

- **Do NOT call `graph.invoke()` directly** — always use `invoke_graph_with_tracing()` to preserve observability
- **Do NOT add config outside `app/config.py`** — it's the single source; update `.env.example` alongside
- **Do NOT skip brace-escaping** in generate nodes — `ChatPromptTemplate` interprets `{`/`}` as variables
- LLM prompt outputs ("ONLY yes/no", "ONLY one of: ...") are parsed with substring heuristics, not strict validation — be defensive when adding new prompt-based classification
- **Do NOT use `# type: ignore`** — fix the actual types instead
- **Do NOT suppress type errors** with `as Any`, `@ts-ignore`, or `cast()` unless absolutely unavoidable

## COMMANDS

```bash
# Dev server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start all services (llama.cpp + streamlit)
./start.sh            # start
./start.sh stop       # stop
./start.sh status     # check

# Linting & type checking (must pass before committing)
uv run ruff check .                  # lint
uv run ruff format --check .         # format check
uv run ruff format .                 # auto-format
uv run mypy app/ demo.py main.py     # type check

# Tests
uv run pytest -m "not e2e"    # unit tests (no infra needed)
uv run pytest -m e2e          # e2e (needs llama.cpp:8080/8081 + Redis + ChromaDB)
uv run pytest                 # all tests
```

## NOTES

- `main.py` at repo root is a stub — the real app is `app/main.py`
- `demo.py` (Streamlit) shares graph/ingestion logic with FastAPI — changes to `app/rag/` affect both
- Sample data auto-ingested on startup; marker file `chroma_data/.sample_ingested` makes it idempotent
- Tests override FastAPI lifespan with no-op to avoid startup side effects
- Redis DB 15 used by conversation tests for isolation (`flushdb` on setup/teardown)
- `test_eval_e2e.py` uses LLM-as-judge pattern (782 lines, 34 tests) — expensive to run
