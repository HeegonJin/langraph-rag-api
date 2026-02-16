# app/ — FastAPI Application Package

## OVERVIEW

FastAPI HTTP layer + config + Pydantic schemas. Delegates all RAG logic to `app/rag/`.

## STRUCTURE

```
app/
├── main.py      # FastAPI app, routes, lifespan
├── config.py    # Env-based settings (single source of truth)
├── models.py    # Pydantic request/response models
└── rag/         # Core RAG pipeline (see rag/AGENTS.md)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add endpoint | `main.py` | Follow existing pattern: `asyncio.to_thread(invoke_graph_with_tracing, ...)` |
| Add config var | `config.py` | `os.getenv()` with default; also update root `.env.example` |
| Add request/response model | `models.py` | Pydantic `BaseModel` with `Field(...)` |

## CONVENTIONS

- **Lifespan context manager** (not deprecated `@app.on_event`) for startup/shutdown
- Routes grouped by tags: `documents`, `single-turn`, `multi-turn`, `system`
- All sync graph calls wrapped in `asyncio.to_thread()` — never block the event loop
- `config.py` uses `dotenv.load_dotenv()` at import time — values available as module-level constants
- `UPLOAD_DIR` and `SAMPLE_DATA_DIR` created as `Path` objects in config (dirs auto-created)

## ANTI-PATTERNS

- Do NOT put business logic in `main.py` — it should only wire HTTP ↔ graph invocations
- Do NOT import from `app.rag.*` at function level for routes — use module-level imports (already done)
