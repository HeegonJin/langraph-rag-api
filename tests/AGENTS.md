# tests/ — Test Suite

## OVERVIEW

pytest-based tests: unit tests (mocked LLM/vectorstore) and e2e evaluation suite (live infra + LLM-as-judge).

## STRUCTURE

```
tests/
├── test_api.py           # FastAPI endpoint unit tests (mocked graphs)
├── test_conversation.py  # ConversationStore + Session (requires Redis)
├── test_eval_e2e.py      # E2E with LLM judge (requires all infra) — 782 lines
├── test_graphs.py        # Graph compilation, node behavior, retry logic
├── test_ingestion.py     # Ingestion pipeline (mocked vectorstore)
├── test_models.py        # Pydantic model validation
└── test_tracing.py       # Tracing decorator and invoke_graph_with_tracing
```

## HOW TO RUN

```bash
pytest -m "not e2e"                    # Unit tests only (safe, no infra)
pytest -m e2e                          # E2E (needs llama.cpp:8080/8081 + Redis + Elasticsearch)
pytest tests/test_api.py               # Single file
pytest tests/test_api.py::TestAskEndpoint::test_ask_success  # Single test
```

## CONVENTIONS

- **No conftest.py** — fixtures defined inline in each test file
- Tests override `app.router.lifespan_context` with no-op to skip startup ingestion/Langfuse
- `unittest.mock.patch` targets `app.main.invoke_graph_with_tracing` for API tests
- `test_conversation.py` uses Redis DB 15 (`flushdb` on setup/teardown) for isolation
- E2E marker: `pytestmark = pytest.mark.e2e` at module level in `test_eval_e2e.py`
- LLM-as-judge pattern in e2e: sends answers to judge LLM, asserts numeric quality thresholds
- Bilingual test inputs (Korean + English) validate multilingual behavior

## WHEN ADDING TESTS

- Mock `invoke_graph_with_tracing` for endpoint tests (not the graph directly)
- Use `@pytest.mark.e2e` for tests requiring live LLM/embedding servers
- Conversation tests need live Redis — skip or mock if unavailable
- Graph unit tests mock `_llm()` factory and override `__or__` for chain simulation
