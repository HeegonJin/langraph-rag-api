# app/rag/ — Core RAG Pipeline

## OVERVIEW

LangGraph state graphs, document ingestion/retrieval, conversation memory, and Langfuse tracing. This is where all RAG business logic lives.

## STRUCTURE

```
rag/
├── graph.py            # Single-turn RAG graph (retrieve → generate → grade → rewrite loop)
├── multiturn_graph.py  # Multi-turn graph (intent → contextualize → retrieve → generate → grade → save)
├── ingestion.py        # Document parsing (Docling), chunking (HybridChunker), Elasticsearch storage, score-filtered retrieval
├── conversation.py     # Redis-backed session store (Turn, Session, ConversationStore)
├── tracing.py          # Langfuse integration (trace_node decorator, invoke_graph_with_tracing)
├── constants.py        # Shared constants (NO_DOCS_ANSWER)
└── sample_ingest.py    # Idempotent auto-ingestion of sample_data/ on startup
```

## KEY PATTERNS

### Graph Architecture

Both graphs follow: **retrieve → generate → grade → (rewrite → retrieve loop, max 1 retry)**

Single-turn (`graph.py`):
```
retrieve → generate → grade ─┬─ grounded → END
                              └─ not grounded → rewrite → retrieve (loop)
```

Multi-turn (`multiturn_graph.py`):
```
classify_intent ─┬─ clear_context → END
                  └─ contextualize_query → retrieve → generate → grade ─┬─ save_turn → END
                                                                         └─ rewrite → retrieve
```

### Node Function Pattern

Every node: `def node_name(state: TypedDict) -> dict` decorated with `@trace_node("name")`

### LLM Access

- `_llm()` — cached `ChatOpenAI` pointing at llama.cpp (not OpenAI API)
- All calls: `ChatPromptTemplate | _llm()` chain pattern
- `lru_cache(maxsize=1)` on `_llm()` — single shared instance

### Retrieval

- `retrieve_with_scores(query, k=4, threshold)` in `ingestion.py`
- Elasticsearch hybrid search (BM25 + dense vector via RRF) using `ElasticsearchStore` with `ApproxRetrievalStrategy(hybrid=True)`
- Scores are 0.0–1.0 from RRF (not raw cosine); threshold > 0.0 filters low-relevance results
- Empty index → returns `[]` (graceful fallback, no crash)
- No docs found → `NO_DOCS_ANSWER` canned response, skip LLM call

### Session Memory

- `ConversationStore` (Redis hash per session, TTL 1h)
- `Session.update_cached_documents()` — SHA-256 dedup, max 20 docs cached
- Mutations auto-persist via `_persist()` back-reference to store
- Intent classification: `follow_up` | `new_topic` | `clear_context`
- Heuristic fast-path: regex for clear_context keywords (skips LLM call)

### Tracing

- `trace_node(name)` — decorator creating Langfuse spans per node
- `invoke_graph_with_tracing()` — creates top-level trace, injects into state as `_langfuse_trace`, attaches LangChain callback handler, flushes async
- When Langfuse disabled: all helpers return `None`/no-op — zero branching needed in business logic

## ANTI-PATTERNS

- Do NOT call `_llm()` outside node functions — it's designed for graph node scope
- Do NOT forget brace-escaping when building system prompts with document/history content
- Do NOT add nodes without `@trace_node()` decorator — breaks observability
- Do NOT bypass `invoke_graph_with_tracing()` — it handles trace lifecycle

## NOTES

- `multiturn_graph.py` is the largest file (487 lines) — 7 nodes + 2 conditional routers
- Intent classifier uses both heuristic regex AND LLM — regex short-circuits obvious cases
- `conversation.py` uses `hashlib.sha256` (not Python `hash()`) for cross-process determinism
- Both `graph.py` and `multiturn_graph.py` export module-level compiled graphs (`rag_graph`, `multiturn_rag_graph`)
