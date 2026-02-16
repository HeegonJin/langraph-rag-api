# LangGraph Multi-Turn RAG API

A **RAG (Retrieval-Augmented Generation)** project with **multi-turn conversation** support, built with:

| Component | Role |
|-----------|------|
| **FastAPI** | HTTP API (single-turn + multi-turn endpoints) |
| **LangChain** | Document loading, splitting, embeddings, retrieval |
| **LangGraph** | Orchestrates the RAG workflow as a stateful graph |
| **llama.cpp** | Local LLM & embedding server (no API keys needed) |
| **ChromaDB** | Vector store for document embeddings |
| **Redis** | Persistent session memory for multi-turn conversations |
| **Langfuse** | Observability – trace & debug every node's I/O |

## Multi-Turn Features (멀티턴 대화)

| Feature | Description |
|---------|-------------|
| **Session Memory (Redis)** | Per-session conversation history persisted to Redis with configurable TTL |
| **Intent Classification** (인텐트 분류) | Detects `follow_up`, `new_topic`, or `clear_context` intent |
| **Query Contextualization** (질문 의도 파악) | Rewrites follow-up questions to be self-contained using chat history |
| **Document Deduplication** (정보 중복 최소화) | Merges retrieved documents with session cache to avoid redundant searches |
| **Chain-of-Thought Prompting** | Step-by-step reasoning with full conversation context |
| **Clear Context** (대화 맥락 초기화) | Explicit conversation reset via API or UI button |

## Architecture

### Single-Turn (original)

```
┌────────┐     ┌──────────┐     ┌──────────┐     ┌───────┐
│retrieve│────▶│ generate  │────▶│  grade   │────▶│  END  │
└────────┘     └──────────┘     └──────────┘     └───────┘
     ▲                               │
     │          ┌──────────┐         │ (not grounded)
     └──────────│ rewrite  │◀────────┘
                └──────────┘
```

### Multi-Turn (new)

```
┌──────────────────┐
│ classify_intent   │
└────────┬─────────┘
         ├─ clear_context ──────────────────────▶ END
         │
┌────────▼─────────┐
│contextualize_query│  (rewrite question with history)
└────────┬─────────┘
         │
┌────────▼─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
│    retrieve       │────▶│ generate  │────▶│  grade   │────▶│ save_turn │──▶ END
│ (+ dedup cache)   │     │ (CoT)     │     └────┬─────┘     └───────────┘
└──────────────────┘     └──────────┘          │
     ▲                                          │ (not grounded)
     │          ┌──────────┐                    │
     └──────────│ rewrite  │◀───────────────────┘
                └──────────┘
```

**Multi-turn workflow:**
1. **classify_intent** – determines if this is a follow-up, new topic, or reset request
2. **contextualize_query** – rewrites the question to be self-contained using conversation history
3. **retrieve** – fetches relevant chunks, merges with session's document cache (dedup)
4. **generate** – Chain-of-Thought answer generation with full conversation context
5. **grade** – groundedness check (retries with rewrite if needed)
6. **save_turn** – persists the Q&A pair into Redis session memory

## Prerequisites

1. **Python 3.11+**
2. **Redis** – `sudo apt install redis-server` or `brew install redis`
3. **llama.cpp** – build from source or grab a release from https://github.com/ggerganov/llama.cpp
4. Download GGUF model files (e.g. from Hugging Face). You need:
   - A **chat model** – we use [GLM-4.7-Flash](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) (Q4_K_XL quantisation)
   - An **embedding model** – we use [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)

```bash
# Download models
mkdir -p ~/models/GLM-4.7-Flash ~/models/Qwen3-Embedding-0.6B
wget -O ~/models/GLM-4.7-Flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
    https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-UD-Q4_K_XL.gguf
wget -O ~/models/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q8_0.gguf \
    https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
```

## Quick Start

The easiest way to start everything is the included `start.sh` script:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Copy and (optionally) edit the env file
cp .env.example .env

# 4. Make sure Redis is running
sudo systemctl start redis-server   # or: redis-server --daemonize yes

# 5. Start all services (llama.cpp chat + embedding + Streamlit demo)
bash start.sh
```

### Service management

```bash
./start.sh          # start all services
./start.sh stop     # stop all services
./start.sh status   # show running services
./start.sh logs     # tail all log files
```

### Manual startup

If you prefer to start services individually:

```bash
# Chat model (port 8080) – reasoning disabled for faster responses
nohup llama-server \
    --model ~/models/GLM-4.7-Flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
    --ctx-size 16384 \
    --host 0.0.0.0 --port 8080 \
    --seed 3407 --temp 0.7 --top-p 1.0 --min-p 0.01 \
    --jinja --reasoning-budget 0 \
    > ~/llama-chat.log 2>&1 &

# Embedding model (port 8081)
nohup llama-server \
    --model ~/models/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --ctx-size 8192 \
    --host 0.0.0.0 --port 8081 \
    --embedding \
    > ~/llama-embed.log 2>&1 &

# FastAPI server
uv run uvicorn app.main:app --reload
```

> **Tip:** Remove `--reasoning-budget 0` to enable chain-of-thought reasoning
> (slower but potentially higher quality). This is a server-level flag only
> (`-1` = unlimited thinking, `0` = disabled).

The API docs will be available at **http://127.0.0.1:8000/docs**.

## Web Demo (Streamlit)

A chat-style UI that talks directly to the RAG pipeline (no FastAPI server needed):

```bash
uv run streamlit run demo.py --server.address 0.0.0.0 --server.port 8502
```

Open **http://localhost:8502** — upload documents in the sidebar, then chat.

## Usage

### Ingest a document
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@my_document.txt"
```

### Ask a question – single-turn (legacy)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Multi-turn conversation

```bash
# Turn 1 – start a new conversation (no session_id)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "이번 주말에 상영하는 액션 영화 목록을 알려줘"}'

# Response includes session_id, e.g. "session_id": "abc123..."

# Turn 2 – follow-up using the same session
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "어벤져스 상영 시간이 궁금해", "session_id": "abc123..."}'

# Clear context (대화 맥락 초기화)
curl -X POST http://127.0.0.1:8000/chat/clear \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123..."}'

# List active sessions
curl http://127.0.0.1:8000/chat/sessions

# Delete a session
curl -X DELETE http://127.0.0.1:8000/chat/sessions/abc123...
```

## Project Structure

```
langraph-rag-api/
├── app/
│   ├── main.py              # FastAPI endpoints (single-turn + multi-turn)
│   ├── config.py            # Environment-based settings
│   ├── models.py            # Pydantic request/response schemas
│   └── rag/
│       ├── constants.py     # Shared constants (NO_DOCS_ANSWER, MAX_RETRIES)
│       ├── conversation.py  # Redis-backed session memory (ConversationStore)
│       ├── graph.py         # LangGraph single-turn RAG workflow
│       ├── multiturn_graph.py  # LangGraph multi-turn RAG workflow
│       ├── ingestion.py     # Doc loading, splitting, vector store
│       ├── sample_ingest.py # Auto-ingest sample_data/ on startup
│       └── tracing.py       # Langfuse observability integration
├── tests/
│   ├── test_api.py          # FastAPI endpoint unit tests
│   ├── test_conversation.py # Redis conversation store tests
│   ├── test_eval_e2e.py     # LLM-as-judge end-to-end evaluation (34 tests)
│   ├── test_graphs.py       # Graph workflow unit tests
│   ├── test_ingestion.py    # Document ingestion tests
│   ├── test_models.py       # Pydantic schema tests
│   └── test_tracing.py      # Langfuse tracing tests
├── sample_data/             # Sample documents auto-ingested on startup
├── demo.py                  # Streamlit web demo (multi-turn)
├── start.sh                 # Service management (start/stop/status/logs)
├── pyproject.toml
├── .env.example
└── README.md
```

## Testing

### Unit tests (107 tests, mocked LLM calls)

```bash
uv run pytest tests/ --ignore=tests/test_eval_e2e.py -v
```

### End-to-end evaluation with LLM-as-judge (34 tests)

The e2e test suite runs the **full agent loop** — retrieval, generation, grading,
intent classification, and session persistence — then uses the same chat model
as an automated evaluator to score answers on a 1–5 Likert scale.

**Requirements:** llama.cpp servers on :8080/:8081, Redis on :6379, ChromaDB populated.

```bash
uv run pytest tests/test_eval_e2e.py -v -s
```

**Evaluation dimensions:**

| Dimension | What it measures |
|-----------|-----------------|
| **Relevance** | Does the answer address the question? |
| **Groundedness** | Is the answer faithful to retrieved documents (no hallucination)? |
| **Coherence** | Is the answer well-structured and easy to understand? |
| **Completeness** | Does the answer cover the key points needed? |

**Test categories:**

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestSingleTurnJudged` | 6 | Factual questions scored by LLM judge |
| `TestOutOfScope` | 3 | Out-of-scope / gibberish handling |
| `TestMultiTurnIntentFlow` | 6 | Intent classification (follow-up / new topic / clear) |
| `TestSessionPersistence` | 3 | Redis session storage across turns |
| `TestSemanticQualityJudged` | 6 | Answer quality, Korean support, no hallucination |
| `TestGroundingRetrieval` | 3 | Retrieval pipeline & retry mechanism |
| `TestExtendedConversationJudged` | 3 | 3+ turn conversations, topic switching |
| `TestAPIEndToEnd` | 4 | FastAPI endpoints with real graph execution |

## Observability (Langfuse)

Every graph invocation is traced with [Langfuse](https://langfuse.com), giving
you full visibility into each node's inputs and outputs.

### What gets traced

| Layer | What you see in Langfuse |
|-------|--------------------------|
| **Trace** (top-level) | One trace per `/ask` or `/chat` request |
| **Spans** (per node) | `classify_intent`, `contextualize_query`, `retrieve`, `generate`, `grade`, `rewrite`, `save_turn` |
| **LLM observations** | Every `ChatOpenAI` call (prompt, completion, tokens, latency) |
| **Metadata** | Session ID, intent, question, grounded status |

### Setup

1. Create a free account at [cloud.langfuse.com](https://cloud.langfuse.com)
   (or self-host).
2. Create a project and copy the API keys.
3. Add them to your `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
```

When `LANGFUSE_ENABLED=false` or keys are empty, tracing is silently disabled
and the app works normally.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Upload & index a document |
| `POST` | `/ask` | Single-turn question (no memory) |
| `POST` | `/chat` | Multi-turn question (with session memory) |
| `POST` | `/chat/clear` | Clear conversation context for a session |
| `GET`  | `/chat/sessions` | List active sessions |
| `DELETE` | `/chat/sessions/{id}` | Delete a session |
| `GET`  | `/health` | Health check |

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_BASE_URL` | `http://localhost:8080/v1` | Chat model endpoint |
| `EMBEDDING_BASE_URL` | `http://localhost:8081/v1` | Embedding model endpoint |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for session storage |
| `REDIS_SESSION_TTL` | `3600` | Session expiry in seconds |
| `MAX_RETRIES` | `3` | Max grounding retries before returning best answer |
| `LANGFUSE_ENABLED` | `false` | Enable/disable Langfuse tracing |
