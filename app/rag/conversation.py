"""Session-based conversation memory for multi-turn RAG – Redis-backed.

Stores per-session chat history in Redis so that the RAG graph can:
  • Maintain context across turns (대화 맥락 유지)
  • Re-use previously retrieved documents to reduce redundant searches
    (정보 중복 처리 최소화)
  • Support explicit context clearing (Clear Context / 대화 맥락 초기화)
  • Survive server restarts and work across multiple workers

Each session is stored as a Redis hash with JSON-serialised fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import redis
from langchain_core.documents import Document

from app import config

logger = logging.getLogger(__name__)

# ── Data classes (used as in-memory representations) ──────────────────────────


@dataclass
class Turn:
    """A single conversation turn (user question + assistant answer)."""

    role: str  # "human" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict) -> Turn:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class Session:
    """All state kept for a single conversation.

    This is a transient in-memory view.  Mutations are persisted back to
    Redis via the :class:`ConversationStore` that owns this session.
    """

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    cached_documents: list[Document] = field(default_factory=list)
    last_intent: str = "new_topic"  # new_topic | follow_up | clear_context
    created_at: float = field(default_factory=time.time)

    # Back-reference to the store so mutation helpers can auto-persist.
    _store: Optional[ConversationStore] = field(default=None, repr=False)

    # ── helpers ────────────────────────────────────────────────────────────

    def add_human_turn(self, content: str) -> None:
        self.turns.append(Turn(role="human", content=content))
        self._persist()

    def add_assistant_turn(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content))
        self._persist()

    def get_history_text(self, max_turns: int = 10) -> str:
        """Return the last *max_turns* turns formatted as plain text."""
        recent = self.turns[-max_turns * 2 :]  # each turn pair = 2 entries
        lines: list[str] = []
        for t in recent:
            prefix = "User" if t.role == "human" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def get_history_messages(self, max_turns: int = 10) -> list[tuple[str, str]]:
        """Return history as (role, content) tuples for ChatPromptTemplate."""
        recent = self.turns[-max_turns * 2 :]
        return [(t.role, t.content) for t in recent]

    def clear(self) -> None:
        """Clear Context – 대화 맥락 초기화.

        Resets conversation turns, document cache, and intent so the
        session starts fresh.
        """
        self.turns.clear()
        self.cached_documents.clear()
        self.last_intent = "new_topic"
        self._persist()

    def update_cached_documents(self, docs: list[Document]) -> list[Document]:
        """Merge new docs with cache, deduplicating by page_content hash.

        Uses a stable SHA-256 hash (not Python's built-in ``hash()``) so
        behaviour is deterministic across processes.

        Returns the merged (deduplicated) document list.
        """
        seen = {_stable_hash(d.page_content) for d in self.cached_documents}
        new_docs: list[Document] = []
        for d in docs:
            h = _stable_hash(d.page_content)
            if h not in seen:
                seen.add(h)
                new_docs.append(d)
        self.cached_documents = self.cached_documents + new_docs
        # Keep at most 20 cached docs to avoid unbounded growth
        self.cached_documents = self.cached_documents[-20:]
        self._persist()
        return self.cached_documents

    def _persist(self) -> None:
        """Write the session back to Redis (no-op if no store attached)."""
        if self._store is not None:
            self._store._save_session(self)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stable_hash(text: str) -> str:
    """Return a deterministic, process-safe hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


def _session_to_json(session: Session) -> dict:
    """Serialise a Session to a dict suitable for Redis storage."""
    return {
        "session_id": session.session_id,
        "turns": json.dumps([t.to_dict() for t in session.turns]),
        "cached_documents": json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in session.cached_documents]
        ),
        "last_intent": session.last_intent,
        "created_at": str(session.created_at),
    }


def _session_from_json(data: dict, store: ConversationStore) -> Session:
    """Deserialise a Session from a Redis hash dict."""
    turns = [Turn.from_dict(t) for t in json.loads(data.get("turns", "[]"))]
    cached_docs_raw = json.loads(data.get("cached_documents", "[]"))
    cached_documents = [
        Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for d in cached_docs_raw
    ]
    return Session(
        session_id=data["session_id"],
        turns=turns,
        cached_documents=cached_documents,
        last_intent=data.get("last_intent", "new_topic"),
        created_at=float(data.get("created_at", time.time())),
        _store=store,
    )


# ── Redis-backed store ────────────────────────────────────────────────────────


_KEY_PREFIX = "rag:session:"


class ConversationStore:
    """Redis-backed store of conversation sessions.

    Each session is stored as a Redis hash under the key
    ``rag:session:<session_id>``.  A TTL is set on every write so
    stale sessions are automatically cleaned up by Redis.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        url = redis_url or getattr(config, "REDIS_URL", "redis://localhost:6379/0")
        self._redis: redis.Redis = redis.Redis.from_url(
            url, decode_responses=True,
        )
        self._ttl = ttl_seconds

    # ── internal ───────────────────────────────────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def _save_session(self, session: Session) -> None:
        """Persist a Session object to Redis."""
        key = self._key(session.session_id)
        self._redis.hset(key, mapping=_session_to_json(session))
        self._redis.expire(key, self._ttl)

    def _load_session(self, session_id: str) -> Session | None:
        """Load a Session from Redis, or return None if not found."""
        key = self._key(session_id)
        data = self._redis.hgetall(key)
        if not data:
            return None
        session = _session_from_json(data, store=self)
        # Refresh TTL on access
        self._redis.expire(key, self._ttl)
        return session

    # ── public API ─────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str | None = None) -> Session:
        """Return an existing session or create a new one."""
        if session_id:
            existing = self._load_session(session_id)
            if existing is not None:
                return existing
        sid = session_id or uuid.uuid4().hex
        session = Session(session_id=sid, _store=self)
        self._save_session(session)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._load_session(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear Context for a session. Returns True if session existed."""
        session = self._load_session(session_id)
        if session:
            session.clear()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        key = self._key(session_id)
        return self._redis.delete(key) > 0

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        prefix_len = len(_KEY_PREFIX)
        keys = self._redis.keys(f"{_KEY_PREFIX}*")
        return [k[prefix_len:] for k in keys]


# Module-level singleton
conversation_store = ConversationStore()
