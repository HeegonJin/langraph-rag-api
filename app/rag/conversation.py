"""Session-based conversation memory for multi-turn RAG.

Stores per-session chat history so that the RAG graph can:
  • Maintain context across turns (대화 맥락 유지)
  • Re-use previously retrieved documents to reduce redundant searches
    (정보 중복 처리 최소화)
  • Support explicit context clearing (Clear Context / 대화 맥락 초기화)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from langchain_core.documents import Document


@dataclass
class Turn:
    """A single conversation turn (user question + assistant answer)."""

    role: str  # "human" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """All state kept for a single conversation."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    cached_documents: list[Document] = field(default_factory=list)
    last_intent: str = "new_topic"  # new_topic | follow_up | clear_context
    created_at: float = field(default_factory=time.time)

    # ── helpers ────────────────────────────────────────────────────────────

    def add_human_turn(self, content: str) -> None:
        self.turns.append(Turn(role="human", content=content))

    def add_assistant_turn(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content))

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

        Only resets the document cache and intent; conversation history
        (turns) is preserved so the agent can recall prior context if
        the user revisits an earlier topic.
        """
        self.cached_documents.clear()
        self.last_intent = "new_topic"

    def update_cached_documents(self, docs: list[Document]) -> list[Document]:
        """Merge new docs with cache, deduplicating by page_content hash.

        Returns the merged (deduplicated) document list.
        """
        seen = {hash(d.page_content) for d in self.cached_documents}
        new_docs: list[Document] = []
        for d in docs:
            h = hash(d.page_content)
            if h not in seen:
                seen.add(h)
                new_docs.append(d)
        self.cached_documents = self.cached_documents + new_docs
        # Keep at most 20 cached docs to avoid unbounded growth
        self.cached_documents = self.cached_documents[-20:]
        return self.cached_documents


class ConversationStore:
    """Thread-safe, in-memory store of conversation sessions.

    For production you would swap this for Redis / a database;
    the in-memory version is perfectly fine for demos and dev.
    """

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    # ── public API ─────────────────────────────────────────────────────────

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Return an existing session or create a new one."""
        with self._lock:
            self._evict_stale()
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            sid = session_id or uuid.uuid4().hex
            session = Session(session_id=sid)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear Context for a session. Returns True if session existed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.clear()
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    # ── internal ───────────────────────────────────────────────────────────

    def _evict_stale(self) -> None:
        """Remove sessions older than TTL when we exceed max_sessions."""
        if len(self._sessions) < self._max_sessions:
            return
        now = time.time()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if now - s.created_at > self._ttl
        ]
        for sid in stale:
            del self._sessions[sid]


# Module-level singleton
conversation_store = ConversationStore()
