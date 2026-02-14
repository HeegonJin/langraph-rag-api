"""Tests for Session, Turn, and Redis-backed ConversationStore."""

import time

import pytest
from langchain_core.documents import Document

from app.rag.conversation import ConversationStore, Session, Turn


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def store():
    """Create a ConversationStore that uses a test-only Redis DB (db=15).

    Flushes the DB before each test to ensure isolation.
    """
    s = ConversationStore(redis_url="redis://localhost:6379/15", ttl_seconds=60)
    s._redis.flushdb()
    yield s
    s._redis.flushdb()


# ── Turn ──────────────────────────────────────────────────────────────────────


class TestTurn:
    def test_creation(self):
        t = Turn(role="human", content="hello")
        assert t.role == "human"
        assert t.content == "hello"
        assert isinstance(t.timestamp, float)

    def test_timestamp_auto(self):
        before = time.time()
        t = Turn(role="assistant", content="hi")
        after = time.time()
        assert before <= t.timestamp <= after

    def test_round_trip(self):
        t = Turn(role="human", content="Q1", timestamp=1234.5)
        d = t.to_dict()
        t2 = Turn.from_dict(d)
        assert t2.role == t.role
        assert t2.content == t.content
        assert t2.timestamp == t.timestamp


# ── Session ───────────────────────────────────────────────────────────────────


class TestSession:
    def _make_session(self, sid: str = "test-session") -> Session:
        return Session(session_id=sid)

    def test_add_human_turn(self):
        s = self._make_session()
        s.add_human_turn("What is RAG?")
        assert len(s.turns) == 1
        assert s.turns[0].role == "human"
        assert s.turns[0].content == "What is RAG?"

    def test_add_assistant_turn(self):
        s = self._make_session()
        s.add_assistant_turn("RAG stands for...")
        assert len(s.turns) == 1
        assert s.turns[0].role == "assistant"

    def test_get_history_text_empty(self):
        s = self._make_session()
        assert s.get_history_text() == ""

    def test_get_history_text_formatting(self):
        s = self._make_session()
        s.add_human_turn("Q1")
        s.add_assistant_turn("A1")
        text = s.get_history_text()
        assert "User: Q1" in text
        assert "Assistant: A1" in text

    def test_get_history_text_max_turns(self):
        s = self._make_session()
        for i in range(10):
            s.add_human_turn(f"Q{i}")
            s.add_assistant_turn(f"A{i}")
        # max_turns=2 → last 4 entries (2 pairs)
        text = s.get_history_text(max_turns=2)
        assert "Q8" in text
        assert "Q9" in text
        assert "Q0" not in text

    def test_get_history_messages(self):
        s = self._make_session()
        s.add_human_turn("Q")
        s.add_assistant_turn("A")
        msgs = s.get_history_messages()
        assert msgs == [("human", "Q"), ("assistant", "A")]

    def test_get_history_messages_max_turns(self):
        s = self._make_session()
        for i in range(5):
            s.add_human_turn(f"Q{i}")
            s.add_assistant_turn(f"A{i}")
        msgs = s.get_history_messages(max_turns=1)
        assert len(msgs) == 2
        assert msgs[0] == ("human", "Q4")

    def test_clear(self):
        s = self._make_session()
        s.add_human_turn("hi")
        s.add_assistant_turn("hello")
        s.cached_documents = [Document(page_content="doc")]
        s.last_intent = "follow_up"

        s.clear()

        assert s.turns == []
        assert s.cached_documents == []
        assert s.last_intent == "new_topic"

    def test_update_cached_documents_dedup(self):
        s = self._make_session()
        doc_a = Document(page_content="AAA")
        doc_b = Document(page_content="BBB")
        doc_a_dup = Document(page_content="AAA")

        result = s.update_cached_documents([doc_a, doc_b])
        assert len(result) == 2

        # Adding a duplicate should not increase count
        result = s.update_cached_documents([doc_a_dup])
        assert len(result) == 2

    def test_update_cached_documents_new_added(self):
        s = self._make_session()
        doc_a = Document(page_content="AAA")
        s.update_cached_documents([doc_a])

        doc_c = Document(page_content="CCC")
        result = s.update_cached_documents([doc_c])
        assert len(result) == 2

    def test_update_cached_documents_max_20(self):
        s = self._make_session()
        # Add 25 unique docs
        docs = [Document(page_content=f"doc-{i}") for i in range(25)]
        result = s.update_cached_documents(docs)
        assert len(result) <= 20

    def test_session_defaults(self):
        s = self._make_session()
        assert s.turns == []
        assert s.cached_documents == []
        assert s.last_intent == "new_topic"
        assert isinstance(s.created_at, float)


# ── ConversationStore (Redis-backed) ──────────────────────────────────────────


class TestConversationStore:
    def test_get_or_create_new(self, store):
        session = store.get_or_create()
        assert isinstance(session, Session)
        assert len(session.session_id) > 0

    def test_get_or_create_with_id(self, store):
        session = store.get_or_create("my-session")
        assert session.session_id == "my-session"

    def test_get_or_create_returns_existing(self, store):
        s1 = store.get_or_create("s1")
        s1.add_human_turn("hi")
        # Re-fetch should see the persisted turn
        s2 = store.get_or_create("s1")
        assert s2.session_id == s1.session_id
        assert len(s2.turns) == 1

    def test_get_existing(self, store):
        store.get_or_create("s1")
        assert store.get("s1") is not None

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None

    def test_clear_session_existing(self, store):
        s = store.get_or_create("s1")
        s.add_human_turn("hi")
        assert store.clear_session("s1") is True
        reloaded = store.get("s1")
        assert len(reloaded.turns) == 0

    def test_clear_session_nonexistent(self, store):
        assert store.clear_session("nope") is False

    def test_delete_session(self, store):
        store.get_or_create("s1")
        assert store.delete_session("s1") is True
        assert store.get("s1") is None

    def test_delete_session_nonexistent(self, store):
        assert store.delete_session("nope") is False

    def test_list_sessions(self, store):
        store.get_or_create("a")
        store.get_or_create("b")
        store.get_or_create("c")
        sessions = store.list_sessions()
        assert set(sessions) == {"a", "b", "c"}

    def test_list_sessions_empty(self, store):
        assert store.list_sessions() == []

    def test_session_persists_turns(self, store):
        """Verify turns survive a round-trip through Redis."""
        s = store.get_or_create("persist-test")
        s.add_human_turn("Hello")
        s.add_assistant_turn("Hi there!")

        reloaded = store.get("persist-test")
        assert len(reloaded.turns) == 2
        assert reloaded.turns[0].role == "human"
        assert reloaded.turns[0].content == "Hello"
        assert reloaded.turns[1].content == "Hi there!"

    def test_session_persists_cached_documents(self, store):
        """Verify cached documents survive a round-trip through Redis."""
        s = store.get_or_create("doc-test")
        docs = [Document(page_content="Doc A"), Document(page_content="Doc B")]
        s.update_cached_documents(docs)

        reloaded = store.get("doc-test")
        assert len(reloaded.cached_documents) == 2
        assert reloaded.cached_documents[0].page_content == "Doc A"

    def test_ttl_expiry(self, store):
        """Sessions with a short TTL should expire automatically."""
        short_ttl_store = ConversationStore(
            redis_url="redis://localhost:6379/15", ttl_seconds=1,
        )
        short_ttl_store.get_or_create("expire-me")
        time.sleep(1.5)
        assert short_ttl_store.get("expire-me") is None
