"""End-to-end evaluation test suite for the RAG pipeline — with LLM-as-judge.

Unlike the unit tests that mock LLM calls, these tests run the **full agent
loop** — retrieval from Elasticsearch, LLM generation via llama.cpp, grading, and
(for multi-turn) intent classification, query contextualisation, and session
persistence via Redis.

**LLM Judge:** Many assertions use the same chat model (GLM-4-Flash) as an
automated evaluator.  The judge scores each answer on a 1–5 Likert scale for
dimensions such as *Relevance*, *Groundedness*, *Coherence*, and
*Completeness*.  This allows semantic evaluation well beyond keyword matching.

Requirements to run:
  - llama.cpp LLM server on :8080
  - llama.cpp embedding server on :8081
  - Redis on :6379
  - Elasticsearch populated with the A2D paper (sample_data auto-ingest)

Run:
    pytest tests/test_eval_e2e.py -v -s
    pytest tests/test_eval_e2e.py -v -s -m e2e          # only e2e
    pytest tests/test_eval_e2e.py -v -s -k "judge"      # only LLM judge tests
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from langchain_openai import ChatOpenAI

from app import config
from app.rag.constants import NO_DOCS_ANSWER
from app.rag.conversation import ConversationStore, conversation_store
from app.rag.graph import rag_graph
from app.rag.multiturn_graph import multiturn_rag_graph
from app.rag.tracing import invoke_graph_with_tracing

# ── Markers ───────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.e2e

# ══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════

_JUDGE_LLM: ChatOpenAI | None = None


def _get_judge_llm() -> ChatOpenAI:
    """Return a shared LLM instance used for judging."""
    global _JUDGE_LLM
    if _JUDGE_LLM is None:
        _JUDGE_LLM = ChatOpenAI(
            model=config.LLAMA_CPP_MODEL,
            base_url=config.LLAMA_CPP_BASE_URL,
            api_key=config.LLAMA_CPP_API_KEY,
            temperature=0,
            timeout=config.LLM_TIMEOUT,
        )
    return _JUDGE_LLM


# Rubric dimensions with rating descriptions the LLM judge uses:
_JUDGE_RUBRIC = """
You are an expert evaluation judge. Given a **question**, some **reference documents** (context), and the system's **answer**, score the answer on EACH of the following dimensions using a 1-5 Likert scale.

## Dimensions

1. **Relevance** – Does the answer address the question that was asked?
   1=completely off-topic, 2=tangentially related, 3=partially addresses the question, 4=mostly relevant, 5=directly and fully addresses the question

2. **Groundedness** – Is the answer faithful to the provided context/documents (no hallucinated facts)?
   1=entirely fabricated, 2=mostly fabricated, 3=mix of grounded and fabricated, 4=mostly grounded with minor liberties, 5=fully grounded in the documents

3. **Coherence** – Is the answer well-structured, logical, and easy to understand?
   1=incoherent/gibberish, 2=hard to follow, 3=somewhat organized, 4=clear and logical, 5=excellent structure and flow

4. **Completeness** – Does the answer cover the key points needed to fully answer the question?
   1=missing entirely, 2=covers very little, 3=covers some key points, 4=covers most key points, 5=comprehensive

## Output format
Reply with ONLY a JSON object (no markdown fencing):
{"relevance": <int>, "groundedness": <int>, "coherence": <int>, "completeness": <int>, "reasoning": "<one-sentence justification>"}
""".strip()


def llm_judge(
    question: str,
    answer: str,
    documents: list[str] | None = None,
) -> dict:
    """Ask the LLM judge to score an answer.

    Returns a dict with keys: relevance, groundedness, coherence,
    completeness (ints 1-5), and reasoning (str).
    """
    context_text = "\n\n---\n\n".join(documents) if documents else "(no documents)"

    prompt = (
        f"{_JUDGE_RUBRIC}\n\n"
        f"## Question\n{question}\n\n"
        f"## Context Documents\n{context_text}\n\n"
        f"## Answer\n{answer}\n\n"
        f"## Your evaluation (JSON only):"
    )

    llm = _get_judge_llm()
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Extract JSON from the response (handle possible markdown fencing)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: pessimistic scores when judge fails to produce valid JSON
        scores = {
            "relevance": 1,
            "groundedness": 1,
            "coherence": 1,
            "completeness": 1,
            "reasoning": f"Judge failed to produce valid JSON: {raw[:200]}",
        }

    return scores


def assert_judge_scores(
    scores: dict,
    *,
    min_relevance: int = 3,
    min_groundedness: int = 3,
    min_coherence: int = 3,
    min_completeness: int = 3,
    label: str = "",
):
    """Assert that LLM judge scores meet minimum thresholds."""
    prefix = f"[{label}] " if label else ""
    reasoning = scores.get("reasoning", "")
    assert scores.get("relevance", 0) >= min_relevance, (
        f"{prefix}Relevance {scores.get('relevance')}/{min_relevance}: {reasoning}"
    )
    assert scores.get("groundedness", 0) >= min_groundedness, (
        f"{prefix}Groundedness {scores.get('groundedness')}/{min_groundedness}: {reasoning}"
    )
    assert scores.get("coherence", 0) >= min_coherence, (
        f"{prefix}Coherence {scores.get('coherence')}/{min_coherence}: {reasoning}"
    )
    assert scores.get("completeness", 0) >= min_completeness, (
        f"{prefix}Completeness {scores.get('completeness')}/{min_completeness}: {reasoning}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _run_single_turn(question: str) -> dict:
    """Invoke the single-turn RAG graph end-to-end."""
    return invoke_graph_with_tracing(
        rag_graph,
        {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "grounded": False,
            "retries": 0,
        },
        trace_name="eval-single-turn",
        tags=["eval"],
    )


def _run_multi_turn(question: str, session_id: str) -> dict:
    """Invoke the multi-turn RAG graph end-to-end."""
    return invoke_graph_with_tracing(
        multiturn_rag_graph,
        {
            "question": question,
            "session_id": session_id,
            "intent": "",
            "contextualized_query": "",
            "rewritten_question": "",
            "documents": [],
            "chat_history": "",
            "answer": "",
            "grounded": False,
            "retries": 0,
        },
        trace_name="eval-multi-turn",
        session_id=session_id,
        tags=["eval"],
    )


def _docs_text(result: dict) -> list[str]:
    """Extract page_content strings from result documents."""
    return [d.page_content for d in result.get("documents", [])]


def _answer_contains_any(answer: str, keywords: list[str]) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def _new_session_id() -> str:
    """Create a fresh session in the module-level conversation_store."""
    sid = f"eval-{uuid.uuid4().hex[:8]}"
    conversation_store.get_or_create(sid)
    return sid


# ══════════════════════════════════════════════════════════════════════════════
# 1. SINGLE-TURN: LLM-judged factual retrieval
# ══════════════════════════════════════════════════════════════════════════════


class TestSingleTurnJudged:
    """LLM-judge evaluation of single-turn answers."""

    def test_judge_method_description(self):
        """Ask what A2D does — judge should rate relevance >= 3."""
        q = "What is the Align-to-Distill (A2D) method and what problem does it solve?"
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            min_coherence=3,
            min_completeness=3,
            label="method_description",
        )

    def test_judge_aam_component(self):
        """Ask about AAM — judge should rate groundedness >= 3."""
        q = "What is the Attention Alignment Module (AAM) described in the A2D paper?"
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            min_coherence=3,
            min_completeness=3,
            label="aam_component",
        )

    def test_judge_nmt_context(self):
        """Ask the NLP task — judge should confirm translation/NMT."""
        q = "What NLP task does the A2D paper focus on?"
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            label="nmt_context",
        )

    def test_judge_knowledge_distillation(self):
        """Ask about KD — answer should be comprehensive."""
        q = "Explain the knowledge distillation approach used in the A2D paper."
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            min_coherence=3,
            min_completeness=3,
            label="kd_explanation",
        )

    def test_judge_experimental_results(self):
        """Ask about experimental results — answer should mention baselines/performance."""
        q = (
            "What experimental results does the A2D paper report "
            "on decoder distillation and translation quality?"
        )
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        # Completeness can be low if retrieval doesn't capture all tables
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            min_completeness=2,
            label="experimental_results",
        )

    def test_judge_pointwise_convolution(self):
        """Ask about a specific technical detail.

        Pointwise convolution is a very specific detail that may not
        appear in all retrieved chunks.  The test verifies that:
        • If the system CAN answer → the answer should be high-quality.
        • If the system declines → the decline is acceptable (no hallucination).
        """
        q = (
            "How does the Attention Alignment Module (AAM) in A2D "
            "use pointwise convolution to align attention heads?"
        )
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        # If relevance >= 3, the system answered substantively → check quality
        if scores.get("relevance", 0) >= 3:
            assert_judge_scores(
                scores,
                min_relevance=3,
                min_groundedness=3,
                min_completeness=2,
                label="pointwise_conv",
            )
        else:
            # System declined or gave a generic answer — verify no hallucination
            answer = result["answer"].lower()
            has_fabrication = any(
                kw in answer for kw in ["3x3 convolution", "batch norm", "relu activation"]
            )
            assert not has_fabrication, (
                f"Decline answer should not hallucinate details: {result['answer'][:200]}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE-TURN: out-of-scope / no-knowledge
# ══════════════════════════════════════════════════════════════════════════════


class TestOutOfScope:
    """Questions the system should correctly decline to answer."""

    def test_unrelated_question_handled_gracefully(self):
        """Cooking question -> no docs or polite refusal."""
        result = _run_single_turn("How do I make a chocolate cake from scratch?")
        answer = result["answer"]
        docs = result.get("documents", [])
        # Either the canned answer or disclaims
        no_docs = len(docs) == 0
        disclaimer = _answer_contains_any(
            answer,
            [
                "don't have enough information",
                "cannot answer",
                "no relevant",
                "not found",
                "관련된 문서를 찾을 수 없",
                NO_DOCS_ANSWER[:30],
            ],
        )
        assert no_docs or disclaimer, (
            f"Expected refusal for unrelated question. docs={len(docs)}, answer={answer[:200]}"
        )

    def test_gibberish_query(self):
        """Gibberish -> no confident answer."""
        result = _run_single_turn("xyzzy plugh qwerty 12345 zxcvbn")
        docs = result.get("documents", [])
        if len(docs) == 0:
            assert _answer_contains_any(
                result["answer"],
                [NO_DOCS_ANSWER[:20], "찾을 수 없", "don't have enough"],
            )

    def test_judge_out_of_scope_groundedness(self):
        """LLM judge: an out-of-scope question should NOT get a hallucinated answer."""
        q = "What is the best recipe for sourdough bread?"
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        # If the system correctly refused, that's fine.
        # If it hallucinated a recipe, groundedness should be low.
        if _answer_contains_any(result["answer"], ["bread", "flour", "dough"]):
            assert scores.get("groundedness", 5) <= 2, (
                f"Hallucinated answer should have low groundedness: {scores}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-TURN: conversation flow & intent classification
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiTurnIntentFlow:
    """Test intent classification: new_topic -> follow_up -> clear_context."""

    def test_first_turn_is_new_topic(self):
        sid = _new_session_id()
        result = _run_multi_turn("What is the Align-to-Distill paper about?", sid)
        assert result["intent"] == "new_topic"
        assert _answer_contains_any(result["answer"], ["A2D", "distill", "attention", "knowledge"])

    def test_follow_up_intent(self):
        sid = _new_session_id()
        _run_multi_turn("What is the A2D method?", sid)
        result = _run_multi_turn("What results did it achieve?", sid)
        assert result["intent"] == "follow_up", f"Expected follow_up, got {result['intent']}"

    def test_follow_up_uses_context_judged(self):
        """LLM judge: follow-up resolving a pronoun should produce a relevant answer."""
        sid = _new_session_id()
        _run_multi_turn(
            "Who are the authors of the Align-to-Distill paper published at COLING?",
            sid,
        )
        q2 = "What institution or company are they affiliated with?"
        r2 = _run_multi_turn(q2, sid)
        scores = llm_judge(q2, r2["answer"], _docs_text(r2))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_coherence=3,
            label="follow_up_context",
        )

    def test_clear_context_korean(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)
        result = _run_multi_turn("초기화해줘", sid)
        assert result["intent"] == "clear_context"
        assert "초기화" in result["answer"]

    def test_clear_context_english(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)
        result = _run_multi_turn("Clear the conversation and start over", sid)
        assert result["intent"] == "clear_context"

    def test_new_topic_switch(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)
        result = _run_multi_turn("How is the weather in Seoul today?", sid)
        assert result["intent"] == "new_topic", f"Expected new_topic, got {result['intent']}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-TURN: session persistence via Redis
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionPersistence:
    """Verify turns are persisted to Redis across graph invocations."""

    def test_turns_persisted_after_two_turns(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)
        _run_multi_turn("What about the attention module?", sid)

        reloaded = conversation_store.get(sid)
        assert reloaded is not None
        # Each Q&A pair = human + assistant = 2 Turn objects
        assert len(reloaded.turns) == 4, (
            f"Expected 4 turns (2 pairs), got {len(reloaded.turns)}: "
            f"{[(t.role, t.content[:30]) for t in reloaded.turns]}"
        )

    def test_clear_resets_turns(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)
        _run_multi_turn("초기화", sid)

        reloaded = conversation_store.get(sid)
        assert reloaded is not None
        assert len(reloaded.turns) == 0, f"Expected 0 turns after clear, got {len(reloaded.turns)}"

    def test_session_survives_fresh_store_load(self):
        """A new ConversationStore instance should see the same session."""
        sid = _new_session_id()
        _run_multi_turn("Explain AAM in the A2D paper.", sid)

        # Simulate another process loading from the same Redis
        store2 = ConversationStore(
            redis_url=config.REDIS_URL,
            ttl_seconds=300,
        )
        reloaded = store2.get(sid)
        assert reloaded is not None
        assert len(reloaded.turns) >= 2, f"Expected >= 2 turns, got {len(reloaded.turns)}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. SEMANTIC QUALITY: LLM-judged answer quality
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticQualityJudged:
    """LLM judge scores on diverse question types."""

    def test_judge_english_explanation(self):
        q = (
            "What is the main contribution of the Align-to-Distill (A2D) paper "
            "on knowledge distillation for neural machine translation?"
        )
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        docs = result.get("documents", [])
        if docs:
            assert_judge_scores(
                scores,
                min_relevance=3,
                min_groundedness=3,
                min_coherence=3,
                min_completeness=2,
                label="english_explanation",
            )
        else:
            # Retrieval can occasionally miss — accept canned answer
            assert _answer_contains_any(
                result["answer"],
                [NO_DOCS_ANSWER[:20], "don't have enough", "찾을 수 없"],
            )

    def test_judge_korean_question(self):
        """Korean question -> should produce Korean answer, judged for quality."""
        q = "A2D 논문에서 제안하는 Attention Alignment Module의 역할은 무엇인가요?"
        sid = _new_session_id()
        result = _run_multi_turn(q, sid)
        answer = result["answer"]

        # Check language: must contain Korean characters
        hangul_count = len(re.findall(r"[\uac00-\ud7af]", answer))
        assert hangul_count > 5, (
            f"Expected Korean response. Hangul chars: {hangul_count}, answer: {answer[:200]}"
        )

        scores = llm_judge(q, answer, _docs_text(result))
        print(f"  Scores: {scores}")
        # Completeness may be limited by retrieval coverage
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_coherence=3,
            min_completeness=2,
            label="korean_answer",
        )

    def test_judge_technical_depth(self):
        """A deep technical question should get a detailed, grounded answer."""
        q = (
            "How does A2D handle the feature mapping problem between teacher "
            "and student models with different numbers of layers and heads?"
        )
        result = _run_single_turn(q)
        scores = llm_judge(q, result["answer"], _docs_text(result))
        print(f"  Scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_groundedness=3,
            min_coherence=3,
            min_completeness=3,
            label="technical_depth",
        )

    def test_no_hallucinated_authors(self):
        """LLM judge: asking about authors should not fabricate names."""
        q = "List all authors of the Align-to-Distill paper."
        result = _run_single_turn(q)
        answer = result["answer"]

        # Use LLM judge to check if the answer fabricates information
        judge_q = (
            "The following answer was generated about the authors of the "
            "Align-to-Distill paper. Does the answer fabricate any author names "
            "that are NOT present in the provided context documents? "
            "Score groundedness strictly."
        )
        scores = llm_judge(judge_q, answer, _docs_text(result))
        print(f"  Scores: {scores}")
        assert scores.get("groundedness", 0) >= 3, f"Author answer may be hallucinated: {scores}"

    def test_no_template_variables_in_answer(self):
        """Answer should never contain raw template variables."""
        result = _run_single_turn("What is the A2D method?")
        answer = result["answer"]
        for var in ["{context}", "{question}", "{history}", "{documents}"]:
            assert var not in answer, f"Raw template variable {var} in: {answer[:200]}"

    def test_answer_stays_on_topic(self):
        """Asking about A2D should not mention cooking, sports, etc."""
        result = _run_single_turn("What experiments were conducted in the A2D paper?")
        answer = result["answer"].lower()
        off_topic = ["recipe", "football", "basketball", "cooking", "weather forecast"]
        for term in off_topic:
            assert term not in answer, f"Off-topic '{term}' in answer: {answer[:300]}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. GROUNDING & RETRIEVAL: pipeline correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestGroundingRetrieval:
    """Verify the retrieve->generate->grade loop works correctly."""

    def test_documents_are_retrieved(self):
        result = _run_single_turn("Explain knowledge distillation in the A2D paper.")
        docs = result.get("documents", [])
        assert len(docs) > 0, "Expected at least one retrieved document"

    def test_retrieved_docs_are_relevant(self):
        result = _run_single_turn("What is knowledge distillation in NMT?")
        docs = result.get("documents", [])
        if docs:
            combined = " ".join(d.page_content.lower() for d in docs)
            assert _answer_contains_any(
                combined, ["distillation", "knowledge", "student", "teacher"]
            ), f"Docs seem irrelevant: {combined[:300]}"

    def test_retry_mechanism_produces_answer(self):
        result = _run_single_turn(
            "How does A2D compare to other knowledge distillation baselines "
            "in low-resource translation settings?"
        )
        assert len(result["answer"].strip()) > 10
        assert result.get("retries", 0) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. MULTI-TURN: extended conversation (3+ turns), LLM-judged
# ══════════════════════════════════════════════════════════════════════════════


class TestExtendedConversationJudged:
    """Simulate realistic multi-turn research conversations with LLM judging."""

    def test_three_turn_deep_dive(self):
        sid = _new_session_id()

        # Turn 1: overview
        r1 = _run_multi_turn("What is the A2D paper about?", sid)
        assert _answer_contains_any(r1["answer"], ["distill", "A2D", "attention", "knowledge"])

        # Turn 2: detail
        q2 = "How does the attention alignment module work specifically?"
        r2 = _run_multi_turn(q2, sid)
        s2 = llm_judge(q2, r2["answer"], _docs_text(r2))
        print(f"  Turn 2 scores: {s2}")
        if s2.get("relevance", 0) >= 3:
            assert_judge_scores(s2, min_relevance=3, min_coherence=3, label="deep_dive_t2")
        else:
            # LLM declined despite docs — no hallucination check
            assert len(r2["answer"].strip()) > 10

        # Turn 3: results
        q3 = "What were the main experimental results?"
        r3 = _run_multi_turn(q3, sid)
        s3 = llm_judge(q3, r3["answer"], _docs_text(r3))
        print(f"  Turn 3 scores: {s3}")
        if s3.get("relevance", 0) >= 3:
            assert_judge_scores(
                s3, min_relevance=3, min_coherence=3, min_completeness=2, label="deep_dive_t3"
            )
        else:
            assert len(r3["answer"].strip()) > 10

    def test_clear_then_fresh_start(self):
        sid = _new_session_id()
        _run_multi_turn("Tell me about the A2D paper.", sid)
        r2 = _run_multi_turn("리셋해줘", sid)
        assert r2["intent"] == "clear_context"

        r3 = _run_multi_turn("What is knowledge distillation?", sid)
        assert r3["intent"] == "new_topic"

    def test_topic_switch_and_return(self):
        sid = _new_session_id()
        _run_multi_turn("What is A2D?", sid)

        r2 = _run_multi_turn("What is the capital of France?", sid)
        assert r2["intent"] == "new_topic"

        q3 = "Going back to A2D, what is the Attention Alignment Module?"
        r3 = _run_multi_turn(q3, sid)
        scores = llm_judge(q3, r3["answer"], _docs_text(r3))
        print(f"  Return-to-topic scores: {scores}")
        # After topic switch, the system should still answer about A2D
        if scores.get("relevance", 0) >= 3:
            assert_judge_scores(
                scores,
                min_relevance=3,
                min_coherence=3,
                min_completeness=2,
                label="topic_return",
            )
        else:
            # LLM declined — accept as long as answer is non-empty
            assert len(r3["answer"].strip()) > 10


# ══════════════════════════════════════════════════════════════════════════════
# 8. API ENDPOINT E2E (FastAPI TestClient, real graphs)
# ══════════════════════════════════════════════════════════════════════════════


class TestAPIEndToEnd:
    """Hit the actual FastAPI endpoints with real graph execution."""

    @pytest.fixture(autouse=True)
    def _client(self):
        from contextlib import asynccontextmanager

        from fastapi.testclient import TestClient

        from app.main import app

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        original = app.router.lifespan_context
        app.router.lifespan_context = _noop_lifespan
        try:
            with TestClient(app, raise_server_exceptions=True) as c:
                self.client = c
                yield
        finally:
            app.router.lifespan_context = original

    def test_ask_endpoint_judged(self):
        """POST /ask with LLM-judge scoring."""
        q = "What is the A2D paper about?"
        resp = self.client.post("/ask", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["answer"]) > 20

        scores = llm_judge(q, data["answer"], data.get("source_documents", []))
        print(f"  /ask scores: {scores}")
        assert_judge_scores(
            scores,
            min_relevance=3,
            min_coherence=3,
            label="api_ask",
        )

    def test_chat_two_turns_judged(self):
        """POST /chat: two turns, judge the follow-up."""
        r1 = self.client.post(
            "/chat",
            json={"question": "What is the Align-to-Distill knowledge distillation method?"},
        )
        assert r1.status_code == 200
        sid = r1.json()["session_id"]

        q2 = "How does it handle attention head alignment between teacher and student?"
        r2 = self.client.post(
            "/chat",
            json={"question": q2, "session_id": sid},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["turn_number"] >= 2

        scores = llm_judge(q2, d2["answer"], d2.get("source_documents", []))
        print(f"  /chat turn-2 scores: {scores}")
        # API follow-up: completeness may be limited by retrieval
        if d2.get("source_documents"):
            assert_judge_scores(
                scores,
                min_relevance=3,
                min_coherence=3,
                min_completeness=2,
                label="api_chat_t2",
            )
        else:
            # Accept canned answer when retrieval fails
            assert len(d2["answer"].strip()) > 10

    def test_chat_clear_then_new_topic(self):
        r1 = self.client.post("/chat", json={"question": "What is A2D?"})
        sid = r1.json()["session_id"]

        r_clear = self.client.post("/chat/clear", json={"session_id": sid})
        assert r_clear.status_code == 200

        r2 = self.client.post(
            "/chat",
            json={"question": "What is knowledge distillation?", "session_id": sid},
        )
        assert r2.json()["intent"] == "new_topic"

    def test_health(self):
        assert self.client.get("/health").json()["status"] == "ok"
