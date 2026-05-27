"""
tests/test_prompt_manager.py
Tests for src/generation/prompt_manager.py
Run from project root: pytest tests/test_prompt_manager.py -v
"""

import pytest
from src.generation.prompt_manager import (
    build_context,
    build_prompt,
    should_refuse,
    prompt_stats,
    REFUSAL_MESSAGE,
    MAX_CONTEXT_CHARS,
    MIN_ANSWER_SCORE,
    SYSTEM_PROMPTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_chunks():
    """Chunks with high scores — should pass refusal check."""
    return [
        {
            "chunk_id": f"srgan_2016_p0{i}_c00",
            "file_name": "srgan_2016.pdf",
            "method": "SRGAN",
            "year": 2016,
            "page_number": i,
            "page_count": 16,
            "title": "SRGAN",
            "authors": "Ledig et al.",
            "venue": "CVPR",
            "score": 0.75 - i * 0.05,
            "citation_index": i + 1,
            "text": f"Perceptual loss and adversarial training are used in chunk {i}. " * 20,
            "char_count": 800,
            "word_count": 120,
        }
        for i in range(4)
    ]


@pytest.fixture
def weak_chunks():
    """Chunks with low scores — should trigger refusal."""
    return [
        {
            "chunk_id": "srgan_2016_p01_c00",
            "file_name": "srgan_2016.pdf",
            "method": "SRGAN",
            "year": 2016,
            "page_number": 1,
            "page_count": 16,
            "title": "SRGAN",
            "authors": "Ledig et al.",
            "venue": "CVPR",
            "score": 0.15,   # below MIN_ANSWER_SCORE
            "citation_index": 1,
            "text": "Some loosely related text.",
            "char_count": 30,
            "word_count": 5,
        }
    ]


@pytest.fixture
def large_chunks():
    """Chunks whose combined text exceeds MAX_CONTEXT_CHARS."""
    return [
        {
            "chunk_id": f"x_p0{i}_c00",
            "file_name": "x.pdf",
            "method": "BigNet",
            "year": 2020,
            "page_number": i,
            "page_count": 10,
            "title": "X",
            "authors": "A",
            "venue": "B",
            "score": 0.9 - i * 0.01,
            "citation_index": i + 1,
            "text": "word " * 800,    # ~4000 chars each
            "char_count": 4000,
            "word_count": 800,
        }
        for i in range(5)   # 5 * 4000 = 20000 chars > MAX_CONTEXT_CHARS
    ]


# ---------------------------------------------------------------------------
# should_refuse
# ---------------------------------------------------------------------------

class TestShouldRefuse:

    def test_refuses_empty_chunks(self):
        refuse, reason = should_refuse([])
        assert refuse is True
        assert reason == "no_chunks"

    def test_refuses_low_score(self, weak_chunks):
        refuse, reason = should_refuse(weak_chunks)
        assert refuse is True
        assert "low_score" in reason

    def test_passes_good_score(self, good_chunks):
        refuse, reason = should_refuse(good_chunks)
        assert refuse is False
        assert reason == "ok"

    def test_custom_threshold(self, good_chunks):
        # Raise threshold above good_chunks top score (0.75)
        refuse, _ = should_refuse(good_chunks, min_score=0.99)
        assert refuse is True


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_includes_all_small_chunks(self, good_chunks):
        _, included = build_context(good_chunks, max_chars=MAX_CONTEXT_CHARS)
        assert len(included) == len(good_chunks)

    def test_respects_char_budget(self, large_chunks):
        context, included = build_context(large_chunks, max_chars=MAX_CONTEXT_CHARS)
        assert len(context) <= MAX_CONTEXT_CHARS + 500   # small buffer for headers

    def test_truncates_when_budget_exceeded(self, large_chunks):
        _, included = build_context(large_chunks, max_chars=MAX_CONTEXT_CHARS)
        assert len(included) < len(large_chunks)

    def test_context_contains_method_name(self, good_chunks):
        context, _ = build_context(good_chunks[:1])
        assert "SRGAN" in context

    def test_context_contains_page_number(self, good_chunks):
        context, _ = build_context(good_chunks[:1])
        assert "page" in context.lower()

    def test_empty_chunks_returns_empty(self):
        context, included = build_context([])
        assert context == ""
        assert included == []


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:

    def test_returns_four_values(self, good_chunks):
        result = build_prompt("What is SRGAN?", good_chunks)
        assert len(result) == 4

    def test_not_refused_with_good_chunks(self, good_chunks):
        _, _, _, refused = build_prompt("What is SRGAN?", good_chunks)
        assert refused is False

    def test_refused_with_empty_chunks(self):
        _, user, included, refused = build_prompt("test", [])
        assert refused is True
        assert user == REFUSAL_MESSAGE
        assert included == []

    def test_refused_with_weak_chunks(self, weak_chunks):
        _, user, _, refused = build_prompt("test", weak_chunks)
        assert refused is True

    def test_question_in_user_prompt(self, good_chunks):
        _, user, _, _ = build_prompt("What loss does SRGAN use?", good_chunks)
        assert "What loss does SRGAN use" in user

    def test_system_prompt_returned(self, good_chunks):
        system, _, _, _ = build_prompt("test", good_chunks, template="v1")
        assert system == SYSTEM_PROMPTS["v1"]

    def test_template_v2_default(self, good_chunks):
        system, _, _, _ = build_prompt("test", good_chunks)
        assert system == SYSTEM_PROMPTS["v2"]

    def test_template_v3_concise(self, good_chunks):
        system, _, _, _ = build_prompt("test", good_chunks, template="v3_concise")
        assert system == SYSTEM_PROMPTS["v3_concise"]

    def test_unknown_template_falls_back(self, good_chunks):
        system, _, _, _ = build_prompt("test", good_chunks, template="nonexistent")
        assert system == SYSTEM_PROMPTS["v2"]


# ---------------------------------------------------------------------------
# prompt_stats
# ---------------------------------------------------------------------------

class TestPromptStats:

    def test_returns_required_keys(self, good_chunks):
        stats = prompt_stats("test", good_chunks)
        for key in ["template", "refused", "chunks_in", "chunks_used",
                    "system_chars", "user_chars", "total_chars", "top_score"]:
            assert key in stats

    def test_refused_true_for_empty(self):
        stats = prompt_stats("test", [])
        assert stats["refused"] is True

    def test_chunks_used_lte_chunks_in(self, good_chunks):
        stats = prompt_stats("test", good_chunks)
        assert stats["chunks_used"] <= stats["chunks_in"]

    def test_total_chars_reasonable(self, good_chunks):
        stats = prompt_stats("test", good_chunks)
        assert stats["total_chars"] < MAX_CONTEXT_CHARS + 2000