"""
tests/test_answer_generator.py
Tests for src/generation/answer_generator.py and src/generation/citations.py
Run from project root: pytest tests/test_answer_generator.py -v
"""

import pytest
from src.generation.answer_generator import (
    build_prompt,
    generate_answer,
)
from src.generation.citations import (
    format_citations,
    extract_cited_indices,
    validate_citations,
    format_answer_with_citations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id":       "srgan_2016_p04_c01",
            "chunk_index":    0,
            "file_name":      "srgan_2016.pdf",
            "title":          "Photo-Realistic SR Using GANs",
            "method":         "SRGAN",
            "authors":        "Ledig et al.",
            "year":           2016,
            "venue":          "CVPR",
            "page_number":    4,
            "page_count":     16,
            "text":           (
                "We propose a perceptual loss function which consists of an "
                "adversarial loss and a content loss. The adversarial loss pushes "
                "our solution to the natural image manifold using a discriminator "
                "network that is trained to differentiate between the super-resolved "
                "images and original photo-realistic images."
            ),
            "char_count":     300,
            "word_count":     52,
            "citation_index": 1,
        },
        {
            "chunk_id":       "swinir_2021_p03_c02",
            "chunk_index":    1,
            "file_name":      "swinir_2021.pdf",
            "title":          "SwinIR: Image Restoration Using Swin Transformer",
            "method":         "SwinIR",
            "authors":        "Liang et al.",
            "year":           2021,
            "venue":          "ICCVW",
            "page_number":    3,
            "page_count":     14,
            "text":           (
                "SwinIR is based on the Swin Transformer and consists of three "
                "parts: shallow feature extraction, deep feature extraction, and "
                "high-quality image reconstruction. The deep feature extraction "
                "module is composed of several residual Swin Transformer blocks."
            ),
            "char_count":     290,
            "word_count":     48,
            "citation_index": 2,
        },
    ]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:

    def test_returns_tuple(self, sample_chunks):
        prompt, included = build_prompt("What is perceptual loss?", sample_chunks)
        assert isinstance(prompt, str)
        assert isinstance(included, list)

    def test_question_in_prompt(self, sample_chunks):
        prompt, _ = build_prompt("What is perceptual loss?", sample_chunks)
        assert "perceptual loss" in prompt.lower()

    def test_chunk_text_in_prompt(self, sample_chunks):
        prompt, _ = build_prompt("test question", sample_chunks)
        assert "adversarial loss" in prompt

    def test_all_chunks_included_when_small(self, sample_chunks):
        _, included = build_prompt("test", sample_chunks)
        assert len(included) == len(sample_chunks)

    def test_context_truncated_when_too_large(self):
        big_chunks = [
            {
                "chunk_id": f"x_p01_c{i:02d}", "chunk_index": i,
                "file_name": "big.pdf", "title": "Big", "method": "BigNet",
                "authors": "A", "year": 2020, "venue": "X",
                "page_number": 1, "page_count": 10,
                "text": "word " * 1000,
                "char_count": 5000, "word_count": 1000, "citation_index": i + 1,
            }
            for i in range(10)
        ]
        prompt, included = build_prompt("test", big_chunks)
        assert len(included) < 10


# ---------------------------------------------------------------------------
# generate_answer (mock mode — no API key needed)
# ---------------------------------------------------------------------------

class TestGenerateAnswer:

    def test_returns_required_keys(self, sample_chunks):
        result = generate_answer(
            question="What loss does SRGAN use?",
            retrieved_chunks=sample_chunks,
            provider="mock",
        )
        for key in ["answer", "sources", "token_usage", "latency_ms", "provider", "model"]:
            assert key in result, f"Missing key: {key}"

    def test_answer_is_string(self, sample_chunks):
        result = generate_answer("test", sample_chunks, provider="mock")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_match_chunks(self, sample_chunks):
        result = generate_answer("test", sample_chunks, provider="mock")
        assert len(result["sources"]) == len(sample_chunks)

    def test_token_usage_keys(self, sample_chunks):
        result = generate_answer("test", sample_chunks, provider="mock")
        usage = result["token_usage"]
        assert "prompt_tokens"     in usage
        assert "completion_tokens" in usage
        assert "total_tokens"      in usage
        assert "estimated_cost_usd" in usage

    def test_latency_is_positive(self, sample_chunks):
        result = generate_answer("test", sample_chunks, provider="mock")
        assert result["latency_ms"] >= 0

    def test_provider_recorded(self, sample_chunks):
        result = generate_answer("test", sample_chunks, provider="mock")
        assert result["provider"] == "mock"

    def test_empty_chunks_still_returns(self):
        result = generate_answer("test", [], provider="mock")
        assert "answer" in result


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

class TestFormatCitations:

    def test_returns_list(self, sample_chunks):
        citations = format_citations(sample_chunks)
        assert isinstance(citations, list)
        assert len(citations) == len(sample_chunks)

    def test_citation_contains_method(self, sample_chunks):
        citations = format_citations(sample_chunks)
        assert "SRGAN" in citations[0]
        assert "SwinIR" in citations[1]

    def test_citation_contains_year(self, sample_chunks):
        citations = format_citations(sample_chunks)
        assert "2016" in citations[0]
        assert "2021" in citations[1]


class TestExtractCitedIndices:

    def test_single_citation(self):
        assert extract_cited_indices("see [1] for details") == [1]

    def test_multiple_citations(self):
        assert extract_cited_indices("[1] and [2] and [3]") == [1, 2, 3]

    def test_deduplication(self):
        assert extract_cited_indices("[1] is like [1]") == [1]

    def test_no_citations(self):
        assert extract_cited_indices("no references here") == []


class TestValidateCitations:

    def test_valid_when_all_cited(self, sample_chunks):
        answer = "SRGAN uses perceptual loss [1]. SwinIR uses transformers [2]."
        report = validate_citations(answer, sample_chunks)
        assert report["valid"] is True
        assert report["missing"] == []

    def test_invalid_when_hallucinated_index(self, sample_chunks):
        answer = "Some paper [5] shows this."
        report = validate_citations(answer, sample_chunks)
        assert report["valid"] is False
        assert 5 in report["missing"]

    def test_unused_sources_reported(self, sample_chunks):
        answer = "Only [1] is cited."
        report = validate_citations(answer, sample_chunks)
        assert 2 in report["unused_sources"]


class TestFormatAnswerWithCitations:

    def test_appends_references_section(self, sample_chunks):
        result = format_answer_with_citations("The answer is X [1].", sample_chunks)
        assert "References" in result
        assert "SRGAN" in result

    def test_empty_sources_returns_answer_unchanged(self):
        answer = "Plain answer."
        result = format_answer_with_citations(answer, [])
        assert result == answer