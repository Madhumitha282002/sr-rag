"""
tests/test_retriever.py
Tests for src/retrieval/retriever.py
Run from project root: pytest tests/test_retriever.py -v
"""

import pytest
from src.retrieval.retriever import (
    Retriever,
    preprocess_query,
    retrieve,
    _deduplicate,
)

VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME  = "sr_papers"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    return Retriever(
        persist_dir=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# preprocess_query
# ---------------------------------------------------------------------------

class TestPreprocessQuery:

    def test_strips_whitespace(self):
        assert preprocess_query("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert preprocess_query("what  is   SRGAN") == "what is SRGAN"

    def test_removes_trailing_question_mark(self):
        assert preprocess_query("What is PSNR?") == "What is PSNR"

    def test_preserves_technical_terms(self):
        result = preprocess_query("How does RCAN use channel attention?")
        assert "RCAN" in result
        assert "channel attention" in result

    def test_empty_string(self):
        assert preprocess_query("") == ""


# ---------------------------------------------------------------------------
# Retriever.retrieve
# ---------------------------------------------------------------------------

class TestRetriever:

    def test_returns_dict_with_required_keys(self, retriever):
        result = retriever.retrieve("perceptual loss SRGAN")
        for key in ["query", "results", "latency_ms", "total_chunks"]:
            assert key in result

    def test_returns_correct_number_of_results(self, retriever):
        result = retriever.retrieve("residual network", top_k=3)
        assert len(result["results"]) <= 3

    def test_results_have_required_keys(self, retriever):
        result = retriever.retrieve("transformer attention")
        for r in result["results"]:
            for key in ["text", "score", "method", "page_number", "citation_index"]:
                assert key in r, f"Missing key '{key}' in result"

    def test_scores_are_descending(self, retriever):
        result = retriever.retrieve("GAN loss function", top_k=5)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_scores_between_0_and_1(self, retriever):
        result = retriever.retrieve("bicubic upsampling baseline")
        for r in result["results"]:
            assert 0 <= r["score"] <= 1

    def test_citation_index_starts_at_1(self, retriever):
        result = retriever.retrieve("PSNR evaluation metric")
        assert result["results"][0]["citation_index"] == 1

    def test_citation_indices_sequential(self, retriever):
        result = retriever.retrieve("image super resolution")
        indices = [r["citation_index"] for r in result["results"]]
        assert indices == list(range(1, len(indices) + 1))

    def test_latency_is_positive(self, retriever):
        result = retriever.retrieve("test query")
        assert result["latency_ms"] >= 0

    def test_total_chunks_positive(self, retriever):
        result = retriever.retrieve("test")
        assert result["total_chunks"] > 0

    def test_query_cleaned_in_output(self, retriever):
        result = retriever.retrieve("  What is PSNR?  ")
        assert result["query"] == "What is PSNR"


# ---------------------------------------------------------------------------
# Retriever.retrieve_by_method
# ---------------------------------------------------------------------------

class TestRetrieveByMethod:

    def test_filters_to_correct_method(self, retriever):
        result = retriever.retrieve_by_method("loss function", "SRGAN")
        for r in result["results"]:
            assert r["method"] == "SRGAN", \
                f"Expected SRGAN, got {r['method']}"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def test_limits_per_page(self):
        chunks = [
            {"file_name": "a.pdf", "page_number": 1, "score": 0.9, "text": "t1"},
            {"file_name": "a.pdf", "page_number": 1, "score": 0.8, "text": "t2"},
            {"file_name": "a.pdf", "page_number": 1, "score": 0.7, "text": "t3"},
            {"file_name": "b.pdf", "page_number": 2, "score": 0.6, "text": "t4"},
        ]
        result = _deduplicate(chunks, max_per_page=2)
        from_page_1 = [r for r in result if r["file_name"] == "a.pdf"]
        assert len(from_page_1) == 2

    def test_preserves_order(self):
        chunks = [
            {"file_name": "a.pdf", "page_number": 1, "score": 0.9, "text": "t1"},
            {"file_name": "b.pdf", "page_number": 2, "score": 0.8, "text": "t2"},
        ]
        result = _deduplicate(chunks)
        assert result[0]["score"] > result[1]["score"]

    def test_empty_input(self):
        assert _deduplicate([]) == []


# ---------------------------------------------------------------------------
# collection_info
# ---------------------------------------------------------------------------

class TestCollectionInfo:

    def test_returns_info_dict(self, retriever):
        info = retriever.collection_info()
        assert "collection_name" in info
        assert "total_chunks" in info
        assert info["total_chunks"] > 0