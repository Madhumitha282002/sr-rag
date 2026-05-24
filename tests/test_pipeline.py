"""
tests/test_pipeline.py
Tests for src/pipeline.py
Run from project root: pytest tests/test_pipeline.py -v
"""

import pytest
from src.pipeline import SRRagPipeline


# ---------------------------------------------------------------------------
# Fixture — one pipeline instance shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    return SRRagPipeline(
        persist_dir="vector_store",
        collection_name="sr_papers",
        llm_provider="mock",
    )


# ---------------------------------------------------------------------------
# pipeline.query
# ---------------------------------------------------------------------------

class TestPipelineQuery:

    def test_returns_required_keys(self, pipeline):
        result = pipeline.query("What loss does SRGAN use?")
        required = {
            "question", "answer", "answer_full", "sources",
            "citations_valid", "token_usage",
            "retrieval_ms", "generation_ms", "total_ms",
            "provider", "model",
        }
        assert required.issubset(result.keys())

    def test_answer_is_non_empty_string(self, pipeline):
        result = pipeline.query("What is perceptual loss?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_list_populated(self, pipeline):
        result = pipeline.query("How does EDSR differ from SRResNet?")
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

    def test_sources_have_citation_index(self, pipeline):
        result = pipeline.query("What is PSNR?")
        for i, src in enumerate(result["sources"]):
            assert "citation_index" in src
            assert src["citation_index"] == i + 1

    def test_answer_full_contains_references(self, pipeline):
        result = pipeline.query("What datasets does RCAN use?")
        assert "References" in result["answer_full"]

    def test_token_usage_keys(self, pipeline):
        result = pipeline.query("test question")
        usage = result["token_usage"]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"]:
            assert key in usage

    def test_latencies_are_positive(self, pipeline):
        result = pipeline.query("test")
        assert result["retrieval_ms"] >= 0
        assert result["generation_ms"] >= 0
        assert result["total_ms"] >= 0

    def test_total_ms_at_least_sum_of_parts(self, pipeline):
        result = pipeline.query("test")
        assert result["total_ms"] >= result["retrieval_ms"]

    def test_provider_recorded(self, pipeline):
        result = pipeline.query("test")
        assert result["provider"] == "mock"

    def test_include_raw_chunks(self, pipeline):
        result = pipeline.query("test", include_raw_chunks=True)
        assert "raw_chunks" in result
        assert isinstance(result["raw_chunks"], list)

    def test_raw_chunks_excluded_by_default(self, pipeline):
        result = pipeline.query("test")
        assert "raw_chunks" not in result

    def test_top_k_controls_source_count(self, pipeline):
        result = pipeline.query("test", top_k=3)
        assert len(result["sources"]) <= 3

    def test_citations_valid_is_bool(self, pipeline):
        result = pipeline.query("test")
        assert isinstance(result["citations_valid"], bool)


# ---------------------------------------------------------------------------
# pipeline.query_with_filter
# ---------------------------------------------------------------------------

class TestPipelineQueryWithFilter:

    def test_method_filter(self, pipeline):
        result = pipeline.query_with_filter(
            "loss function", method="SRGAN", top_k=5
        )
        for src in result["sources"]:
            assert src["method"] == "SRGAN", \
                f"Expected SRGAN, got {src['method']}"

    def test_year_filter(self, pipeline):
        result = pipeline.query_with_filter(
            "attention mechanism", year_from=2021, top_k=5
        )
        for src in result["sources"]:
            assert src["year"] >= 2021, \
                f"Expected year >= 2021, got {src['year']}"

    def test_no_filter_same_as_query(self, pipeline):
        result = pipeline.query_with_filter("perceptual loss")
        assert "answer" in result


# ---------------------------------------------------------------------------
# pipeline.info
# ---------------------------------------------------------------------------

class TestPipelineInfo:

    def test_returns_info_dict(self, pipeline):
        info = pipeline.info()
        for key in ["collection_name", "total_chunks", "embedding_model"]:
            assert key in info

    def test_total_chunks_positive(self, pipeline):
        info = pipeline.info()
        assert info["total_chunks"] > 0