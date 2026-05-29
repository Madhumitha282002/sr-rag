"""
tests/test_reranker.py
Tests for src/retrieval/reranker.py
Run from project root: pytest tests/test_reranker.py -v

Note: first run downloads the cross-encoder model (~90MB).
Subsequent runs use the cached model and are fast.
"""

import pytest
from src.retrieval.reranker import Reranker, rerank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reranker():
    return Reranker()


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": f"srgan_p0{i}_c00",
            "file_name": "srgan_2016.pdf",
            "method": "SRGAN",
            "year": 2016,
            "page_number": i,
            "score": 0.8 - i * 0.1,
            "citation_index": i + 1,
            "text": text,
        }
        for i, text in enumerate([
            "We propose a perceptual loss function consisting of adversarial loss and content loss for super-resolution.",
            "The generator network upscales the low-resolution input image using residual blocks.",
            "We evaluate our method on Set5 and BSD100 datasets using PSNR and SSIM metrics.",
            "The training procedure uses Adam optimiser with a learning rate of 1e-4.",
            "Related work includes bicubic interpolation and sparse coding approaches.",
        ])
    ]


# ---------------------------------------------------------------------------
# Reranker.rerank
# ---------------------------------------------------------------------------

class TestReranker:

    def test_returns_list(self, reranker, sample_chunks):
        result = reranker.rerank("What loss does SRGAN use?", sample_chunks)
        assert isinstance(result, list)

    def test_same_length_as_input(self, reranker, sample_chunks):
        result = reranker.rerank("perceptual loss", sample_chunks)
        assert len(result) == len(sample_chunks)

    def test_top_k_limits_output(self, reranker, sample_chunks):
        result = reranker.rerank("perceptual loss", sample_chunks, top_k=3)
        assert len(result) == 3

    def test_reranker_score_attached(self, reranker, sample_chunks):
        result = reranker.rerank("perceptual loss", sample_chunks)
        for r in result:
            assert "reranker_score" in r
            assert isinstance(r["reranker_score"], float)

    def test_dense_score_preserved(self, reranker, sample_chunks):
        result = reranker.rerank("perceptual loss", sample_chunks)
        for r in result:
            assert "dense_score" in r

    def test_sorted_by_reranker_score(self, reranker, sample_chunks):
        result = reranker.rerank("perceptual loss function adversarial", sample_chunks)
        scores = [r["reranker_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_citation_indices_reassigned(self, reranker, sample_chunks):
        result = reranker.rerank("test", sample_chunks)
        indices = [r["citation_index"] for r in result]
        assert indices == list(range(1, len(result) + 1))

    def test_most_relevant_chunk_ranks_higher(self, reranker, sample_chunks):
        # The chunk about perceptual loss should rank higher for this query
        result = reranker.rerank("What perceptual loss does SRGAN use?", sample_chunks)
        top_text = result[0]["text"].lower()
        assert "perceptual" in top_text or "loss" in top_text

    def test_empty_chunks_returns_empty(self, reranker):
        result = reranker.rerank("test query", [])
        assert result == []

    def test_score_pair_returns_float(self, reranker):
        score = reranker.score_pair("What is PSNR?", "PSNR is a metric for image quality.")
        assert isinstance(score, float)

    def test_relevant_pair_scores_higher(self, reranker):
        relevant = reranker.score_pair(
            "What loss does SRGAN use?",
            "SRGAN uses perceptual loss and adversarial loss for training.",
        )
        irrelevant = reranker.score_pair(
            "What loss does SRGAN use?",
            "The weather today is sunny and warm.",
        )
        assert relevant > irrelevant


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------

class TestRerankFunction:

    def test_rerank_function_works(self, sample_chunks):
        result = rerank("perceptual loss", sample_chunks, top_k=3)
        assert len(result) == 3
        assert "reranker_score" in result[0]