"""
src/retrieval/reranker.py
--------------------------
Cross-encoder reranker that re-scores retrieved chunks using a
query-document pair model. Significantly improves precision over
dense retrieval alone, at the cost of ~300-500ms extra latency.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Lightweight (22M params), fast on CPU
  - Strong performance on technical text
  - No GPU required for a 10-paper corpus

Workflow:
  1. Dense retriever fetches top-K candidates (e.g. top-20)
  2. Cross-encoder rescores all candidates against the query
  3. Top-k' results returned (e.g. top-5) in reranked order

Note on hybrid search (future improvement):
  Dense retrieval can miss exact keyword matches for technical terms
  like "RCAN" or "perceptual loss" when they are rare in the embedding
  space. BM25 + dense hybrid search is the standard fix — worth naming
  in the README as a planned improvement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    """
    Load a cross-encoder model, caching it in memory.
    Second call with same model_name returns instantly.
    """
    if model_name not in _reranker_cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Run: pip install sentence-transformers"
            )
        logger.info("Loading reranker model: %s", model_name)
        t0 = time.time()
        _reranker_cache[model_name] = CrossEncoder(model_name)
        logger.info("Reranker loaded in %.1f s", time.time() - t0)

    return _reranker_cache[model_name]


# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------

class Reranker:
    """
    Cross-encoder reranker. Instantiate once and reuse.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = None   # lazy-loaded

    @property
    def model(self):
        if self._model is None:
            self._model = load_reranker(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks using the cross-encoder.

        Args:
            query  : the original query string
            chunks : list of chunk dicts (must have 'text' key)
            top_k  : how many to return after reranking.
                     None = return all, reordered.

        Returns chunks sorted by reranker score (descending),
        with 'reranker_score' and original 'score' both preserved.
        """
        if not chunks:
            return []

        t0 = time.time()

        # Build query-document pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs in one batch
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach reranker scores and sort
        reranked = []
        for chunk, score in zip(chunks, scores):
            reranked.append({
                **chunk,
                "dense_score":    chunk.get("score", 0.0),   # original retrieval score
                "reranker_score": float(score),
                "score":          float(score),               # overwrite for downstream compat
            })

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Re-assign citation indices after reordering
        for i, chunk in enumerate(reranked):
            chunk["citation_index"] = i + 1

        elapsed = (time.time() - t0) * 1000
        logger.info(
            "Reranked %d chunks in %.0f ms. Top score: %.3f -> %.3f",
            len(chunks),
            elapsed,
            chunks[0].get("score", 0) if chunks else 0,
            reranked[0]["reranker_score"] if reranked else 0,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair. Useful for debugging."""
        scores = self.model.predict([(query, text)], show_progress_bar=False)
        return float(scores[0])


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------

_default_reranker: Reranker | None = None


def get_reranker(**kwargs) -> Reranker:
    """Return a module-level singleton reranker."""
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker(**kwargs)
    return _default_reranker


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """One-liner reranking using the default singleton reranker."""
    return get_reranker().rerank(query, chunks, top_k=top_k)