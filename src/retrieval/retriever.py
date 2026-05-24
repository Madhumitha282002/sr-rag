"""
src/retrieval/retriever.py
---------------------------
Clean interface for retrieving relevant chunks from ChromaDB.
Wraps embed_query + query_collection with query preprocessing,
optional metadata filtering, and result post-processing.

This is the single entry point for retrieval — the pipeline,
Streamlit app, and FastAPI all call this, never the indexing
modules directly.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.indexing.embeddings import embed_query
from src.indexing.vector_store import load_vector_store, query_collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Stateful retriever that holds open connections to the embedding
    model and ChromaDB collection. Instantiate once and reuse.
    """

    def __init__(
        self,
        persist_dir: str = "vector_store",
        collection_name: str = "sr_papers",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir      = persist_dir
        self.collection_name  = collection_name
        self.embedding_model  = embedding_model
        self._collection      = None   # lazy-loaded

    @property
    def collection(self):
        if self._collection is None:
            self._collection = load_vector_store(
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )
        return self._collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
        deduplicate: bool = True,
    ) -> dict[str, Any]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            query       : natural language question
            top_k       : number of chunks to return
            where       : optional ChromaDB metadata filter
                          e.g. {"method": "SRGAN"}
                          e.g. {"year": {"$gte": 2020}}
            deduplicate : remove near-duplicate chunks from same page

        Returns a dict:
            {
                "query":        cleaned query string,
                "results":      list of chunk dicts with score,
                "latency_ms":   retrieval time,
                "total_chunks": collection size,
            }
        """
        cleaned = preprocess_query(query)
        logger.info("Retrieving top-%d for: %s", top_k, cleaned)

        t0 = time.time()
        qv = embed_query(cleaned, model_name=self.embedding_model)
        results = query_collection(
            self.collection, qv, top_k=top_k, where=where
        )
        latency_ms = (time.time() - t0) * 1000

        if deduplicate:
            results = _deduplicate(results)

        # Add citation index
        for i, r in enumerate(results):
            r["citation_index"] = i + 1

        logger.info(
            "Retrieved %d chunks in %.0f ms (top score=%.3f)",
            len(results), latency_ms,
            results[0]["score"] if results else 0,
        )

        return {
            "query":        cleaned,
            "results":      results,
            "latency_ms":   round(latency_ms, 1),
            "total_chunks": self.collection.count(),
        }

    def retrieve_by_method(
        self,
        query: str,
        method: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Restrict retrieval to a specific SR method.
        Example: retrieve_by_method("loss function", "SRGAN")
        """
        return self.retrieve(query, top_k=top_k, where={"method": method})

    def retrieve_by_year_range(
        self,
        query: str,
        start_year: int,
        end_year: int,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Restrict retrieval to papers published in [start_year, end_year].
        """
        return self.retrieve(
            query,
            top_k=top_k,
            where={"year": {"$gte": start_year, "$lte": end_year}},
        )

    def collection_info(self) -> dict[str, Any]:
        """Return basic stats about the indexed collection."""
        col = self.collection
        return {
            "collection_name": col.name,
            "total_chunks":    col.count(),
            "persist_dir":     self.persist_dir,
            "embedding_model": self.embedding_model,
        }


# ---------------------------------------------------------------------------
# Functional interface (for simple scripts / notebooks)
# ---------------------------------------------------------------------------

_default_retriever: Retriever | None = None


def get_retriever(**kwargs) -> Retriever:
    """Return a module-level singleton retriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever(**kwargs)
    return _default_retriever


def retrieve(query: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
    """
    One-liner retrieval using the default singleton retriever.
    Returns just the results list.
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k, **kwargs)["results"]


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """
    Light query cleaning before embedding:
    - Strip whitespace
    - Collapse multiple spaces
    - Remove trailing punctuation that adds no semantic value
    - Preserve technical terms like PSNR, SSIM, SRGAN exactly
    """
    query = query.strip()
    query = re.sub(r"\s+", " ", query)      # collapse whitespace
    query = re.sub(r"[?!]+$", "", query)    # remove trailing ? or !
    return query


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(
    results: list[dict[str, Any]],
    max_per_page: int = 2,
) -> list[dict[str, Any]]:
    """
    Prevent the same page from dominating results.
    Keeps at most max_per_page chunks per (file_name, page_number) pair.
    Preserves original score ordering.
    """
    seen: dict[tuple, int] = {}
    deduped = []

    for r in results:
        key = (r.get("file_name", ""), r.get("page_number", 0))
        count = seen.get(key, 0)
        if count < max_per_page:
            deduped.append(r)
            seen[key] = count + 1

    return deduped