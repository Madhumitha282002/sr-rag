"""
src/pipeline.py
----------------
End-to-end RAG pipeline: question -> retrieve -> generate -> cited answer.

This is the single entry point for the Streamlit app and FastAPI.
Neither should import from src.retrieval or src.generation directly.

Usage:
    pipeline = SRRagPipeline()
    result = pipeline.query("What loss does SRGAN use?")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.retrieval.retriever import Retriever
from src.generation.answer_generator import generate_answer
from src.generation.citations import validate_citations, format_answer_with_citations

logger = logging.getLogger(__name__)


class SRRagPipeline:
    """
    Orchestrates retrieval and generation into a single query() call.

    Instantiate once and reuse — the retriever holds the ChromaDB
    connection and the embedding model in memory.
    """

    def __init__(
        self,
        persist_dir: str = "vector_store",
        collection_name: str = "sr_papers",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        self.retriever = Retriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        self.llm_provider = llm_provider   # None -> reads from .env
        self.llm_model    = llm_model      # None -> reads from .env

        logger.info(
            "SRRagPipeline ready (collection=%s, provider=%s)",
            collection_name, llm_provider or "from .env",
        )

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        include_raw_chunks: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline for a question.

        Args:
            question          : natural language question
            top_k             : chunks to retrieve
            where             : optional ChromaDB metadata filter
            include_raw_chunks: include raw retrieved chunks in output

        Returns:
            {
                "question":     original question,
                "answer":       LLM-generated answer with [N] citations,
                "answer_full":  answer + formatted references block,
                "sources":      list of source chunk dicts,
                "citations_valid": bool,
                "token_usage":  dict with tokens + cost,
                "retrieval_ms": retrieval latency,
                "generation_ms": generation latency,
                "total_ms":     end-to-end latency,
                "provider":     which LLM was used,
                "model":        which model was used,
                "raw_chunks":   (optional) raw retrieval results,
            }
        """
        t_start = time.time()

        # Step 1 — Retrieve
        retrieval = self.retriever.retrieve(
            question, top_k=top_k, where=where
        )
        retrieval_ms = retrieval["latency_ms"]
        chunks = retrieval["results"]

        if not chunks:
            logger.warning("No chunks retrieved for: %s", question)
            return self._empty_result(question)

        # Step 2 — Generate
        t_gen = time.time()
        gen = generate_answer(
            question=question,
            retrieved_chunks=chunks,
            provider=self.llm_provider,
            model=self.llm_model,
        )
        generation_ms = (time.time() - t_gen) * 1000
        total_ms = (time.time() - t_start) * 1000

        # Step 3 — Validate citations
        citation_report = validate_citations(gen["answer"], gen["sources"])

        # Step 4 — Format full answer with references
        answer_full = format_answer_with_citations(gen["answer"], gen["sources"])

        result = {
            "question":        question,
            "answer":          gen["answer"],
            "answer_full":     answer_full,
            "sources":         gen["sources"],
            "citations_valid": citation_report["valid"],
            "token_usage":     gen["token_usage"],
            "retrieval_ms":    round(retrieval_ms, 1),
            "generation_ms":   round(generation_ms, 1),
            "total_ms":        round(total_ms, 1),
            "provider":        gen["provider"],
            "model":           gen["model"],
        }

        if include_raw_chunks:
            result["raw_chunks"] = chunks

        logger.info(
            "Query complete in %.0f ms (ret=%.0f, gen=%.0f) | citations_valid=%s",
            total_ms, retrieval_ms, generation_ms, citation_report["valid"],
        )
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def query_with_filter(
        self,
        question: str,
        method: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Query with optional method and/or year filters.

        Examples:
            pipeline.query_with_filter("loss function", method="SRGAN")
            pipeline.query_with_filter("attention", year_from=2021)
        """
        where: dict | None = None

        if method and year_from:
            where = {"$and": [{"method": method}, {"year": {"$gte": year_from}}]}
        elif method:
            where = {"method": method}
        elif year_from and year_to:
            where = {"year": {"$gte": year_from, "$lte": year_to}}
        elif year_from:
            where = {"year": {"$gte": year_from}}

        return self.query(question, top_k=top_k, where=where)

    def log_feedback(self, question: str, answer: str, helpful: bool) -> None:
        """Proxy to answer_generator's feedback logger."""
        from src.generation.answer_generator import log_feedback
        log_feedback(question=question, answer=answer, helpful=helpful)

    def info(self) -> dict[str, Any]:
        """Return pipeline configuration and collection stats."""
        col_info = self.retriever.collection_info()
        return {
            "collection_name":  col_info["collection_name"],
            "total_chunks":     col_info["total_chunks"],
            "embedding_model":  col_info["embedding_model"],
            "persist_dir":      col_info["persist_dir"],
            "llm_provider":     self.llm_provider or "from .env",
            "llm_model":        self.llm_model or "from .env",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_result(self, question: str) -> dict[str, Any]:
        return {
            "question":        question,
            "answer":          "No relevant chunks found in the corpus for this question.",
            "answer_full":     "No relevant chunks found in the corpus for this question.",
            "sources":         [],
            "citations_valid": True,
            "token_usage":     {"prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0, "estimated_cost_usd": 0.0},
            "retrieval_ms":    0.0,
            "generation_ms":   0.0,
            "total_ms":        0.0,
            "provider":        "none",
            "model":           "none",
        }