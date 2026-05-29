"""
src/pipeline.py  (Day 13 update)
----------------------------------
Added optional reranking step between retrieval and generation.

Changes from Day 8:
  - Accepts use_reranker=True in query()
  - When enabled, fetches top_k * 3 candidates, reranks, returns top_k
  - Reranker latency tracked separately in result dict
  - Reranker instance lazy-loaded on first use
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
    End-to-end RAG pipeline: retrieve → (rerank) → generate → cite.
    Instantiate once and reuse.
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
        self.llm_provider = llm_provider
        self.llm_model    = llm_model
        self._reranker    = None   # lazy-loaded on first use

        logger.info(
            "SRRagPipeline ready (collection=%s, provider=%s)",
            collection_name, llm_provider or "from .env",
        )

    # ------------------------------------------------------------------
    # Reranker (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def reranker(self):
        if self._reranker is None:
            from src.retrieval.reranker import Reranker
            self._reranker = Reranker()
            logger.info("Reranker loaded.")
        return self._reranker

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranker: bool = False,
        where: dict | None = None,
        include_raw_chunks: bool = False,
        prompt_template: str = "v2",
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline.

        Args:
            question          : natural language question
            top_k             : final number of chunks to pass to LLM
            use_reranker      : if True, fetch top_k*3 candidates and rerank
            where             : optional ChromaDB metadata filter
            include_raw_chunks: include raw retrieved chunks in result
            prompt_template   : 'v1' | 'v2' | 'v3_concise'

        Returns dict with answer, sources, latencies, token_usage, etc.
        """
        t_start = time.time()

        # Step 1 — Retrieve
        # Fetch more candidates when reranking so the reranker has
        # enough material to work with
        fetch_k = top_k * 3 if use_reranker else top_k
        retrieval = self.retriever.retrieve(question, top_k=fetch_k, where=where)
        retrieval_ms = retrieval["latency_ms"]
        chunks = retrieval["results"]

        if not chunks:
            logger.warning("No chunks retrieved for: %s", question)
            return self._empty_result(question)

        # Step 2 — Rerank (optional)
        rerank_ms = 0.0
        if use_reranker:
            t_rerank = time.time()
            chunks = self.reranker.rerank(question, chunks, top_k=top_k)
            rerank_ms = (time.time() - t_rerank) * 1000
            logger.info("Reranking complete in %.0f ms", rerank_ms)

        # Step 3 — Generate
        t_gen = time.time()
        gen = generate_answer(
            question=question,
            retrieved_chunks=chunks,
            provider=self.llm_provider,
            model=self.llm_model,
            prompt_template=prompt_template,
        )
        generation_ms = (time.time() - t_gen) * 1000
        total_ms = (time.time() - t_start) * 1000

        # Step 4 — Validate + format
        citation_report = validate_citations(gen["answer"], gen["sources"])
        answer_full = format_answer_with_citations(gen["answer"], gen["sources"])

        result = {
            "question":        question,
            "answer":          gen["answer"],
            "answer_full":     answer_full,
            "sources":         gen["sources"],
            "citations_valid": citation_report["valid"],
            "token_usage":     gen["token_usage"],
            "retrieval_ms":    round(retrieval_ms, 1),
            "rerank_ms":       round(rerank_ms, 1),
            "generation_ms":   round(generation_ms, 1),
            "total_ms":        round(total_ms, 1),
            "provider":        gen["provider"],
            "model":           gen["model"],
            "refused":         gen.get("refused", False),
            "use_reranker":    use_reranker,
        }

        if include_raw_chunks:
            result["raw_chunks"] = chunks

        logger.info(
            "Query done in %.0f ms (ret=%.0f, rerank=%.0f, gen=%.0f) | reranker=%s",
            total_ms, retrieval_ms, rerank_ms, generation_ms, use_reranker,
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
        use_reranker: bool = False,
    ) -> dict[str, Any]:
        where: dict | None = None
        if method and year_from:
            where = {"$and": [{"method": method}, {"year": {"$gte": year_from}}]}
        elif method:
            where = {"method": method}
        elif year_from and year_to:
            where = {"year": {"$gte": year_from, "$lte": year_to}}
        elif year_from:
            where = {"year": {"$gte": year_from}}
        return self.query(question, top_k=top_k, where=where, use_reranker=use_reranker)

    def log_feedback(self, question: str, answer: str, helpful: bool) -> None:
        from src.generation.answer_generator import log_feedback
        log_feedback(question=question, answer=answer, helpful=helpful)

    def info(self) -> dict[str, Any]:
        col_info = self.retriever.collection_info()
        return {
            "collection_name":  col_info["collection_name"],
            "total_chunks":     col_info["total_chunks"],
            "embedding_model":  col_info["embedding_model"],
            "persist_dir":      col_info["persist_dir"],
            "llm_provider":     self.llm_provider or "from .env",
            "llm_model":        self.llm_model or "from .env",
            "reranker_loaded":  self._reranker is not None,
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
            "rerank_ms":       0.0,
            "generation_ms":   0.0,
            "total_ms":        0.0,
            "provider":        "none",
            "model":           "none",
            "refused":         False,
            "use_reranker":    False,
        }