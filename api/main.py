"""
api/main.py - with direct ChromaDB test endpoint
"""

from __future__ import annotations

import csv
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.indexing.vector_store import load_vector_store, query_collection
from src.monitoring.logger import setup_logging
from src.pipeline import SRRagPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
pipeline: SRRagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    setup_logging(log_dir=str(PROJECT_ROOT / "logs"), console=True)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Loading SR-RAG pipeline...")
    pipeline = SRRagPipeline(persist_dir=str(PROJECT_ROOT / "vector_store"))
    count = pipeline.info()["total_chunks"]
    logger.info("Pipeline ready. Collection: %d chunks", count)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="SR-RAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, example="What loss function does SRGAN use?"
    )
    top_k: int = Field(5, ge=1, le=15)
    use_reranker: bool = Field(False)
    prompt_template: str = Field("v2")
    method_filter: str | None = Field(None, example=None)
    year_from: int | None = Field(None, example=None)
    year_to: int | None = Field(None, example=None)


class SourceModel(BaseModel):
    citation_index: int
    method: str
    year: int
    file_name: str
    page_number: int
    score: float
    text: str


class TokenUsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_full: str
    sources: list[SourceModel]
    citations_valid: bool
    refused: bool
    token_usage: TokenUsageModel
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    provider: str
    model: str
    use_reranker: bool


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool


class FeedbackResponse(BaseModel):
    status: str
    message: str


@app.get("/debug", tags=["System"])
async def debug() -> dict:
    vs_path = str(PROJECT_ROOT / "vector_store")
    col = load_vector_store(persist_dir=vs_path, collection_name="sr_papers")
    return {
        "project_root": str(PROJECT_ROOT),
        "vector_store": vs_path,
        "chunk_count": col.count(),
        "pipeline_none": pipeline is None,
        "retriever_path": str(pipeline.retriever.persist_dir) if pipeline else "N/A",
        "pipeline_chunks": pipeline.info()["total_chunks"] if pipeline else 0,
    }


@app.get("/test-query", tags=["System"])
async def test_query() -> dict:
    """Bypass pipeline entirely — query ChromaDB directly."""
    from src.indexing.embeddings import embed_query

    vs_path = str(PROJECT_ROOT / "vector_store")
    col = load_vector_store(persist_dir=vs_path, collection_name="sr_papers")
    count = col.count()

    qv = embed_query("What loss does SRGAN use?")
    results = query_collection(col, qv, top_k=5)

    return {
        "collection_count": count,
        "results_returned": len(results),
        "top_score": results[0]["score"] if results else 0,
        "top_method": results[0]["method"] if results else "none",
        "top_text": results[0]["text"][:100] if results else "none",
    }


@app.post("/query-direct", tags=["System"])
async def query_direct(request: QueryRequest) -> dict:
    """Bypass pipeline class — do everything inline like test-query."""
    from src.generation.answer_generator import generate_answer
    from src.generation.citations import format_answer_with_citations
    from src.indexing.embeddings import embed_query

    vs_path = str(PROJECT_ROOT / "vector_store")
    col = load_vector_store(persist_dir=vs_path, collection_name="sr_papers")
    qv = embed_query(request.question)
    chunks = query_collection(col, qv, top_k=request.top_k)

    gen = generate_answer(request.question, chunks, provider="mock")

    return {
        "chunks_found": len(chunks),
        "answer": gen["answer"],
        "sources": len(gen["sources"]),
        "top_method": chunks[0]["method"] if chunks else "none",
    }


@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": pipeline is not None,
    }


@app.get("/info", tags=["System"])
async def info() -> dict[str, Any]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return pipeline.info()


@app.get("/corpus", tags=["Corpus"])
async def corpus(sort_by: str = Query("year")) -> dict[str, Any]:
    csv_path = PROJECT_ROOT / "data" / "processed" / "paper_metadata.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="paper_metadata.csv not found")
    papers = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            papers.append(
                {
                    "method": row.get("method", ""),
                    "title": row.get("title", ""),
                    "authors": row.get("authors", ""),
                    "year": int(row.get("year", 0)),
                    "venue": row.get("venue", ""),
                    "datasets": row.get("datasets", ""),
                    "key_contribution": row.get("key_contribution", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
    papers.sort(key=lambda p: p.get(sort_by, ""), reverse=(sort_by == "year"))
    return {"total": len(papers), "papers": papers}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest) -> QueryResponse:
    import time

    from src.generation.answer_generator import generate_answer
    from src.generation.citations import format_answer_with_citations, validate_citations
    from src.indexing.embeddings import embed_query

    vs_path = str(PROJECT_ROOT / "vector_store")
    t_start = time.time()

    # Build metadata filter
    where: dict | None = None
    if request.method_filter and request.year_from:
        where = {"$and": [{"method": request.method_filter}, {"year": {"$gte": request.year_from}}]}
    elif request.method_filter:
        where = {"method": request.method_filter}
    elif request.year_from and request.year_to:
        where = {"year": {"$gte": request.year_from, "$lte": request.year_to}}
    elif request.year_from:
        where = {"year": {"$gte": request.year_from}}

    logger.info("Query: '%s' | top_k=%d | where=%s", request.question, request.top_k, where)

    try:
        t_ret = time.time()
        qv = embed_query(request.question)
        col = load_vector_store(persist_dir=vs_path, collection_name="sr_papers")
        chunks = query_collection(col, qv, top_k=request.top_k, where=where)
        retrieval_ms = (time.time() - t_ret) * 1000

        logger.info("Retrieved %d chunks in %.0f ms", len(chunks), retrieval_ms)

        if not chunks:
            return QueryResponse(
                question=request.question,
                answer="No relevant chunks found in the corpus for this question.",
                answer_full="No relevant chunks found in the corpus for this question.",
                sources=[],
                citations_valid=True,
                refused=False,
                token_usage=TokenUsageModel(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                ),
                retrieval_ms=round(retrieval_ms, 1),
                rerank_ms=0.0,
                generation_ms=0.0,
                total_ms=round((time.time() - t_start) * 1000, 1),
                provider="none",
                model="none",
                use_reranker=False,
            )

        t_gen = time.time()
        gen = generate_answer(
            request.question,
            chunks,
            provider=None,
            prompt_template=request.prompt_template,
        )
        generation_ms = (time.time() - t_gen) * 1000
        total_ms = (time.time() - t_start) * 1000

        citation_report = validate_citations(gen["answer"], gen["sources"])
        answer_full = format_answer_with_citations(gen["answer"], gen["sources"])

    except Exception as exc:
        logger.error("Query error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceModel(
            citation_index=src.get("citation_index", 0),
            method=src.get("method", ""),
            year=src.get("year", 0),
            file_name=src.get("file_name", ""),
            page_number=src.get("page_number", 0),
            score=src.get("score", 0.0),
            text=src.get("text", "")[:500],
        )
        for src in gen.get("sources", [])
    ]

    token_usage = gen.get("token_usage", {})
    return QueryResponse(
        question=request.question,
        answer=gen["answer"],
        answer_full=answer_full,
        sources=sources,
        citations_valid=citation_report["valid"],
        refused=gen.get("refused", False),
        token_usage=TokenUsageModel(
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
            estimated_cost_usd=token_usage.get("estimated_cost_usd", 0.0),
        ),
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=0.0,
        generation_ms=round(generation_ms, 1),
        total_ms=round(total_ms, 1),
        provider=gen.get("provider", ""),
        model=gen.get("model", ""),
        use_reranker=False,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Query"])
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    pipeline.log_feedback(request.question, request.answer, request.helpful)
    return FeedbackResponse(status="ok", message="Feedback recorded. Thank you!")
