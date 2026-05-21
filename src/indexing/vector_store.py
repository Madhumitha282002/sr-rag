"""
src/indexing/vector_store.py
-----------------------------
Wraps ChromaDB for storing and querying chunk embeddings.

Collection schema per chunk:
  - id        : chunk_id (e.g. "srgan_2016_p03_c01")
  - embedding : float vector from sentence-transformer
  - document  : chunk text (Chroma's 'documents' field)
  - metadata  : file_name, title, method, year, page_number, etc.

IMPORTANT: if you change chunk_size or the embedding model,
delete vector_store/ and re-index from scratch. Mixed collections
silently return wrong results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION   = "sr_papers"
DEFAULT_VECTOR_STORE = "vector_store"


# ---------------------------------------------------------------------------
# Client + collection loader
# ---------------------------------------------------------------------------

def load_vector_store(
    persist_dir: str | Path = DEFAULT_VECTOR_STORE,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """
    Open (or create) a persistent ChromaDB collection.
    Safe to call multiple times — returns existing collection if present.
    """
    persist_dir = str(Path(persist_dir).resolve())
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for sentence-transformers
    )
    count = collection.count()
    logger.info(
        "Opened collection '%s' at %s (%d chunks indexed)",
        collection_name, persist_dir, count,
    )
    return collection


def reset_vector_store(
    persist_dir: str | Path = DEFAULT_VECTOR_STORE,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """
    Delete and recreate the collection from scratch.
    Call this whenever chunk_size or embedding model changes.
    """
    persist_dir = str(Path(persist_dir).resolve())
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        pass   # collection didn't exist — fine

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created fresh collection '%s'", collection_name)
    return collection


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunks(
    collection: chromadb.Collection,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """
    Upsert all chunks (with pre-computed embeddings) into ChromaDB.
    Chunks must already have an 'embedding' key — run embed_chunks() first.
    Uses batching to stay within ChromaDB's memory limits.
    """
    if not chunks:
        logger.warning("index_chunks called with empty list — nothing to do")
        return

    if "embedding" not in chunks[0]:
        raise ValueError(
            "Chunks must have 'embedding' key. "
            "Run embed_chunks() before index_chunks()."
        )

    total = len(chunks)
    indexed = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        ids        = [c["chunk_id"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        documents  = [c["text"] for c in batch]
        metadatas  = [_build_metadata(c) for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        indexed += len(batch)
        logger.info("Indexed %d / %d chunks", indexed, total)

    logger.info("Indexing complete. Collection now has %d chunks.", collection.count())


def _build_metadata(chunk: dict[str, Any]) -> dict:
    """
    Extract only ChromaDB-safe metadata fields (str, int, float, bool).
    Lists and nested dicts are not supported by Chroma.
    """
    return {
        "file_name":   str(chunk.get("file_name", "")),
        "title":       str(chunk.get("title", "")),
        "method":      str(chunk.get("method", "")),
        "authors":     str(chunk.get("authors", "")),
        "year":        int(chunk.get("year", 0)),
        "venue":       str(chunk.get("venue", "")),
        "page_number": int(chunk.get("page_number", 0)),
        "page_count":  int(chunk.get("page_count", 0)),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "word_count":  int(chunk.get("word_count", 0)),
        "char_count":  int(chunk.get("char_count", 0)),
    }


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top_k most similar chunks for a query embedding.
    Returns a list of result dicts with text, metadata, and score.

    Optional 'where' filter follows ChromaDB syntax, e.g.:
        where={"method": "SRGAN"}
        where={"year": {"$gte": 2020}}
    """
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "text":         doc,
            "score":        round(1 - dist, 4),   # cosine distance -> similarity
            "chunk_id":     meta.get("chunk_id", ""),
            "file_name":    meta.get("file_name", ""),
            "title":        meta.get("title", ""),
            "method":       meta.get("method", ""),
            "year":         meta.get("year", 0),
            "page_number":  meta.get("page_number", 0),
            "citation_index": len(results) + 1,
        })

    return results