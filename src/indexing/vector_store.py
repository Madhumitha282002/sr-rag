"""
src/indexing/vector_store.py
-----------------------------
Wraps ChromaDB for storing and querying chunk embeddings.

FIX: ChromaDB's PersistentClient must be kept alive at the module
level. If the client is local to load_vector_store() and goes out
of scope, Python's GC collects it, which silently breaks query()
(count() still works because it hits SQLite, but query() needs
the HNSW index which lives in the client).
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
# Module-level client cache — prevents garbage collection
# ---------------------------------------------------------------------------
_clients: dict[str, chromadb.PersistentClient] = {}


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    """Return a cached PersistentClient for the given path."""
    if persist_dir not in _clients:
        logger.info("Creating ChromaDB client at: %s", persist_dir)
        _clients[persist_dir] = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _clients[persist_dir]


# ---------------------------------------------------------------------------
# Client + collection loader
# ---------------------------------------------------------------------------

def load_vector_store(
    persist_dir: str | Path = DEFAULT_VECTOR_STORE,
    collection_name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """
    Open (or create) a persistent ChromaDB collection.
    The underlying client is kept alive in a module-level cache
    to prevent garbage collection from breaking HNSW queries.
    """
    persist_dir = str(Path(persist_dir).resolve())
    client = _get_client(persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
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
    """Delete and recreate the collection from scratch."""
    persist_dir = str(Path(persist_dir).resolve())

    # Remove cached client so we get a fresh one after reset
    _clients.pop(persist_dir, None)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # Store fresh client in cache
    _clients[persist_dir] = client

    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:
        pass

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
    """Upsert all chunks into ChromaDB in batches."""
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
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_build_metadata(c) for c in batch],
        )
        indexed += len(batch)
        logger.info("Indexed %d / %d chunks", indexed, total)

    logger.info("Indexing complete. Collection now has %d chunks.", collection.count())


def _build_metadata(chunk: dict[str, Any]) -> dict:
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
    """Retrieve the top_k most similar chunks for a query embedding."""
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
            "text":           doc,
            "score":          round(1 - dist, 4),
            "chunk_id":       meta.get("chunk_id", ""),
            "file_name":      meta.get("file_name", ""),
            "title":          meta.get("title", ""),
            "method":         meta.get("method", ""),
            "year":           meta.get("year", 0),
            "page_number":    meta.get("page_number", 0),
            "citation_index": len(results) + 1,
        })

    return results