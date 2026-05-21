"""
src/indexing/embeddings.py
---------------------------
Loads a sentence-transformer model and embeds chunks.

Keeps the model as a module-level singleton so it is only
loaded once per process — critical for Streamlit performance
(use @st.cache_resource on top of get_model() in the app).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config — change here or override via configs/settings.py
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternatives to try in Week 3 evaluation:
#   "BAAI/bge-small-en-v1.5"   — often beats MiniLM on technical text
#   "all-mpnet-base-v2"        — slower but higher quality

_model_cache: dict[str, SentenceTransformer] = {}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load a sentence-transformer model, caching it in memory.
    Second call with the same model_name returns instantly.
    """
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        t0 = time.time()
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Model loaded in %.1f s", time.time() - t0)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of strings. Returns a list of float vectors.
    Uses batching to avoid OOM on large corpora.
    """
    model = load_embedding_model(model_name)
    t0 = time.time()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    logger.info(
        "Embedded %d texts in %.1f s (%.0f texts/s)",
        len(texts), elapsed, len(texts) / max(elapsed, 0.001),
    )
    return vectors.tolist()


def embed_chunks(
    chunks: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Add an 'embedding' key to each chunk dict in-place.
    Returns the same list with embeddings attached.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, model_name=model_name,
                          batch_size=batch_size, show_progress=show_progress)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    logger.info("Attached embeddings to %d chunks (dim=%d)", len(chunks), len(vectors[0]))
    return chunks


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> list[float]:
    """
    Embed a single query string for retrieval.
    Returns a float vector.
    """
    model = load_embedding_model(model_name)
    vector = model.encode([query], convert_to_numpy=True)
    return vector[0].tolist()


def get_embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the output dimension of the model (e.g. 384 for MiniLM)."""
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()