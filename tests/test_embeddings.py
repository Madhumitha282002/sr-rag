"""
tests/test_embeddings.py
Unit + integration tests for src/indexing/embeddings.py
and src/indexing/vector_store.py.
Run from project root: pytest tests/test_embeddings.py -v
"""

import pickle
import pytest
from pathlib import Path

from src.indexing.embeddings import (
    load_embedding_model,
    embed_texts,
    embed_chunks,
    embed_query,
    get_embedding_dim,
)
from src.indexing.vector_store import (
    load_vector_store,
    reset_vector_store,
    index_chunks,
    query_collection,
)

CHUNKS_PKL = Path("data/processed/chunks.pkl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return load_embedding_model()


@pytest.fixture(scope="module")
def sample_chunks():
    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)
    return chunks[:20]   # small slice for speed


@pytest.fixture(scope="module")
def embedded_chunks(sample_chunks):
    chunks = [dict(c) for c in sample_chunks]   # copy to avoid mutating fixture
    return embed_chunks(chunks, show_progress=False)


@pytest.fixture
def fresh_collection(tmp_path):
    """Empty ChromaDB collection in a temp directory."""
    return reset_vector_store(
        persist_dir=str(tmp_path / "test_vs"),
        collection_name="test_col",
    )


# ---------------------------------------------------------------------------
# load_embedding_model
# ---------------------------------------------------------------------------

class TestLoadEmbeddingModel:

    def test_returns_model(self, model):
        assert model is not None

    def test_cached_same_object(self):
        m1 = load_embedding_model()
        m2 = load_embedding_model()
        assert m1 is m2   # same object, not reloaded


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:

    def test_returns_list_of_vectors(self, model):
        vectors = embed_texts(["hello world", "super resolution"], show_progress=False)
        assert isinstance(vectors, list)
        assert len(vectors) == 2

    def test_vector_dimension_consistent(self):
        vectors = embed_texts(["a", "bb", "ccc"], show_progress=False)
        dims = [len(v) for v in vectors]
        assert len(set(dims)) == 1, f"Inconsistent dims: {dims}"

    def test_vectors_are_floats(self):
        vectors = embed_texts(["test"], show_progress=False)
        assert all(isinstance(x, float) for x in vectors[0])

    def test_different_texts_different_vectors(self):
        v1 = embed_texts(["perceptual loss in SRGAN"], show_progress=False)[0]
        v2 = embed_texts(["transformer attention mechanism"], show_progress=False)[0]
        assert v1 != v2


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:

    def test_returns_single_vector(self):
        v = embed_query("What loss does SRGAN use?")
        assert isinstance(v, list)
        assert len(v) > 0

    def test_same_dim_as_embed_texts(self):
        vq = embed_query("test query")
        vt = embed_texts(["test query"], show_progress=False)[0]
        assert len(vq) == len(vt)


# ---------------------------------------------------------------------------
# get_embedding_dim
# ---------------------------------------------------------------------------

class TestGetEmbeddingDim:

    def test_returns_positive_int(self):
        dim = get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_minilm_dim_is_384(self):
        dim = get_embedding_dim("sentence-transformers/all-MiniLM-L6-v2")
        assert dim == 384


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:

    def test_embedding_key_added(self, embedded_chunks):
        for chunk in embedded_chunks:
            assert "embedding" in chunk

    def test_embedding_length_matches_model_dim(self, embedded_chunks):
        dim = get_embedding_dim()
        for chunk in embedded_chunks:
            assert len(chunk["embedding"]) == dim

    def test_original_keys_preserved(self, embedded_chunks):
        for chunk in embedded_chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "method" in chunk


# ---------------------------------------------------------------------------
# index_chunks + query_collection
# ---------------------------------------------------------------------------

class TestVectorStore:

    def test_index_and_count(self, fresh_collection, embedded_chunks):
        index_chunks(fresh_collection, embedded_chunks)
        assert fresh_collection.count() == len(embedded_chunks)

    def test_query_returns_top_k(self, fresh_collection, embedded_chunks):
        index_chunks(fresh_collection, embedded_chunks)
        qv = embed_query("perceptual loss function")
        results = query_collection(fresh_collection, qv, top_k=3)
        assert len(results) == 3

    def test_result_keys(self, fresh_collection, embedded_chunks):
        index_chunks(fresh_collection, embedded_chunks)
        qv = embed_query("residual network architecture")
        results = query_collection(fresh_collection, qv, top_k=1)
        required = {"text", "score", "file_name", "method", "page_number", "citation_index"}
        assert required.issubset(results[0].keys())

    def test_scores_between_0_and_1(self, fresh_collection, embedded_chunks):
        index_chunks(fresh_collection, embedded_chunks)
        qv = embed_query("GAN discriminator training")
        results = query_collection(fresh_collection, qv, top_k=5)
        for r in results:
            assert 0 <= r["score"] <= 1, f"Score out of range: {r['score']}"

    def test_scores_descending(self, fresh_collection, embedded_chunks):
        index_chunks(fresh_collection, embedded_chunks)
        qv = embed_query("image quality PSNR metric")
        results = query_collection(fresh_collection, qv, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_embeddings_raises(self, fresh_collection, sample_chunks):
        raw = [dict(c) for c in sample_chunks]   # no embedding key
        with pytest.raises(ValueError, match="embedding"):
            index_chunks(fresh_collection, raw)