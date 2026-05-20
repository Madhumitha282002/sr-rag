"""
tests/test_chunker.py
Unit tests for src/ingestion/chunker.py
Run from project root: pytest tests/test_chunker.py -v
"""

import pickle
import pytest
from pathlib import Path
from src.ingestion.chunker import (
    chunk_pages,
    chunk_corpus,
    chunk_stats,
    save_chunks,
    _make_chunk_id,
)

EXTRACTED_PAGES_PKL = Path("data/processed/extracted_pages.pkl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_pages():
    """Load real extracted pages for integration-style tests."""
    with open(EXTRACTED_PAGES_PKL, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def tiny_pages():
    """Minimal synthetic pages for fast unit tests."""
    return [
        {
            "file_name":   "fake_2020.pdf",
            "page_number": 1,
            "page_count":  5,
            "title":       "Fake Paper",
            "method":      "FakeNet",
            "authors":     "Nobody",
            "year":        2020,
            "venue":       "FakeConf",
            "text":        "Super resolution improves image quality. " * 40,
            "char_count":  1600,
            "word_count":  240,
        },
        {
            "file_name":   "fake_2020.pdf",
            "page_number": 2,
            "page_count":  5,
            "title":       "Fake Paper",
            "method":      "FakeNet",
            "authors":     "Nobody",
            "year":        2020,
            "venue":       "FakeConf",
            "text":        "The loss function uses perceptual features. " * 40,
            "char_count":  1720,
            "word_count":  280,
        },
    ]


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------

class TestChunkPages:

    def test_returns_list(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_required_keys_present(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        required = {
            "chunk_id", "chunk_index", "file_name", "page_number",
            "title", "method", "year", "text", "char_count", "word_count",
        }
        for chunk in chunks:
            assert required.issubset(chunk.keys())

    def test_metadata_inherited(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        for chunk in chunks:
            assert chunk["method"] == "FakeNet"
            assert chunk["year"] == 2020

    def test_chunk_ids_unique(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_no_empty_chunk_text(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        for chunk in chunks:
            assert chunk["text"].strip() != ""
            assert chunk["char_count"] > 0
            assert chunk["word_count"] > 0

    def test_char_count_matches_text(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["text"])

    def test_word_count_matches_text(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        for chunk in chunks:
            assert chunk["word_count"] == len(chunk["text"].split())

    def test_respects_chunk_size(self, tiny_pages):
        chunks = chunk_pages(tiny_pages, chunk_size=300, chunk_overlap=50)
        oversized = [c for c in chunks if c["char_count"] > 400]
        assert len(oversized) == 0, \
            f"{len(oversized)} chunks exceed size limit"

    def test_empty_pages_list(self):
        assert chunk_pages([]) == []

    def test_page_with_empty_text_skipped(self):
        pages = [{"file_name": "x.pdf", "page_number": 1, "page_count": 1,
                  "title": "", "method": "", "authors": "", "year": 0,
                  "venue": "", "text": "", "char_count": 0, "word_count": 0}]
        assert chunk_pages(pages) == []


# ---------------------------------------------------------------------------
# chunk_corpus (integration)
# ---------------------------------------------------------------------------

class TestChunkCorpus:

    def test_loads_and_chunks_pkl(self):
        chunks = chunk_corpus(EXTRACTED_PAGES_PKL)
        assert len(chunks) > 200, f"Expected >200 chunks, got {len(chunks)}"

    def test_all_10_papers_present(self):
        chunks = chunk_corpus(EXTRACTED_PAGES_PKL)
        methods = {c["method"] for c in chunks}
        assert len(methods) == 10, f"Expected 10 methods, got {len(methods)}: {methods}"

    def test_missing_pkl_raises(self):
        with pytest.raises(FileNotFoundError):
            chunk_corpus("data/processed/does_not_exist.pkl")


# ---------------------------------------------------------------------------
# chunk_stats
# ---------------------------------------------------------------------------

class TestChunkStats:

    def test_returns_expected_keys(self, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        stats = chunk_stats(chunks)
        for key in ["total_chunks", "avg_words", "min_words", "max_words"]:
            assert key in stats

    def test_empty_list_returns_empty_dict(self):
        assert chunk_stats([]) == {}


# ---------------------------------------------------------------------------
# save_chunks
# ---------------------------------------------------------------------------

class TestSaveChunks:

    def test_saves_json_and_pkl(self, tmp_path, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        out = tmp_path / "chunks"
        save_chunks(chunks, out)
        assert (tmp_path / "chunks.json").exists()
        assert (tmp_path / "chunks.pkl").exists()

    def test_saved_pkl_reloads_correctly(self, tmp_path, tiny_pages):
        chunks = chunk_pages(tiny_pages)
        out = tmp_path / "chunks"
        save_chunks(chunks, out)
        with open(tmp_path / "chunks.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == len(chunks)
        assert loaded[0]["chunk_id"] == chunks[0]["chunk_id"]


# ---------------------------------------------------------------------------
# _make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:

    def test_format(self):
        cid = _make_chunk_id("srgan_2016.pdf", 3, 1)
        assert cid == "srgan_2016_p03_c01"

    def test_uniqueness_across_pages(self):
        id1 = _make_chunk_id("srgan_2016.pdf", 1, 0)
        id2 = _make_chunk_id("srgan_2016.pdf", 2, 0)
        assert id1 != id2

    def test_uniqueness_across_chunks(self):
        id1 = _make_chunk_id("srgan_2016.pdf", 1, 0)
        id2 = _make_chunk_id("srgan_2016.pdf", 1, 1)
        assert id1 != id2