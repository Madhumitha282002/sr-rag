"""
src/ingestion/chunker.py
-------------------------
Splits page-level text into overlapping chunks using
RecursiveCharacterTextSplitter. Attaches all metadata so every
chunk is self-contained and traceable back to its source page.

Output format per chunk:
{
    "chunk_id":     "srgan_2016_p03_c01",
    "chunk_index":  0,
    "file_name":    "srgan_2016.pdf",
    "title":        "...",
    "method":       "SRGAN / SRResNet",
    "year":         2016,
    "page_number":  3,
    "page_count":   16,
    "text":         "... chunk text ...",
    "char_count":   742,
    "word_count":   124,
}

IMPORTANT: if you change CHUNK_SIZE or CHUNK_OVERLAP after indexing,
delete vector_store/ and re-run the full ingestion pipeline before
Day 5. Stale embeddings from old chunk sizes will silently corrupt
retrieval results.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via configs/settings.py
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE    = 900
DEFAULT_CHUNK_OVERLAP = 150

# Separators tried in order: paragraph -> sentence -> word -> character
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split a list of page dicts into overlapping chunks.
    Each chunk inherits all metadata from its source page.
    Returns a flat list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )

    all_chunks: list[dict[str, Any]] = []

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        splits = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = _make_chunk_id(
                page["file_name"], page["page_number"], chunk_index
            )

            chunk = {
                "chunk_id":    chunk_id,
                "chunk_index": chunk_index,
                # Source metadata
                "file_name":   page["file_name"],
                "page_number": page["page_number"],
                "page_count":  page.get("page_count", 0),
                # Paper metadata
                "title":       page.get("title", ""),
                "method":      page.get("method", ""),
                "authors":     page.get("authors", ""),
                "year":        page.get("year", 0),
                "venue":       page.get("venue", ""),
                # Chunk content
                "text":        chunk_text,
                "char_count":  len(chunk_text),
                "word_count":  len(chunk_text.split()),
            }
            all_chunks.append(chunk)

    logger.info(
        "Chunked %d pages -> %d chunks (size=%d, overlap=%d)",
        len(pages), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


def chunk_corpus(
    extracted_pages_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Load extracted_pages.pkl produced by Day 3 and chunk the full corpus.
    Convenience wrapper around chunk_pages().
    """
    path = Path(extracted_pages_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Extracted pages not found: {path}\n"
            "Run notebooks/01_pdf_extraction.ipynb first."
        )

    with open(path, "rb") as f:
        pages = pickle.load(f)

    logger.info("Loaded %d pages from %s", len(pages), path)
    return chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def save_chunks(chunks: list[dict[str, Any]], out_path: str | Path) -> None:
    """Save chunks to both JSON (human-readable) and pickle (fast loading)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    pkl_path = out_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Saved %d chunks -> %s and %s", len(chunks), json_path, pkl_path)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a summary stats dict for a list of chunks."""
    if not chunks:
        return {}

    word_counts = [c["word_count"] for c in chunks]
    char_counts = [c["char_count"] for c in chunks]
    methods     = list({c["method"] for c in chunks})

    return {
        "total_chunks": len(chunks),
        "papers":       len(methods),
        "methods":      sorted(methods),
        "avg_words":    round(sum(word_counts) / len(word_counts), 1),
        "min_words":    min(word_counts),
        "max_words":    max(word_counts),
        "avg_chars":    round(sum(char_counts) / len(char_counts), 1),
        "min_chars":    min(char_counts),
        "max_chars":    max(char_counts),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(file_name: str, page_number: int, chunk_index: int) -> str:
    """
    Deterministic chunk ID: e.g. srgan_2016_p03_c01
    Falls back to a short hash for unexpectedly long filenames.
    """
    stem = Path(file_name).stem
    base = f"{stem}_p{page_number:02d}_c{chunk_index:02d}"
    if len(base) <= 40:
        return base
    h = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"chunk_{h}"