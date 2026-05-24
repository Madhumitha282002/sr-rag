"""
src/generation/citations.py
-----------------------------
Formats source chunks into clean citation strings and
validates that every [N] reference in an answer has a matching source.
"""

from __future__ import annotations

import re
from typing import Any


def format_citations(sources: list[dict[str, Any]]) -> list[str]:
    """
    Format a list of source chunks into numbered citation strings.

    Example output:
        ["[1] SRGAN (Ledig et al., 2016) — srgan_2016.pdf, page 4",
         "[2] SwinIR (Liang et al., 2021) — swinir_2021.pdf, page 7"]
    """
    citations = []
    for src in sources:
        idx     = src.get("citation_index", "?")
        method  = src.get("method", "Unknown")
        authors = src.get("authors", "")
        year    = src.get("year", "")
        fname   = src.get("file_name", "")
        page    = src.get("page_number", "")

        # Short author format: "Ledig et al." -> keep as-is; "Ledig, C." -> "Ledig et al."
        short_authors = _shorten_authors(authors)

        citations.append(
            f"[{idx}] {method} ({short_authors}, {year}) — {fname}, page {page}"
        )
    return citations


def extract_cited_indices(answer: str) -> list[int]:
    """
    Parse all [N] references from an answer string.
    Returns a sorted list of unique citation indices found.

    Example: "method [1] is better [2] and also [1]" -> [1, 2]
    """
    matches = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(int(m) for m in matches))


def validate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Check that every [N] in the answer maps to an actual source.
    Returns a validation report dict.

    Report keys:
        valid         : bool — True if all cited indices have a source
        cited_indices : list of ints mentioned in answer
        missing       : list of indices with no matching source
        unused        : list of source indices not cited in answer
    """
    cited   = extract_cited_indices(answer)
    present = {src.get("citation_index") for src in sources}
    missing = [i for i in cited if i not in present]
    unused  = [i for i in present if i not in cited]

    return {
        "valid":         len(missing) == 0,
        "cited_indices": cited,
        "missing":       missing,
        "unused_sources": unused,
    }


def format_answer_with_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> str:
    """
    Append a formatted references section to the answer text.
    Returns the full display string for the Streamlit UI or API.
    """
    if not sources:
        return answer

    citations = format_citations(sources)
    refs = "\n".join(citations)
    return f"{answer}\n\n**References**\n{refs}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shorten_authors(authors: str) -> str:
    """
    Convert 'Ledig, C. and Theis, L. and ...' or 'Ledig et al.'
    to 'Ledig et al.' for display.
    """
    if not authors:
        return ""
    if "et al" in authors:
        return authors.strip()
    parts = re.split(r"\band\b|,|\.", authors)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return authors
    last_name = parts[0].split()[-1]
    if len(parts) > 1:
        return f"{last_name} et al."
    return last_name