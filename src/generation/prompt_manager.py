"""
src/generation/prompt_manager.py
----------------------------------
Manages prompt templates, context building, and the
"refuse to answer" threshold.

Keeps all prompt logic out of answer_generator.py so
templates can be swapped and tested independently.

Key responsibilities:
  1. Build the user prompt from retrieved chunks with a
     character budget (context length limiter)
  2. Decide whether to answer or refuse based on top score
  3. Provide versioned prompt templates for A/B testing
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context limiter config
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 6000    # ~1500 tokens — safe for gpt-4o-mini
MIN_ANSWER_SCORE  = 0.30    # refuse if best chunk score is below this


# ---------------------------------------------------------------------------
# Prompt templates (versioned for A/B testing in Week 3)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {

    "v1": """You are an expert assistant on image super-resolution (SR) research.
You answer questions strictly based on the provided paper excerpts.

Rules:
- Ground every claim in the provided context. Cite sources as [1], [2], etc.
- If the context does not contain enough information, say so clearly.
- Do not hallucinate methods, numbers, or paper names not present in the context.
- Be concise and precise. Use technical language appropriate for ML researchers.
- When comparing methods, structure your answer clearly.""",

    "v2": """You are a knowledgeable research assistant specialising in image super-resolution.
Your answers are grounded exclusively in the paper excerpts provided below.

Guidelines:
- Every factual claim must cite its source using [N] notation.
- If multiple papers support a claim, cite all relevant ones: [1][3].
- If the provided context is insufficient to answer confidently, respond with:
  "The provided corpus does not contain enough information to answer this question."
- Never invent or infer details not explicitly stated in the excerpts.
- Structure comparative answers as clear bullet points or a short table.
- Keep answers under 300 words unless the question explicitly asks for detail.""",

    "v3_concise": """You are a precise SR research assistant. Answer only from the provided excerpts.
Cite every claim as [N]. If the answer is not in the context, say so. Be brief.""",
}

DEFAULT_TEMPLATE = "v2"


# ---------------------------------------------------------------------------
# Context builder with length limiter
# ---------------------------------------------------------------------------

def build_context(
    chunks: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build a context string from retrieved chunks within a character budget.

    Chunks are included in similarity-score order (highest first).
    Each chunk is truncated at the last sentence boundary if it would
    exceed the remaining budget. Truncation events are logged.

    Returns:
        (context_string, included_chunks)
    """
    included: list[dict[str, Any]] = []
    char_budget = max_chars
    truncated = False

    for chunk in chunks:
        text = chunk["text"].strip()

        if len(text) <= char_budget:
            included.append(chunk)
            char_budget -= len(text)
        else:
            # Truncate at last sentence boundary within budget
            fragment = text[:char_budget]
            last_period = fragment.rfind(". ")
            if last_period > char_budget // 2:
                fragment = fragment[:last_period + 1]
            else:
                fragment = fragment  # no good sentence boundary — keep raw

            if fragment.strip():
                included.append({**chunk, "text": fragment.strip()})
                truncated = True
                logger.info(
                    "Context truncated at chunk %s (%d -> %d chars)",
                    chunk.get("chunk_id", "?"), len(text), len(fragment),
                )
            break   # budget exhausted

    if truncated:
        logger.warning(
            "Context budget hit (%d chars). Included %d / %d chunks.",
            max_chars, len(included), len(chunks),
        )

    return _format_context(included), included


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] {chunk.get('method', '?')} ({chunk.get('year', '?')}) — "
            f"{chunk.get('file_name', '?')}, page {chunk.get('page_number', '?')}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Refusal check
# ---------------------------------------------------------------------------

def should_refuse(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_ANSWER_SCORE,
) -> tuple[bool, str]:
    """
    Decide whether to refuse answering based on retrieval quality.

    Returns (refuse: bool, reason: str).
    Refuse when:
      - No chunks were retrieved
      - The top chunk score is below min_score (weak match)
    """
    if not chunks:
        return True, "no_chunks"

    top_score = chunks[0].get("score", 0.0)
    if top_score < min_score:
        return True, f"low_score:{top_score:.3f}<{min_score}"

    return False, "ok"


REFUSAL_MESSAGE = (
    "The provided corpus does not contain enough information to answer "
    "this question confidently. Please rephrase your question or check "
    "whether the relevant paper is included in the corpus."
)


# ---------------------------------------------------------------------------
# Full prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    template: str = DEFAULT_TEMPLATE,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, str, list[dict[str, Any]], bool]:
    """
    Build system prompt + user prompt from question and retrieved chunks.

    Returns:
        (system_prompt, user_prompt, included_chunks, refused)

    If refused=True, user_prompt is the refusal message and
    included_chunks is empty — the caller should skip the LLM call.
    """
    # Refusal check
    refuse, reason = should_refuse(chunks)
    if refuse:
        logger.info("Refusing to answer: %s", reason)
        system = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS[DEFAULT_TEMPLATE])
        return system, REFUSAL_MESSAGE, [], True

    # Build context
    context, included = build_context(chunks, max_chars=max_chars)
    system = SYSTEM_PROMPTS.get(template, SYSTEM_PROMPTS[DEFAULT_TEMPLATE])

    user = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer (cite sources as [1], [2], etc.):"
    )

    return system, user, included, False


# ---------------------------------------------------------------------------
# Prompt stats helper (useful for notebook analysis)
# ---------------------------------------------------------------------------

def prompt_stats(
    question: str,
    chunks: list[dict[str, Any]],
    template: str = DEFAULT_TEMPLATE,
) -> dict[str, Any]:
    """Return diagnostic stats about a prompt without calling the LLM."""
    system, user, included, refused = build_prompt(question, chunks, template)
    return {
        "template":       template,
        "refused":        refused,
        "chunks_in":      len(chunks),
        "chunks_used":    len(included),
        "system_chars":   len(system),
        "user_chars":     len(user),
        "total_chars":    len(system) + len(user),
        "top_score":      chunks[0].get("score", 0.0) if chunks else 0.0,
        "truncated":      len(included) < len(chunks),
    }