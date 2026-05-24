"""
src/generation/answer_generator.py
------------------------------------
Builds a prompt from retrieved chunks and calls an LLM to generate
a grounded answer. Supports OpenAI, Gemini, Ollama, and a mock mode
for development without an API key.

Set LLM_PROVIDER in .env to switch providers:
    LLM_PROVIDER=openai     + OPENAI_API_KEY=sk-...
    LLM_PROVIDER=gemini     + GEMINI_API_KEY=...
    LLM_PROVIDER=ollama     (no key needed, runs locally)
    LLM_PROVIDER=mock       (default when no key is set)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert assistant on image super-resolution (SR) research.
You answer questions strictly based on the provided paper excerpts.

Rules:
- Ground every claim in the provided context. Cite sources as [1], [2], etc.
- If the context does not contain enough information, say so clearly.
- Do not hallucinate methods, numbers, or paper names not present in the context.
- Be concise and precise. Use technical language appropriate for ML researchers.
- When comparing methods, structure your answer clearly.
"""

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 6000   # ~1500 tokens — safe for gpt-4o-mini context window


def build_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build the user prompt from retrieved chunks.
    Chunks are included in similarity order until MAX_CONTEXT_CHARS is reached.
    Returns (prompt_text, included_chunks).
    """
    included, char_budget = [], MAX_CONTEXT_CHARS

    for chunk in retrieved_chunks:
        text = chunk["text"].strip()
        if len(text) > char_budget:
            # Truncate at last sentence boundary within budget
            truncated = text[:char_budget]
            last_period = truncated.rfind(". ")
            text = truncated[:last_period + 1] if last_period > 0 else truncated
            included.append({**chunk, "text": text})
            break
        included.append(chunk)
        char_budget -= len(text)

    # Build context block
    context_parts = []
    for i, chunk in enumerate(included, start=1):
        context_parts.append(
            f"[{i}] {chunk['method']} ({chunk['year']}) — "
            f"{chunk['file_name']}, page {chunk['page_number']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer (cite sources as [1], [2], etc.):"
    )

    return prompt, included


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, model: str) -> tuple[str, dict]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    text = response.choices[0].message.content
    usage = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
    return text, usage


def _call_gemini(prompt: str, model: str) -> tuple[str, dict]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llm = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = llm.generate_content(prompt)
    text = response.text
    usage = {
        "prompt_tokens":     getattr(response.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "total_tokens":      getattr(response.usage_metadata, "total_token_count", 0),
    }
    return text, usage


def _call_ollama(prompt: str, model: str) -> tuple[str, dict]:
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["message"]["content"]
    usage = {
        "prompt_tokens":     data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return text, usage


def _call_mock(prompt: str, model: str) -> tuple[str, dict]:
    """
    Returns a deterministic fake answer for development and testing.
    No API key or network connection needed.
    """
    answer = (
        "Based on the provided context, the super-resolution methods use various "
        "loss functions and architectural components. [1] describes the use of "
        "perceptual loss combined with adversarial training. [2] introduces "
        "residual channel attention to focus on informative features. "
        "This is a mock response — set LLM_PROVIDER in .env to get real answers."
    )
    usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 60, "total_tokens": 0}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return answer, usage


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": _call_openai,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
    "mock":   _call_mock,
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3",
    "mock":   "mock-model",
}

COST_PER_1K = {
    "gpt-4o-mini":       {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o":            {"prompt": 0.005,   "completion": 0.015},
    "gemini-1.5-flash":  {"prompt": 0.00035, "completion": 0.00105},
    "gemini-1.5-pro":    {"prompt": 0.0035,  "completion": 0.0105},
}


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks.

    Returns a dict with:
        answer       : str
        sources      : list of included chunk dicts (with citation_index)
        token_usage  : prompt/completion/total tokens + estimated cost
        latency_ms   : end-to-end generation time
        provider     : which LLM was used
        model        : which model was used
    """
    # Resolve provider
    provider = provider or os.getenv("LLM_PROVIDER", "mock")
    if provider not in PROVIDERS:
        logger.warning("Unknown provider '%s', falling back to mock", provider)
        provider = "mock"

    model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS[provider])

    # Build prompt
    prompt, included_chunks = build_prompt(question, retrieved_chunks)

    # Add citation index to sources
    sources = [
        {**chunk, "citation_index": i + 1}
        for i, chunk in enumerate(included_chunks)
    ]

    # Call LLM
    logger.info("Generating answer with %s / %s (%d chunks in context)",
                provider, model, len(included_chunks))
    t0 = time.time()
    answer, raw_usage = PROVIDERS[provider](prompt, model)
    latency_ms = (time.time() - t0) * 1000

    # Estimate cost
    rates = COST_PER_1K.get(model, {"prompt": 0.0, "completion": 0.0})
    cost = (
        raw_usage["prompt_tokens"]     / 1000 * rates["prompt"] +
        raw_usage["completion_tokens"] / 1000 * rates["completion"]
    )

    token_usage = {**raw_usage, "estimated_cost_usd": round(cost, 8)}

    # Log to JSONL
    _log_query(question, answer, token_usage, latency_ms, provider, model)

    return {
        "answer":      answer,
        "sources":     sources,
        "token_usage": token_usage,
        "latency_ms":  round(latency_ms, 1),
        "provider":    provider,
        "model":       model,
    }


# ---------------------------------------------------------------------------
# Query logger
# ---------------------------------------------------------------------------

LOG_PATH = Path("logs/query_log.jsonl")


def _log_query(
    question: str,
    answer: str,
    token_usage: dict,
    latency_ms: float,
    provider: str,
    model: str,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question":    question,
        "answer":      answer[:300],
        "provider":    provider,
        "model":       model,
        "latency_ms":  round(latency_ms, 1),
        "token_usage": token_usage,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_feedback(question: str, answer: str, helpful: bool) -> None:
    """Log thumbs up/down feedback from the Streamlit UI."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type":      "feedback",
        "question":  question,
        "answer":    answer[:300],
        "helpful":   helpful,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")