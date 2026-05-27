"""
src/generation/answer_generator.py  (Day 10 update)
-----------------------------------------------------
Updated to delegate all prompt logic to prompt_manager.py.

Changes from Day 6:
- build_prompt() removed — now lives in prompt_manager.py
- generate_answer() accepts a `prompt_template` arg for A/B testing
- Refusal logic integrated: skips LLM call when corpus has no good match
- Context limiter logs truncation events to logs/query_log.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.generation.prompt_manager import build_prompt, REFUSAL_MESSAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider implementations (unchanged from Day 6)
# ---------------------------------------------------------------------------

def _call_openai(system: str, prompt: str, model: str) -> tuple[str, dict]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
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


def _call_gemini(system: str, prompt: str, model: str) -> tuple[str, dict]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Run: pip install google-generativeai")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llm = genai.GenerativeModel(model_name=model, system_instruction=system)
    response = llm.generate_content(prompt)
    text = response.text
    usage = {
        "prompt_tokens":     getattr(response.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "total_tokens":      getattr(response.usage_metadata, "total_token_count", 0),
    }
    return text, usage


def _call_ollama(system: str, prompt: str, model: str) -> tuple[str, dict]:
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
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


def _call_mock(system: str, prompt: str, model: str) -> tuple[str, dict]:
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
    "gpt-4o-mini":      {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o":           {"prompt": 0.005,   "completion": 0.015},
    "gemini-1.5-flash": {"prompt": 0.00035, "completion": 0.00105},
    "gemini-1.5-pro":   {"prompt": 0.0035,  "completion": 0.0105},
}


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    prompt_template: str = "v2",
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        question          : natural language question
        retrieved_chunks  : list of chunk dicts from the retriever
        provider          : 'openai' | 'gemini' | 'ollama' | 'mock'
        model             : model name (None = read from .env)
        prompt_template   : 'v1' | 'v2' | 'v3_concise' (see prompt_manager)

    Returns a dict with answer, sources, token_usage, latency_ms, etc.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "mock")
    if provider not in PROVIDERS:
        logger.warning("Unknown provider '%s', falling back to mock", provider)
        provider = "mock"
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS[provider])

    # Build prompt (handles refusal + context limiting internally)
    system, user_prompt, included_chunks, refused = build_prompt(
        question=question,
        chunks=retrieved_chunks,
        template=prompt_template,
    )

    # Add citation index to sources
    sources = [
        {**chunk, "citation_index": i + 1}
        for i, chunk in enumerate(included_chunks)
    ]

    # Short-circuit if refused
    if refused:
        result = {
            "answer":      REFUSAL_MESSAGE,
            "sources":     [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0, "estimated_cost_usd": 0.0},
            "latency_ms":  0.0,
            "provider":    provider,
            "model":       model,
            "refused":     True,
        }
        _log_query(question, REFUSAL_MESSAGE, result["token_usage"], 0.0, provider, model)
        return result

    # Call LLM
    logger.info("Generating with %s/%s (%d chunks, template=%s)",
                provider, model, len(included_chunks), prompt_template)
    t0 = time.time()
    answer, raw_usage = PROVIDERS[provider](system, user_prompt, model)
    latency_ms = (time.time() - t0) * 1000

    # Estimate cost
    rates = COST_PER_1K.get(model, {"prompt": 0.0, "completion": 0.0})
    cost = (
        raw_usage["prompt_tokens"]     / 1000 * rates["prompt"] +
        raw_usage["completion_tokens"] / 1000 * rates["completion"]
    )
    token_usage = {**raw_usage, "estimated_cost_usd": round(cost, 8)}

    _log_query(question, answer, token_usage, latency_ms, provider, model)

    return {
        "answer":      answer,
        "sources":     sources,
        "token_usage": token_usage,
        "latency_ms":  round(latency_ms, 1),
        "provider":    provider,
        "model":       model,
        "refused":     False,
    }


# ---------------------------------------------------------------------------
# Query logger
# ---------------------------------------------------------------------------

LOG_PATH = Path("logs/query_log.jsonl")


def _log_query(question, answer, token_usage, latency_ms, provider, model):
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