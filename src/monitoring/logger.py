"""
src/monitoring/logger.py
-------------------------
Centralised logging setup for SR-RAG.

Sets up structured JSON logging to logs/app.log alongside
human-readable console output. Call setup_logging() once at
startup (in pipeline.py, main scripts, or the Streamlit app).

All src/ modules use standard logging.getLogger(__name__) —
this module configures the root logger so every module's output
is captured automatically.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Makes logs easy to parse with jq or pandas.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """
    Configure root logger with:
      - JSON file handler  → logs/app.log
      - Console handler    → stdout (human-readable)

    Call once at startup. Safe to call multiple times
    (idempotent — won't add duplicate handlers).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    # Avoid adding handlers twice (e.g. Streamlit reruns)
    if root.handlers:
        return

    root.setLevel(level)

    # File handler — JSON
    file_handler = logging.FileHandler(
        log_dir / log_file, encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # Console handler — human-readable
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root.info("Logging initialised → %s/%s", log_dir, log_file)


# ---------------------------------------------------------------------------
# Query event logger (structured, separate from app.log)
# ---------------------------------------------------------------------------

QUERY_LOG = Path("logs/query_log.jsonl")


def log_query_event(
    question: str,
    answer: str,
    provider: str,
    model: str,
    latency_ms: float,
    token_usage: dict[str, Any],
    refused: bool = False,
    use_reranker: bool = False,
    retrieval_ms: float = 0.0,
    rerank_ms: float = 0.0,
    generation_ms: float = 0.0,
) -> None:
    """
    Append a structured query record to logs/query_log.jsonl.
    Each line is a valid JSON object parseable by analyze_logs.py.
    """
    QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type":          "query",
        "question":      question[:200],
        "answer":        answer[:300],
        "provider":      provider,
        "model":         model,
        "refused":       refused,
        "use_reranker":  use_reranker,
        "latency_ms":    round(latency_ms, 1),
        "retrieval_ms":  round(retrieval_ms, 1),
        "rerank_ms":     round(rerank_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "token_usage":   token_usage,
    }
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_feedback_event(
    question: str,
    answer: str,
    helpful: bool,
) -> None:
    """Append a feedback event to query_log.jsonl."""
    QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type":      "feedback",
        "question":  question[:200],
        "answer":    answer[:300],
        "helpful":   helpful,
    }
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Log reader helper (used by analyze_logs.py and the notebook)
# ---------------------------------------------------------------------------

def load_query_log(
    log_path: str | Path = QUERY_LOG,
) -> list[dict[str, Any]]:
    """
    Load all records from query_log.jsonl.
    Returns a list of dicts (one per line).
    """
    path = Path(log_path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records