"""
tests/test_logger.py
Tests for src/monitoring/logger.py and scripts/analyze_logs.py
Run from project root: pytest tests/test_logger.py -v
"""

import json
import pytest
from pathlib import Path
from src.monitoring.logger import (
    load_query_log,
    log_query_event,
    log_feedback_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_file(tmp_path):
    """Return path to a temp JSONL log file."""
    return tmp_path / "test_query_log.jsonl"


@pytest.fixture
def populated_log(log_file):
    """Write 5 query records and 2 feedback records to a temp log."""
    records = []
    for i in range(5):
        records.append({
            "timestamp":     f"2024-01-0{i+1}T10:00:00Z",
            "type":          "query",
            "question":      f"Question {i}",
            "answer":        f"Answer {i}",
            "provider":      "mock",
            "model":         "mock-model",
            "refused":       i == 4,
            "use_reranker":  i % 2 == 0,
            "latency_ms":    100 + i * 50,
            "retrieval_ms":  30 + i * 5,
            "rerank_ms":     20 if i % 2 == 0 else 0,
            "generation_ms": 50 + i * 10,
            "token_usage": {
                "prompt_tokens":      100 + i * 10,
                "completion_tokens":  50 + i * 5,
                "total_tokens":       150 + i * 15,
                "estimated_cost_usd": 0.00005 * (i + 1),
            },
        })

    for helpful in [True, False]:
        records.append({
            "timestamp": "2024-01-06T10:00:00Z",
            "type":      "feedback",
            "question":  "Test question",
            "answer":    "Test answer",
            "helpful":   helpful,
        })

    with open(log_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return log_file


# ---------------------------------------------------------------------------
# load_query_log
# ---------------------------------------------------------------------------

class TestLoadQueryLog:

    def test_loads_all_records(self, populated_log):
        records = load_query_log(populated_log)
        assert len(records) == 7   # 5 queries + 2 feedback

    def test_returns_empty_for_missing_file(self, tmp_path):
        records = load_query_log(tmp_path / "nonexistent.jsonl")
        assert records == []

    def test_records_are_dicts(self, populated_log):
        records = load_query_log(populated_log)
        for r in records:
            assert isinstance(r, dict)

    def test_skips_malformed_lines(self, tmp_path):
        log = tmp_path / "bad.jsonl"
        log.write_text('{"valid": true}\nNOT JSON\n{"also": "valid"}\n')
        records = load_query_log(log)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# log_query_event and log_feedback_event
# ---------------------------------------------------------------------------

class TestLogEvents:

    def test_query_event_written(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        import src.monitoring.logger as logger_mod
        monkeypatch.setattr(logger_mod, "QUERY_LOG", log_path)

        log_query_event(
            question="What is PSNR?",
            answer="PSNR is a metric.",
            provider="mock",
            model="mock-model",
            latency_ms=120.5,
            token_usage={"prompt_tokens": 50, "completion_tokens": 20,
                         "total_tokens": 70, "estimated_cost_usd": 0.0},
        )

        records = load_query_log(log_path)
        assert len(records) == 1
        assert records[0]["type"] == "query"
        assert records[0]["question"] == "What is PSNR?"

    def test_feedback_event_written(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        import src.monitoring.logger as logger_mod
        monkeypatch.setattr(logger_mod, "QUERY_LOG", log_path)

        log_feedback_event("test q", "test a", helpful=True)

        records = load_query_log(log_path)
        assert len(records) == 1
        assert records[0]["type"] == "feedback"
        assert records[0]["helpful"] is True

    def test_multiple_events_appended(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        import src.monitoring.logger as logger_mod
        monkeypatch.setattr(logger_mod, "QUERY_LOG", log_path)

        for _ in range(3):
            log_query_event(
                question="q", answer="a", provider="mock", model="m",
                latency_ms=100,
                token_usage={"prompt_tokens": 10, "completion_tokens": 5,
                             "total_tokens": 15, "estimated_cost_usd": 0.0},
            )

        records = load_query_log(log_path)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# analyze_logs (integration)
# ---------------------------------------------------------------------------

class TestAnalyzeLogs:

    def test_analyze_returns_report(self, populated_log):
        import sys
        sys.path.insert(0, "scripts")
        from analyze_logs import analyze
        report = analyze(populated_log)
        assert "total_queries" in report
        assert report["total_queries"] == 5

    def test_latency_stats_present(self, populated_log):
        from analyze_logs import analyze
        report = analyze(populated_log)
        for key in ["mean", "p50", "p95", "p99", "max"]:
            assert key in report["latency_ms"]

    def test_refusal_count_correct(self, populated_log):
        from analyze_logs import analyze
        report = analyze(populated_log)
        assert report["refusals"]["count"] == 1

    def test_feedback_count_correct(self, populated_log):
        from analyze_logs import analyze
        report = analyze(populated_log)
        assert report["feedback"]["total"] == 2
        assert report["feedback"]["helpful"] == 1

    def test_token_totals_correct(self, populated_log):
        from analyze_logs import analyze
        report = analyze(populated_log)
        assert report["token_usage"]["total_tokens"] > 0

    def test_empty_log_returns_error(self, tmp_path):
        from analyze_logs import analyze
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        report = analyze(empty)
        assert "error" in report