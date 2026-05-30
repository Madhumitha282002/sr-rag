"""
scripts/analyze_logs.py
------------------------
Summarises query_log.jsonl into a performance report.

Usage:
    python scripts/analyze_logs.py
    python scripts/analyze_logs.py --log logs/query_log.jsonl
    python scripts/analyze_logs.py --json   (machine-readable output)

Metrics reported:
  - Total queries, refusal rate
  - Latency: mean, p50, p95, p99, max
  - Per-stage latency: retrieval, reranking, generation
  - Token usage and estimated cost
  - Feedback: thumbs up/down rate
  - Provider breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.logger import load_query_log


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return round(sorted_vals[idx], 1)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(log_path: str | Path) -> dict:
    records = load_query_log(log_path)
    if not records:
        return {"error": f"No records found in {log_path}"}

    queries   = [r for r in records if r.get("type", "query") == "query"]
    feedbacks = [r for r in records if r.get("type", "feedback") == "feedback"]

    if not queries:
        return {"error": "No query records found"}

    # Latency
    latencies    = [r["latency_ms"]    for r in queries if "latency_ms"    in r]
    ret_ms       = [r["retrieval_ms"]  for r in queries if "retrieval_ms"  in r]
    rerank_ms    = [r["rerank_ms"]     for r in queries if "rerank_ms"     in r and r["rerank_ms"] > 0]
    gen_ms       = [r["generation_ms"] for r in queries if "generation_ms" in r]

    # Tokens + cost
    prompt_tokens      = [r["token_usage"]["prompt_tokens"]     for r in queries if "token_usage" in r]
    completion_tokens  = [r["token_usage"]["completion_tokens"] for r in queries if "token_usage" in r]
    costs              = [r["token_usage"].get("estimated_cost_usd", 0) for r in queries if "token_usage" in r]

    # Refusals
    refused  = [r for r in queries if r.get("refused", False)]
    reranked = [r for r in queries if r.get("use_reranker", False)]

    # Providers
    providers: dict[str, int] = {}
    for r in queries:
        p = r.get("provider", "unknown")
        providers[p] = providers.get(p, 0) + 1

    # Feedback
    helpful   = sum(1 for f in feedbacks if f.get("helpful") is True)
    unhelpful = sum(1 for f in feedbacks if f.get("helpful") is False)

    return {
        "total_queries":   len(queries),
        "total_records":   len(records),
        "date_range": {
            "first": queries[0].get("timestamp",  "?"),
            "last":  queries[-1].get("timestamp", "?"),
        },
        "latency_ms": {
            "mean": mean(latencies),
            "p50":  percentile(latencies, 50),
            "p95":  percentile(latencies, 95),
            "p99":  percentile(latencies, 99),
            "max":  max(latencies) if latencies else 0,
        },
        "stage_latency_ms": {
            "retrieval_mean":  mean(ret_ms),
            "rerank_mean":     mean(rerank_ms) if rerank_ms else 0,
            "generation_mean": mean(gen_ms),
        },
        "refusals": {
            "count": len(refused),
            "rate":  round(len(refused) / len(queries), 3),
        },
        "reranker": {
            "queries_using_reranker": len(reranked),
            "rate": round(len(reranked) / len(queries), 3),
        },
        "token_usage": {
            "total_prompt_tokens":     sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
            "total_tokens":            sum(prompt_tokens) + sum(completion_tokens),
            "mean_tokens_per_query":   mean([p + c for p, c in zip(prompt_tokens, completion_tokens)]),
            "total_cost_usd":          round(sum(costs), 6),
            "mean_cost_per_query_usd": round(mean(costs), 8),
        },
        "providers": providers,
        "feedback": {
            "total":     len(feedbacks),
            "helpful":   helpful,
            "unhelpful": unhelpful,
            "rate":      round(helpful / len(feedbacks), 3) if feedbacks else None,
        },
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    SEP = "=" * 55

    print(f"\n{SEP}")
    print(f"  SR-RAG Query Log Report")
    print(SEP)
    print(f"  Total queries  : {report['total_queries']}")
    print(f"  Date range     : {report['date_range']['first'][:10]} → {report['date_range']['last'][:10]}")

    print(f"\n--- Latency (ms) ---")
    lat = report["latency_ms"]
    print(f"  Mean : {lat['mean']}")
    print(f"  p50  : {lat['p50']}")
    print(f"  p95  : {lat['p95']}")
    print(f"  p99  : {lat['p99']}")
    print(f"  Max  : {lat['max']}")

    print(f"\n--- Stage breakdown (mean ms) ---")
    stage = report["stage_latency_ms"]
    print(f"  Retrieval  : {stage['retrieval_mean']}")
    if stage["rerank_mean"]:
        print(f"  Reranking  : {stage['rerank_mean']}")
    print(f"  Generation : {stage['generation_mean']}")

    print(f"\n--- Refusals ---")
    ref = report["refusals"]
    print(f"  Count : {ref['count']}")
    print(f"  Rate  : {ref['rate']:.1%}")

    print(f"\n--- Token usage ---")
    tok = report["token_usage"]
    print(f"  Total tokens      : {tok['total_tokens']:,}")
    print(f"  Mean per query    : {tok['mean_tokens_per_query']:.0f}")
    print(f"  Total cost (USD)  : ${tok['total_cost_usd']:.6f}")
    print(f"  Mean cost/query   : ${tok['mean_cost_per_query_usd']:.8f}")

    print(f"\n--- Providers ---")
    for provider, count in report["providers"].items():
        print(f"  {provider:<12} : {count} queries")

    fb = report["feedback"]
    if fb["total"] > 0:
        print(f"\n--- Feedback ---")
        print(f"  Total     : {fb['total']}")
        print(f"  Helpful   : {fb['helpful']}")
        print(f"  Unhelpful : {fb['unhelpful']}")
        print(f"  Rate      : {fb['rate']:.1%}")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse SR-RAG query logs.")
    parser.add_argument(
        "--log", default="logs/query_log.jsonl",
        help="Path to query_log.jsonl (default: logs/query_log.jsonl)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of human-readable report",
    )
    args = parser.parse_args()

    report = analyze(args.log)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()