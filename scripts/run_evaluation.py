"""
scripts/run_evaluation.py
--------------------------
Runs the full SR-RAG evaluation suite and saves a report.

Covers:
  1. Retrieval metrics (Recall@K, MRR, Hit@K) — dense vs reranked
  2. Answer quality metrics (keyword coverage, citation coverage, grounding)
  3. Refusal accuracy on unanswerable questions
  4. Latency and token cost summary

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --output data/processed/eval_report.json
    python scripts/run_evaluation.py --reranker   (include reranked comparison)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever
from src.pipeline import SRRagPipeline
from src.evaluation.retrieval_metrics import (
    load_eval_questions,
    recall_at_k,
    precision_at_k,
    mrr,
    hit_at_k,
    evaluate_batch,
)
from src.evaluation.answer_metrics import evaluate_batch_answers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EVAL_CSV     = "data/processed/eval_questions.csv"
K_VALUES     = [1, 3, 5]


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def run_retrieval_eval(
    questions: list[dict],
    retriever: Retriever,
    use_reranker: bool = False,
    top_k: int = 5,
) -> tuple[list[dict], dict]:
    """Run retrieval on all questions and compute metrics."""
    from src.retrieval.reranker import Reranker

    reranker = Reranker() if use_reranker else None
    results, latencies = [], []

    for q in questions:
        t0 = time.time()
        retrieval = retriever.retrieve(q["question"], top_k=top_k * 3 if use_reranker else top_k)
        chunks = retrieval["results"]

        if use_reranker and reranker and chunks:
            chunks = reranker.rerank(q["question"], chunks, top_k=top_k)

        latencies.append((time.time() - t0) * 1000)
        retrieved_methods = [c["method"] for c in chunks]

        results.append({
            "question_id":       q["question_id"],
            "question":          q["question"],
            "difficulty":        q["difficulty"],
            "expected_methods":  q["expected_methods"],
            "retrieved_methods": retrieved_methods,
            "refused":           len(chunks) == 0,
            "latency_ms":        round(latencies[-1], 1),
        })

    metrics = evaluate_batch(results, k_values=K_VALUES)
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
    metrics["p95_latency_ms"] = round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
    return results, metrics


# ---------------------------------------------------------------------------
# Answer quality evaluation
# ---------------------------------------------------------------------------

def run_answer_eval(
    questions: list[dict],
    pipeline: SRRagPipeline,
) -> tuple[list[dict], dict]:
    """Run full pipeline on all questions and evaluate answer quality."""
    eval_records, latencies, costs = [], [], []

    for i, q in enumerate(questions):
        logger.info("[%d/%d] %s", i + 1, len(questions), q["question"][:60])
        t0 = time.time()

        result = pipeline.query(q["question"], top_k=5)
        latencies.append((time.time() - t0) * 1000)
        costs.append(result["token_usage"].get("estimated_cost_usd", 0))

        eval_records.append({
            "question":          q["question"],
            "question_id":       q["question_id"],
            "difficulty":        q["difficulty"],
            "expected_keywords": q["expected_keywords"],
            "answer":            result["answer"],
            "sources":           result["sources"],
            "refused":           result.get("refused", False),
        })

    report = evaluate_batch_answers(eval_records)
    report["overall"]["avg_latency_ms"]    = round(sum(latencies) / len(latencies), 1)
    report["overall"]["p95_latency_ms"]    = round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
    report["overall"]["total_cost_usd"]    = round(sum(costs), 6)
    report["overall"]["mean_cost_per_query"] = round(sum(costs) / len(costs), 8)

    return eval_records, report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  SR-RAG Full Evaluation Report")
    print(SEP)

    # Retrieval — dense
    print("\n── Retrieval (dense only) ──────────────────────────────")
    dense = report["retrieval_dense"]["overall"]
    for k in K_VALUES:
        print(f"  Recall@{k}    : {dense.get(f'recall@{k}', 0):.3f}")
    print(f"  MRR         : {dense.get('mrr', 0):.3f}")
    for k in K_VALUES:
        print(f"  Hit@{k}       : {dense.get(f'hit@{k}', 0):.3f}")
    print(f"  Avg latency : {dense.get('avg_latency_ms', 0):.0f} ms")

    # Retrieval — reranked (if present)
    if "retrieval_reranked" in report:
        print("\n── Retrieval (with reranker) ───────────────────────────")
        reranked = report["retrieval_reranked"]["overall"]
        for k in K_VALUES:
            delta = reranked.get(f"recall@{k}", 0) - dense.get(f"recall@{k}", 0)
            print(f"  Recall@{k}    : {reranked.get(f'recall@{k}', 0):.3f}  ({delta:+.3f})")
        delta_mrr = reranked.get("mrr", 0) - dense.get("mrr", 0)
        print(f"  MRR         : {reranked.get('mrr', 0):.3f}  ({delta_mrr:+.3f})")
        print(f"  Avg latency : {reranked.get('avg_latency_ms', 0):.0f} ms")

    # Answer quality
    if "answer_quality" in report:
        print("\n── Answer Quality ──────────────────────────────────────")
        aq = report["answer_quality"]["overall"]
        print(f"  Composite score  : {aq.get('composite_score', 0):.3f}")
        print(f"  Keyword coverage : {aq.get('keyword_coverage', 0):.3f}")
        print(f"  Citation coverage: {aq.get('citation_coverage', 0):.3f}")
        print(f"  Source grounding : {aq.get('source_grounding', 0):.3f}")
        print(f"  Length OK rate   : {aq.get('length_ok_rate', 0):.3f}")
        if "refusal_accuracy" in aq:
            print(f"  Refusal accuracy : {aq.get('refusal_accuracy', 0):.3f}")
        print(f"  Avg latency      : {aq.get('avg_latency_ms', 0):.0f} ms")
        print(f"  Total cost       : ${aq.get('total_cost_usd', 0):.6f}")

        # By difficulty
        print("\n── Answer Quality by Difficulty ────────────────────────")
        for diff, metrics in report["answer_quality"].get("by_difficulty", {}).items():
            print(f"  {diff:<14}: composite={metrics.get('composite_score', 0):.3f}  n={metrics.get('n', 0)}")

    print(f"\n{SEP}")
    print(f"  Report saved to: {report.get('output_path', 'N/A')}")
    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SR-RAG evaluation suite.")
    parser.add_argument("--output", default="data/processed/eval_report.json")
    parser.add_argument("--reranker", action="store_true",
                        help="Also run reranked retrieval for comparison")
    parser.add_argument("--skip-answers", action="store_true",
                        help="Skip answer quality eval (retrieval only)")
    args = parser.parse_args()

    logger.info("Loading eval questions...")
    questions = load_eval_questions(EVAL_CSV)
    logger.info("Loaded %d questions", len(questions))

    retriever = Retriever()
    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. Dense retrieval
    logger.info("Running dense retrieval eval...")
    _, dense_metrics = run_retrieval_eval(questions, retriever, use_reranker=False)
    report["retrieval_dense"] = dense_metrics

    # 2. Reranked retrieval (optional)
    if args.reranker:
        logger.info("Running reranked retrieval eval...")
        _, reranked_metrics = run_retrieval_eval(questions, retriever, use_reranker=True)
        report["retrieval_reranked"] = reranked_metrics

    # 3. Answer quality
    if not args.skip_answers:
        logger.info("Running answer quality eval (%d questions)...", len(questions))
        pipeline = SRRagPipeline()
        _, answer_report = run_answer_eval(questions, pipeline)
        report["answer_quality"] = answer_report

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(out_path)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", out_path)

    print_report(report)


if __name__ == "__main__":
    main()