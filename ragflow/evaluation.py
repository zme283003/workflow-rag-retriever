"""Evaluation framework.

The important lesson from the real-world evaluation this project is based on:
recall alone systematically overstates quality. A case can have recall=1 (all
expected documents were retrieved) and still get a wrong answer. So this module
reports a ``perfect_recall_correct_rate`` and counts
``recall_one_but_not_correct`` explicitly.

All example cases here are synthetic.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from .models import Chunk, EvalCase, EvalReport


def recall_at_k(chunk_docs: Sequence[str], expected_docs: Sequence[str], k: int = 8) -> float:
    """Fraction of expected documents found in the top-k retrieved documents."""
    if not expected_docs:
        return 0.0
    retrieved = list(dict.fromkeys(chunk_docs))[:k]
    hits = sum(1 for doc in expected_docs if doc in retrieved)
    return hits / len(expected_docs)


def evaluate(
    cases: List[EvalCase],
    retrieve_fn,
    k: int = 8,
) -> EvalReport:
    """Run every case through ``retrieve_fn(case.query) -> list[Chunk]``."""
    recalls: List[float] = []
    hits = 0
    full_hits = 0
    perfect_total = 0
    perfect_correct = 0
    recall_one_not_correct = 0
    by_cat: Dict[str, List[float]] = {}

    for case in cases:
        chunks: List[Chunk] = retrieve_fn(case.query)
        docs = [c.document_id for c in chunks]
        rec = recall_at_k(docs, case.expected_docs, k)
        recalls.append(rec)
        by_cat.setdefault(case.category, []).append(rec)
        if any(doc in docs for doc in case.expected_docs):
            hits += 1
        if all(doc in docs for doc in case.expected_docs):
            full_hits += 1
        if rec == 1.0:
            perfect_total += 1
            if case.relevance == "correct":
                perfect_correct += 1
            else:
                recall_one_not_correct += 1

    n = len(cases) or 1
    return EvalReport(
        total=len(cases),
        recall_mean=statistics.mean(recalls) if recalls else 0.0,
        recall_median=statistics.median(recalls) if recalls else 0.0,
        hit_rate=hits / n,
        full_hit_rate=full_hits / n,
        perfect_recall_correct_rate=perfect_correct / perfect_total if perfect_total else 0.0,
        recall_one_but_not_correct=recall_one_not_correct,
        by_category={cat: statistics.mean(vals) for cat, vals in by_cat.items()},
    )


def report_text(report: EvalReport) -> str:
    lines = [
        f"cases: {report.total}",
        f"recall mean / median: {report.recall_mean:.3f} / {report.recall_median:.3f}",
        f"hit rate (>=1 expected doc): {report.hit_rate:.1%}",
        f"full hit rate (all expected docs): {report.full_hit_rate:.1%}",
        f"recall=1 & answer correct: {report.perfect_recall_correct_rate:.1%}",
        f"recall=1 but answer not correct: {report.recall_one_but_not_correct}",
    ]
    if report.by_category:
        lines.append("by category: " + ", ".join(
            f"{cat}={val:.3f}" for cat, val in sorted(report.by_category.items())
        ))
    return "\n".join(lines)
