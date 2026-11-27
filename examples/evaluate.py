"""Offline evaluation demo. Run: python examples/evaluate.py"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import build
from ragflow.evaluation import evaluate, report_text
from ragflow.models import EvalCase, RetrieveRequest

# Synthetic labeled cases. ``relevance`` is the judged answer quality.
CASES = [
    EvalCase("s1", "maintenance interval for EQ-1001", ["DOC-A"], "correct"),
    EvalCase("s2", "lubricant grade of EQ-1001", ["DOC-A"], "partial"),
    EvalCase("s3", "vibration inspection of EQ-1001", ["DOC-B"], "correct"),
    EvalCase("s4", "startup procedure EQ-2002", ["DOC-C"], "correct"),
    EvalCase("s5", "shutdown procedure EQ-2002", ["DOC-C"], "wrong", category="table"),
]


def main() -> None:
    workflow = build()
    report = evaluate(CASES, lambda q: workflow.retrieve(RetrieveRequest(query=q)).chunks)
    print(report_text(report))


if __name__ == "__main__":
    main()
