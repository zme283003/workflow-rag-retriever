from ragflow.evaluation import recall_at_k, evaluate, report_text
from ragflow.models import Chunk, EvalCase, Source


def test_recall_at_k():
    docs = ["DOC-A", "DOC-B", "DOC-C"]
    assert recall_at_k(docs, ["DOC-A", "DOC-C"], k=3) == 1.0
    assert recall_at_k(docs, ["DOC-A", "DOC-Z"], k=3) == 0.5
    assert recall_at_k(docs, ["DOC-A"], k=1) == 1.0


def test_evaluate_reports_metrics():
    cases = [
        EvalCase("1", "q1", ["DOC-A"], "correct"),
        EvalCase("2", "q2", ["DOC-B"], "wrong"),  # recall=1 but wrong
    ]
    retrieved = {
        "q1": [Chunk("c1", "t", 1.0, "DOC-A", source=Source.VECTOR)],
        "q2": [Chunk("c2", "t", 1.0, "DOC-B", source=Source.VECTOR)],
    }
    report = evaluate(cases, lambda q: retrieved[q])
    assert report.recall_mean == 1.0
    assert report.recall_one_but_not_correct == 1
    assert abs(report.perfect_recall_correct_rate - 0.5) < 1e-6
    assert "recall=1 but answer not correct" in report_text(report)
