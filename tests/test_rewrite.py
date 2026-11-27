from ragflow.models import Chunk, Source
from ragflow.rewrite import MultiTurnRewriter, SynonymRewriter, QueryRewriter


class EchoLLM:
    def complete(self, system, user, temperature=0.0):
        return "resolved question about EQ-1001"


def test_synonym_expansion():
    rw = SynonymRewriter({"maintenance": ["servicing"]})
    out = rw.rewrite("maintenance manual")
    assert "servicing manual" in out


def test_multiturn_uses_history():
    rw = MultiTurnRewriter(EchoLLM())
    assert rw.rewrite("what about it?", [{"role": "user", "content": "EQ-1001"}]) == [
        "resolved question about EQ-1001"
    ]


def test_query_rewriter_capped_and_deduped():
    rw = QueryRewriter(synonym=SynonymRewriter({"manual": ["handbook", "guide"]}))
    out = rw.rewrite("maintenance manual", max_count=3)
    assert len(out) <= 3
    assert len(out) == len(set(out))
