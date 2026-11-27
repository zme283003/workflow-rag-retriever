from ragflow.fusion import dedupe, normalize, rrf_fuse
from ragflow.models import Chunk, Source


def chunk(cid, score):
    return Chunk(cid, cid, score, "DOC-A", source=Source.VECTOR)


def test_normalize_to_unit_interval():
    out = normalize([chunk("a", 2), chunk("b", 4)])
    assert [round(c.score, 3) for c in out] == [0.0, 1.0]


def test_rrf_rewards_consensus():
    list_a = [chunk("a", 0.9), chunk("b", 0.8)]
    list_b = [chunk("a", 0.7), chunk("c", 0.6)]
    fused = rrf_fuse([list_a, list_b])
    assert fused[0].id == "a"


def test_dedupe_keeps_best():
    out = dedupe([chunk("a", 0.5), chunk("a", 0.9)])
    assert len(out) == 1
    assert out[0].score == 0.9
