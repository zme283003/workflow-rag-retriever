import json
import os
import tempfile

from ragflow.pageindex import iter_nodes, strip_text, PageIndexRetriever


TREE = {
    "structure": [
        {
            "node_id": "n1",
            "title": "Maintenance",
            "summary": "Interval is 180 days.",
            "text": "heavy body text",
            "nodes": [
                {"node_id": "n1.1", "title": "Interval", "summary": "180 days", "text": "x"},
            ],
        }
    ]
}


def test_iter_nodes_flattens():
    ids = [n["node_id"] for n in iter_nodes(TREE)]
    assert ids == ["n1", "n1.1"]


def test_strip_text_removes_fields():
    slim = strip_text(TREE)
    assert "text" not in json.dumps(slim)
    assert slim["structure"][0]["title"] == "Maintenance"


def test_lexical_fallback_prefers_specific_child():
    class Store:
        def load(self, path):
            return TREE

    retriever = PageIndexRetriever(Store(), llm=None)
    selected = retriever.select_node_ids(TREE, "interval")
    # The specific child ranks above its generic parent (hierarchy-first rule).
    assert selected[0] == "n1.1"
    assert "n1" in selected
