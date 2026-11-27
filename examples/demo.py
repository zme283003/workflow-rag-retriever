"""Offline end-to-end demo. Run: python examples/demo.py"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragflow.backends.demo import build_demo_bundle
from ragflow.metadata import InMemoryMetadataProvider
from ragflow.models import RetrieveRequest
from ragflow.pageindex import PageIndexRetriever
from ragflow.workflow import WorkflowRetriever


def build() -> WorkflowRetriever:
    bundle = build_demo_bundle()
    return WorkflowRetriever(
        vector_retriever=bundle.retriever,
        metadata_provider=InMemoryMetadataProvider(bundle.metadata),
        tree_store=bundle.tree_store,
        page_index=PageIndexRetriever(bundle.tree_store),
    )


def main() -> None:
    workflow = build()
    questions = [
        "What is the maintenance interval for EQ-1001?",
        "How do I start up EQ-2002?",
        "Tell me about the vibration inspection of EQ-1001.",
    ]
    for q in questions:
        print("=" * 64)
        print("Q:", q)
        result = workflow.retrieve(RetrieveRequest(query=q, top_k=5))
        print("entities:", [e.text for e in result.entities] or "none (summary fallback)")
        print("documents:", [d.doc_id for d in result.documents])
        for c in result.chunks:
            print(f"  [{c.source.value:9s}] {c.score:.3f}  {c.text[:70]}")


if __name__ == "__main__":
    main()
