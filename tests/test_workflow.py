import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragflow.backends.demo import build_demo_bundle
from ragflow.metadata import InMemoryMetadataProvider
from ragflow.models import RetrieveRequest
from ragflow.pageindex import PageIndexRetriever
from ragflow.workflow import WorkflowRetriever


def build():
    bundle = build_demo_bundle()
    return WorkflowRetriever(
        vector_retriever=bundle.retriever,
        metadata_provider=InMemoryMetadataProvider(bundle.metadata),
        tree_store=bundle.tree_store,
        page_index=PageIndexRetriever(bundle.tree_store),
    )


def test_entity_constraint_narrows_documents():
    wf = build()
    result = wf.retrieve(RetrieveRequest(query="maintenance interval EQ-1001"))
    assert any(e.text == "EQ-1001" for e in result.entities)
    assert set(d.doc_id for d in result.documents) == {"DOC-A", "DOC-B"}
    assert result.chunks


def test_summary_fallback_when_no_entity():
    wf = build()
    result = wf.retrieve(RetrieveRequest(query="general operating guide"))
    assert result.used_fallback is True
    assert result.chunks


def test_pageindex_source_present():
    wf = build()
    result = wf.retrieve(RetrieveRequest(query="vibration inspection EQ-1001", top_k=10))
    assert any(c.source.value == "pageindex" for c in result.chunks)


def test_progress_events_emitted():
    wf = build()
    result = wf.retrieve(RetrieveRequest(query="startup EQ-2002"))
    assert len(result.progress) >= 5
    assert result.progress[-1].message.startswith("done")
