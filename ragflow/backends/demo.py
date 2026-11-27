"""In-memory demo backend.

Fully synthetic data: generic equipment IDs (EQ-xxxx), generic document titles
and hand-written passages. No customer names, real identifiers, endpoints or
secrets. Runs offline with zero third-party dependencies.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models import Chunk, Source
from ..ports import TreeStore, VectorRetriever
from ..progress import NullProgressSink

# --- Synthetic metadata: entity -> documents -------------------------------
DEMO_METADATA: Dict[str, List[Dict[str, str]]] = {
    "EQ-1001": [
        {"doc_id": "DOC-A", "doc_name": "EQ-1001 maintenance manual", "toc_path": "data/demo/tree_EQ-1001.json"},
        {"doc_id": "DOC-B", "doc_name": "EQ-1001 inspection report", "toc_path": "data/demo/tree_EQ-1001.json"},
    ],
    "EQ-2002": [
        {"doc_id": "DOC-C", "doc_name": "EQ-2002 operating guide", "toc_path": "data/demo/tree_EQ-2002.json"},
    ],
}

# --- Synthetic passages ----------------------------------------------------
DEMO_CHUNKS: List[Chunk] = [
    Chunk("c-sum-A", "Equipment EQ-1001 maintenance cycle is 180 days under normal load.", 0.9, "DOC-A", "EQ-1001 maintenance manual", Source.SUMMARY),
    Chunk("c-sum-C", "Equipment EQ-2002 operating guide covers startup and shutdown procedures.", 0.8, "DOC-C", "EQ-2002 operating guide", Source.SUMMARY),
    Chunk("c-full-A1", "The maintenance interval for EQ-1001 is 180 days, with a tolerance of plus or minus 10 days.", 0.7, "DOC-A", "EQ-1001 maintenance manual", Source.VECTOR),
    Chunk("c-full-A2", "Lubricant grade for EQ-1001 is ISO VG 46; replace every 180 days.", 0.6, "DOC-A", "EQ-1001 maintenance manual", Source.VECTOR),
    Chunk("c-full-B1", "Inspection of EQ-1001 recorded a vibration level of 2.1 mm/s, within the acceptable range.", 0.65, "DOC-B", "EQ-1001 inspection report", Source.VECTOR),
    Chunk("c-full-C1", "The operating guide for EQ-2002: startup requires a 30 second preheat; shutdown follows the reverse sequence.", 0.75, "DOC-C", "EQ-2002 operating guide", Source.VECTOR),
]


class DemoVectorRetriever:
    """Lexical overlap retriever over synthetic passages."""

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks = chunks if chunks is not None else [Chunk(**c.__dict__) for c in DEMO_CHUNKS]

    def search(self, query: str, doc_ids: List[str], scope: str = "full_text", top_k: int = 10) -> List[Chunk]:
        wanted = set(doc_ids)
        tokens = set(re.findall(r"\w+", query.lower()))
        results: List[Chunk] = []
        for chunk in self.chunks:
            if scope == "summary" and chunk.source != Source.SUMMARY:
                continue
            if scope == "full_text" and chunk.source == Source.SUMMARY:
                continue
            if wanted and chunk.document_id not in wanted:
                continue
            overlap = sum(1 for t in tokens if t and t in chunk.text.lower())
            if overlap or not tokens:
                results.append(Chunk(**{**chunk.__dict__, "score": chunk.score + 0.1 * overlap}))
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]


class LocalTreeStore:
    """Load structure trees from a local directory."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

    def load(self, toc_path: str) -> Optional[Dict[str, Any]]:
        import json

        path = toc_path
        if not os.path.isabs(path) and self.base_dir:
            # base_dir is <root>/ragflow/backends; project root is two levels up.
            path = os.path.normpath(os.path.join(self.base_dir, "..", "..", toc_path))
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


@dataclass
class DemoBundle:
    metadata: Dict[str, List[Dict[str, str]]]
    retriever: DemoVectorRetriever
    tree_store: LocalTreeStore
    progress: NullProgressSink


def build_demo_bundle() -> DemoBundle:
    """Build a ready-to-use, fully offline bundle."""
    return DemoBundle(
        metadata=DEMO_METADATA,
        retriever=DemoVectorRetriever(),
        tree_store=LocalTreeStore(),
        progress=NullProgressSink(),
    )
