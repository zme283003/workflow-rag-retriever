"""Core data models.

All domain names are intentionally generic (equipment IDs, report titles). No
real customer, well, organization, endpoint or credential data is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Source(str, Enum):
    """Which retrieval channel produced a chunk."""

    VECTOR = "vector"
    SUMMARY = "summary"
    PAGEINDEX = "pageindex"


@dataclass
class Entity:
    """A normalized business entity extracted from the query.

    ``raw`` is the surface form seen in the user query; ``text`` is the
    standardized identifier used to look up metadata.
    """

    raw: str
    text: str
    type: str = "asset"


@dataclass
class DocumentRef:
    """A document resolved from metadata, plus its optional structure tree."""

    doc_id: str
    doc_name: str
    toc_path: Optional[str] = None


@dataclass
class Chunk:
    """A retrieved passage ready to be handed to the generator."""

    id: str
    text: str
    score: float
    document_id: str
    document_name: str = ""
    source: Source = Source.VECTOR
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source": self.source.value,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class RetrieveRequest:
    query: str
    top_k: int = 8
    # Optional hint to skip entity extraction when the caller already knows it.
    entity_text: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressEvent:
    key: str
    message: str


@dataclass
class RetrieveResult:
    """Full trace of a workflow retrieval, useful for debugging and eval."""

    query: str
    chunks: List[Chunk]
    entities: List[Entity] = field(default_factory=list)
    documents: List[DocumentRef] = field(default_factory=list)
    used_fallback: bool = False
    progress: List[ProgressEvent] = field(default_factory=list)

    def chunks_as_dicts(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.chunks]


@dataclass
class EvalCase:
    """One labeled evaluation example."""

    case_id: str
    query: str
    expected_docs: List[str]
    relevance: str  # "correct" | "partial" | "wrong"
    answer: str = ""
    category: str = "text"


@dataclass
class EvalReport:
    total: int
    recall_mean: float
    recall_median: float
    hit_rate: float
    full_hit_rate: float
    perfect_recall_correct_rate: float
    recall_one_but_not_correct: int
    by_category: Dict[str, float] = field(default_factory=dict)
