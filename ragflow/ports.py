"""Abstract ports.

The workflow depends only on these interfaces, never on concrete services.
Swap the in-memory demo implementations for real ones without touching the
orchestration code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import Chunk, DocumentRef, Entity


@runtime_checkable
class LLM(Protocol):
    """Minimal chat-completion style interface."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, query: str) -> List[Entity]:
        ...


@runtime_checkable
class MetadataProvider(Protocol):
    """Resolve standardized entities to the documents that mention them."""

    def lookup(self, entities: List[Entity]) -> List[DocumentRef]:
        ...


@runtime_checkable
class VectorRetriever(Protocol):
    """Dense/sparse/hybrid search over chunk collections.

    ``doc_ids`` restricts the search to a document set; empty means no
    restriction. ``scope`` selects a slice such as "summary" or "full_text".
    """

    def search(
        self,
        query: str,
        doc_ids: List[str],
        scope: str = "full_text",
        top_k: int = 10,
    ) -> List[Chunk]:
        ...


@runtime_checkable
class TreeStore(Protocol):
    """Load a document structure tree by path (local file, object store...)."""

    def load(self, toc_path: str) -> Optional[Dict[str, Any]]:
        ...


@runtime_checkable
class ProgressSink(Protocol):
    def report(self, key: str, message: str) -> None:
        ...
