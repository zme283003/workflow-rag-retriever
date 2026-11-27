"""Metadata provider.

Metadata is the bridge between a business entity and the documents about it.
The key design idea: instead of guessing which documents are relevant by
semantic similarity over the whole corpus, resolve the entity to a narrow
document set first, then search inside that set.

This module ships an in-memory implementation. A production implementation
would call an upstream metadata service; the workflow only depends on the
``MetadataProvider`` protocol.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .models import DocumentRef, Entity


class InMemoryMetadataProvider:
    """Resolve entities to documents from a simple mapping.

    ``index`` maps a canonical entity id to a list of document records, each
    with ``doc_id``, ``doc_name`` and optional ``toc_path``.
    """

    def __init__(self, index: Dict[str, Iterable[Dict[str, str]]]):
        self.index = {k.upper(): list(v) for k, v in index.items()}

    def lookup(self, entities: List[Entity]) -> List[DocumentRef]:
        refs: List[DocumentRef] = []
        seen = set()
        for entity in entities:
            for record in self.index.get(entity.text.upper(), []):
                doc_id = record.get("doc_id", "")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                refs.append(
                    DocumentRef(
                        doc_id=doc_id,
                        doc_name=record.get("doc_name", doc_id),
                        toc_path=record.get("toc_path"),
                    )
                )
        return refs
