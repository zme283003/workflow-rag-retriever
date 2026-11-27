"""workflow-rag-retriever

A clean reference implementation of an entity-constrained, multi-stage RAG
retrieval workflow:

    entity extraction -> metadata constraint -> summary fallback
    -> parallel (vector + document-tree) retrieval -> fusion -> response

The package is dependency-free at import time. External services (LLM, vector
store, metadata store, object storage) are abstracted behind small protocols in
:mod:`ragflow.ports`, so the included in-memory backend runs offline while a
production deployment can plug in its own implementations.
"""

from .models import (
    Chunk,
    DocumentRef,
    Entity,
    EvalCase,
    EvalReport,
    ProgressEvent,
    RetrieveRequest,
    RetrieveResult,
)
from .workflow import WorkflowRetriever

__all__ = [
    "Chunk",
    "DocumentRef",
    "Entity",
    "EvalCase",
    "EvalReport",
    "ProgressEvent",
    "RetrieveRequest",
    "RetrieveResult",
    "WorkflowRetriever",
]

__version__ = "0.1.0"
