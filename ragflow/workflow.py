"""The multi-stage retrieval workflow.

This is the heart of the project. Stages:

1. entity extraction (with a hint override),
2. metadata lookup to constrain the document set,
3. summary-vector fallback when metadata misses,
4. parallel full-text vector search + PageIndex tree search,
5. query rewriting executed as additional vector queries,
6. RRF fusion, de-duplication, top-k truncation.

Every external dependency is a port, so the workflow is unit-testable offline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .entity import Normalizer, RuleEntityExtractor, SchemaEntityExtractor, SchemaItem, ExampleData
from .fusion import dedupe, rrf_fuse
from .metadata import InMemoryMetadataProvider
from .models import Chunk, DocumentRef, Entity, ProgressEvent, RetrieveRequest, RetrieveResult, Source
from .pageindex import PageIndexRetriever
from .ports import (
    EntityExtractor,
    LLM,
    MetadataProvider,
    ProgressSink,
    TreeStore,
    VectorRetriever,
)
from .progress import NullProgressSink
from .rewrite import QueryRewriter, SynonymRewriter


class WorkflowRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        metadata_provider: MetadataProvider,
        entity_extractor: Optional[EntityExtractor] = None,
        tree_store: Optional[TreeStore] = None,
        page_index: Optional[PageIndexRetriever] = None,
        rewriter: Optional[QueryRewriter] = None,
        progress: Optional[ProgressSink] = None,
        top_k: int = 8,
    ):
        self.vector = vector_retriever
        self.metadata = metadata_provider
        self.entity_extractor = entity_extractor or RuleEntityExtractor()
        self.tree_store = tree_store
        self.page_index = page_index
        self.rewriter = rewriter or QueryRewriter(synonym=SynonymRewriter())
        self.progress = progress or NullProgressSink()
        self.top_k = top_k

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        vector_retriever: VectorRetriever,
        metadata_provider: MetadataProvider,
        tree_store: Optional[TreeStore] = None,
        progress: Optional[ProgressSink] = None,
        schema: Optional[List[SchemaItem]] = None,
        examples: Optional[List[ExampleData]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> "WorkflowRetriever":
        """Convenience constructor wiring the schema-driven extractor and navigator.

        The extractor follows the production design: schema-driven few-shot
        extraction, then canonicalization via ``Normalizer`` (aliases and
        separator/case rules), with deterministic rule extraction as fallback
        when no schema is provided.
        """
        normalizer = Normalizer(aliases)
        if schema:
            extractor: object = SchemaEntityExtractor(
                llm, schema, examples or (), normalizer
            )
        else:
            extractor = RuleEntityExtractor(normalizer=normalizer)
        return cls(
            vector_retriever=vector_retriever,
            metadata_provider=metadata_provider,
            entity_extractor=extractor,  # type: ignore[arg-type]
            tree_store=tree_store,
            page_index=PageIndexRetriever(tree_store, llm) if tree_store else None,
            rewriter=QueryRewriter(synonym=SynonymRewriter()),
            progress=progress,
        )

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        events: List[ProgressEvent] = []

        def emit(message: str) -> None:
            events.append(ProgressEvent(key=request.extra.get("key", ""), message=message))
            self.progress.report(request.extra.get("key", ""), message)

        # 1. Entity extraction
        emit("start entity extraction")
        if request.entity_text:
            entities = [Entity(raw=request.entity_text, text=request.entity_text)]
        else:
            entities = self.entity_extractor.extract(request.query)
        emit(
            "entities: " + (", ".join(e.text for e in entities) if entities else "none")
        )

        # 2. Metadata constraint
        emit("metadata lookup")
        documents: List[DocumentRef] = self.metadata.lookup(entities)
        doc_ids = [d.doc_id for d in documents]
        toc_paths = [d.toc_path for d in documents if d.toc_path]
        emit(f"metadata resolved {len(documents)} documents")

        # 3. Summary fallback when metadata produced nothing
        used_fallback = False
        if not documents:
            emit("summary fallback")
            used_fallback = True
            summaries = self.vector.search(
                request.query, [], scope="summary", top_k=5
            )
            for s in summaries:
                ref = DocumentRef(
                    doc_id=s.document_id,
                    doc_name=s.document_name,
                    toc_path=s.extra.get("toc_path", ""),
                )
                if ref.doc_id not in doc_ids:
                    documents.append(ref)
                    doc_ids.append(ref.doc_id)
                    if ref.toc_path and ref.toc_path not in toc_paths:
                        toc_paths.append(ref.toc_path)

        # 4. Parallel full-text vector search + PageIndex tree search
        emit("parallel retrieval: vector + pageindex")
        queries = self.rewriter.rewrite(request.query, request.extra.get("history"))
        vector_lists: List[List[Chunk]] = []

        def search_one(q: str) -> List[Chunk]:
            return self.vector.search(q, doc_ids, scope="full_text", top_k=self.top_k)

        if len(queries) <= 1:
            vector_lists.append(search_one(queries[0] if queries else request.query))
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
                vector_lists.extend(pool.map(search_one, queries))

        pageindex_chunks: List[Chunk] = []
        if self.page_index and toc_paths:
            doc_names = {d.toc_path or "": d.doc_name for d in documents if d.toc_path}
            pageindex_chunks = self.page_index.retrieve(toc_paths, request.query, doc_names)
        emit(
            f"vector hits: {sum(len(x) for x in vector_lists)}, "
            f"pageindex hits: {len(pageindex_chunks)}"
        )

        # 5. Fusion & de-duplication
        emit("fusion and de-duplication")
        all_lists = vector_lists + ([pageindex_chunks] if pageindex_chunks else [])
        fused = dedupe(rrf_fuse(all_lists))
        final = fused[: request.top_k or self.top_k]

        result = RetrieveResult(
            query=request.query,
            chunks=final,
            entities=entities,
            documents=documents,
            used_fallback=used_fallback,
            progress=events,
        )
        emit(f"done: {len(final)} chunks")
        return result
