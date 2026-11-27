# workflow-rag-retriever

**Entity-constrained, multi-stage RAG retrieval — a clean reference implementation.**

A retrieval workflow for document-heavy domains where questions name specific
business entities. Instead of guessing relevant documents by semantic similarity
over the whole corpus, it resolves the entity to a narrow document set first,
then combines vector search with a structure-tree ("PageIndex") search, fuses
the results, and reports honest evaluation metrics.

> **Fully de-identified.** All data in this repository is synthetic (generic
> equipment IDs such as `EQ-1001`). No real customer names, identifiers,
> endpoints, credentials or proprietary data are included. This is a clean
> re-implementation of a production design, written for learning and reuse.

---

## Why this design

Naive RAG embeds a question and searches the entire corpus. In specialist
domains that fails in two ways:

1. **Low precision** — semantically similar but unrelated documents crowd out
   the right one.
2. **No structure awareness** — chunking loses the document's section layout,
   so answers that live in a specific section are missed.

This project addresses both:

- **Entity constraint (metadata narrowing):** extract the entity from the
  question, look it up in metadata, and restrict retrieval to the documents
  that entity points to. This turns "search everything" into "search these few
  documents".
- **PageIndex tree search:** load a document's section tree (titles +
  summaries, text stripped) and let an LLM navigate to the relevant nodes,
  recovering structure-aware passages with page/line locations.
- **Honest evaluation:** recall alone overstates quality. We also report how
  often full recall still produced a wrong answer.

## Architecture

```
                         user question
                               |
                  [1] entity extraction
             schema few-shot LLM + regex/normalizer
                               |
                  [2] metadata lookup
             entity -> document IDs + TOC paths
                               |
        +----------------------+----------------------+
        |                                             |
 [3] summary fallback                          (documents found)
  (only when metadata                                  |
   found nothing)                                      |
        +----------------------+----------------------+
                               |
        [4] parallel retrieval
         full-text vector  ||  PageIndex tree search
        (+ rewritten queries as extra vector queries)
                               |
              [5] RRF fusion -> de-dup -> top_k
                               |
                     [6] response + progress
```

Each box is a small protocol in `ragflow/ports.py`; the orchestration in
`ragflow/workflow.py` depends only on those interfaces. Swap in your own
vector store / metadata service / LLM without changing the workflow.

## Modules

| Module | Responsibility |
|---|---|
| `ragflow/entity.py` | Schema-driven few-shot entity extraction, normalization, time rules + regex fallback |
| `ragflow/metadata.py` | Resolve entities to documents (metadata narrowing) |
| `ragflow/pageindex.py` | Document-tree navigation, LLM node selection, lexical fallback |
| `ragflow/rewrite.py` | Multi-turn and synonym query rewriting |
| `ragflow/fusion.py` | Score normalization, Reciprocal Rank Fusion, de-duplication |
| `ragflow/workflow.py` | The six-stage orchestration |
| `ragflow/evaluation.py` | Recall, hit rate, full-hit rate, recall-perfect-correct rate |
| `ragflow/progress.py` | Fire-and-forget progress events |
| `ragflow/backends/demo.py` | Synthetic, offline, zero-dependency backend |

## Quick start

No external services or third-party packages required.

```bash
python examples/demo.py       # run the workflow end to end
python examples/evaluate.py   # run the evaluation demo
python run_tests.py           # 21 tests, zero dependencies
```

Example output:

```
Q: What is the maintenance interval for EQ-1001?
entities: ['EQ-1001']
documents: ['DOC-A', 'DOC-B']
  [vector   ] 0.049  The maintenance interval for EQ-1001 is 180 days...
  [pageindex] 0.032  Maintenance interval: Scheduled maintenance cycle is 180 days...
```

## Wire in your own services

All integrations implement a protocol from `ragflow/ports.py`:

```python
from ragflow import WorkflowRetriever
from ragflow.entity import SchemaItem, ExampleData
from ragflow.workflow import WorkflowRetriever  # or .from_llm(...)

retriever = WorkflowRetriever.from_llm(
    llm=my_llm,                        # implements .complete(system, user)
    vector_retriever=my_vector_store,  # implements .search(...)
    metadata_provider=my_metadata,     # implements .lookup(entities)
    tree_store=my_tree_store,          # implements .load(toc_path)
    schema=[SchemaItem("asset", "EQ-1001, EQ-2002")],
    aliases={"EQ 1001": "EQ-1001"},
)
result = retriever.retrieve(RetrieveRequest(query="..."))
```

`WorkflowRetriever` falls back to a rule-based extractor when no LLM is
configured, so the pipeline still works offline.

## Evaluation: the key lesson

A real evaluation of this design (300 labeled questions) produced:

- mean recall **0.90**, ≥1 expected document found **97%** of the time;
- but only **54%** of answers judged fully correct;
- **29 cases had recall = 1 yet a wrong or partial answer**.

Conclusion: high recall does not imply a correct answer — retrieval can find
the right document while extraction/generation still fails. `evaluation.py`
therefore reports `perfect_recall_correct_rate` and counts
`recall_one_but_not_correct` explicitly, so a system cannot hide generation
errors behind a good recall number.

## Design choices

- **RRF over score averaging:** vector scores and tree-search scores are not
  comparable. Reciprocal Rank Fusion merges ranked lists by rank, avoiding
  fragile cross-channel score calibration.
- **Summary fallback:** when metadata finds nothing, a summary-level search
  recovers the candidate document set instead of returning nothing.
- **Normalization before lookup:** aliases, separators and case are normalized
  so a surface form like `eq 1001` still resolves to `EQ-1001`.
- **Hierarchy-first tree selection:** specific child nodes outrank generic
  parents when both match.
- **Zero-dependency core:** the algorithm is the point; infrastructure is
  behind protocols.

## License

MIT — see [LICENSE](LICENSE).
