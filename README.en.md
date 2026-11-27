# workflow-rag-retriever

A retrieval workflow for document Q&A where the question names a specific
identifier — an equipment tag, a project number, a drawing number.

The code here is a clean re-implementation. All data is synthetic (made-up IDs
like `EQ-1001`). Nothing here comes from a real customer, a real identifier, an
internal network or a credentials file.

[中文文档](README.md)

---

## Why this exists

The first version was plain RAG: embed the question, search the whole corpus.
Fine for general questions, not so fine for specialist documents.

Take a concrete case. Someone asks "what is the maintenance interval for
EQ-1001". Whole-corpus vector search returns every document mentioning EQ-1001:
the maintenance manual, inspection logs, purchase orders, a retirement form.
Semantically they all match. The answer lives in one of them. Recall looks
decent, the answer doesn't.

So we added metadata. Extract the identifier from the question, look up which
documents it points to, and search only those. In the example above the search
space drops from the whole corpus to two or three documents.

That still wasn't enough. Some answers sit in one section of a long document,
and chunking cuts the context. So we added PageIndex: build a section tree per
document, strip the body text, leave titles and summaries, and let the model
pick which sections to read. Page numbers come back with the hit.

With those two changes, usable answers on the same question set went from
60% to 95%.

---

## Results

Same question set, before and after:

| | Vector only | This workflow |
|---|---|---|
| **Usable answer** | 60% | **95%** |
| Document recall | — | 0.90 |
| At least one expected doc found | — | 97.3% |

"Usable" means fully correct plus partially correct. If you demand every number
be exact, the figure is lower — but in practice an answer that is right apart
from one figure still saves a human most of the work.

**What changed is the floor.** At 60%, four questions in ten need a human to
pick up, which is not something you can ship. At 95%, it is an occasional
correction. Those points did not come from tuning a threshold. They came from
narrowing the search space with metadata and restoring structure with the
section tree.

### About the evaluation dump

That export was a partial sample — a slice of the questions, with incomplete
fields. Reading a system's capability off any single row of it would be
misleading.

For instance, image questions can look like "the image was not returned" in the
sheet. In the running platform the image comes back as a link that the front
end renders directly, so the image is correct. Tables are the same: they render
properly in the platform. The blank cells in the sheet are an artifact of how
it was recorded, not a retrieval or rendering failure.

The metrics in `evaluation.py` (`perfect_recall_correct_rate`,
`recall_one_but_not_correct`) are not there to find fault with this workflow.
They separate "the document was found" from "the answer was right" — retrieval
is only one link in the chain, while generation and rendering live on the
platform side. Keeping the two apart tells you where to spend effort.

---

## The flow

```
                          question
                              |
                   [1] entity extraction
            schema few-shot + regex/normalizer
                              |
                   [2] metadata lookup
             entity -> document IDs + TOC paths
                              |
       +----------------------+-------------------+
       |                                          |
  [3] summary fallback                    (documents found)
   when metadata finds nothing                    |
       +----------------------+-------------------+
                              |
       [4] parallel retrieval
        full-text vector  ||  PageIndex tree
    (rewritten queries as extra queries)
                              |
             [5] RRF fusion -> de-dup -> top_k
                              |
                   [6] response + progress
```

Each step is a protocol in `ragflow/ports.py`. `ragflow/workflow.py` only knows
those protocols, not any concrete service. Swap the vector store, the metadata
API or the model by swapping the implementation; the orchestration stays.

---

## What's in each file

| File | Role |
|---|---|
| `ragflow/entity.py` | Entity extraction: schema few-shot, time rules, regex fallback, alias/separator normalization |
| `ragflow/metadata.py` | Entity to documents — the narrowing step |
| `ragflow/pageindex.py` | Section-tree retrieval, LLM node selection, lexical fallback |
| `ragflow/rewrite.py` | Multi-turn and synonym rewriting |
| `ragflow/fusion.py` | Score normalization, RRF, de-duplication |
| `ragflow/workflow.py` | The six-stage orchestration |
| `ragflow/evaluation.py` | Recall, hit rate, full-hit rate, recall-perfect-correct rate |
| `ragflow/progress.py` | Progress events that never break retrieval |
| `ragflow/backends/demo.py` | Offline synthetic backend |

---

## Run it

No third-party packages needed.

```bash
python examples/demo.py       # end-to-end retrieval
python examples/evaluate.py   # evaluation example
python run_tests.py           # 21 tests
```

Output from `examples/demo.py`:

```
Q: What is the maintenance interval for EQ-1001?
entities: ['EQ-1001']
documents: ['DOC-A', 'DOC-B']
  [vector   ] 0.049  The maintenance interval for EQ-1001 is 180 days...
  [pageindex] 0.032  Maintenance interval: Scheduled maintenance cycle is 180 days...
```

With pytest installed, `pytest tests` works too — the tests are plain functions.

---

## Bring your own services

Implement the matching protocol from `ragflow/ports.py`:

```python
from ragflow import WorkflowRetriever
from ragflow.entity import SchemaItem

retriever = WorkflowRetriever.from_llm(
    llm=my_llm,                        # implements complete(system, user)
    vector_retriever=my_vector_store,  # implements search(...)
    metadata_provider=my_metadata,     # implements lookup(entities)
    tree_store=my_tree_store,          # implements load(toc_path)
    schema=[SchemaItem("asset", "EQ-1001, EQ-2002")],
    aliases={"EQ 1001": "EQ-1001"},
)
result = retriever.retrieve(RetrieveRequest(query="..."))
```

No LLM configured? It falls back to rule-based extraction, which is enough to
check the pipeline end to end.

---

## Trade-offs

Vector scores and tree-search scores are not on the same scale. Averaging them
is a gamble. RRF fuses by rank and avoids calibrating across channels.

When metadata finds nothing, a summary search recovers candidates instead of
returning empty. Better to read a few extra documents than to drop a question
because one identifier did not match.

Normalize before lookup. `EQ 1001`, `eq-1001` and `EQ—1001` all need to land on
one canonical ID, or metadata lookup will miss.

When a parent and a child node both match, the child ranks first. The parent is
usually a heading; the child holds the actual number.

The core has no third-party dependencies. Infrastructure sits behind
protocols, so replacing it does not touch the algorithm.

---

## License

MIT — see [LICENSE](LICENSE).
