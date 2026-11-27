"""PageIndex: document-tree retrieval.

A document's structure tree is a hierarchy of nodes, each with a title and a
summary. Instead of chunking blindly, we let an LLM act as a "document
navigator": it sees the tree with full text stripped (to save tokens) and
returns the node IDs most relevant to the question. Those nodes become high
precision retrieval hits with page/line locations.

Fallback: when no LLM is available, a lexical score over title+summary keeps
the module usable offline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .models import Chunk, Source
from .ports import LLM, TreeStore

NAVIGATE_PROMPT = """You navigate a document structure tree to find nodes that answer a question.
Each node has node_id, title and summary (full text removed).
Rules:
- Pick only nodes directly relevant to the question; prefer specific child nodes over vague parents.
- For questions about numbers, prefer nodes whose summary mentions the relevant metric.
- At most 5 nodes. If the document has no answer, return an empty list.
Return STRICT JSON only: {"node_list":["node_id",...]}
Question: {question}
Tree: {tree}"""


def strip_text(tree: Any, fields=("text", "content")) -> Any:
    """Recursively remove heavy text fields before sending a tree to the LLM."""
    if isinstance(tree, dict):
        return {k: strip_text(v, fields) for k, v in tree.items() if k not in fields}
    if isinstance(tree, list):
        return [strip_text(item, fields) for item in tree]
    return tree


def iter_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a tree into a list of node dicts (handles 'structure' wrapper)."""
    root = tree.get("structure", tree) if isinstance(tree, dict) else tree
    nodes: List[Dict[str, Any]] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if "node_id" in item:
                nodes.append(item)
            for child in item.get("nodes", []) or []:
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(root)
    return nodes


class PageIndexRetriever:
    def __init__(self, tree_store: TreeStore, llm: Optional[LLM] = None, max_nodes: int = 5):
        self.tree_store = tree_store
        self.llm = llm
        self.max_nodes = max_nodes

    def select_node_ids(self, tree: Dict[str, Any], query: str) -> List[str]:
        nodes = iter_nodes(tree)
        if not nodes:
            return []
        if self.llm is not None:
            slim = strip_text(tree.get("structure", tree) if isinstance(tree, dict) else tree)
            prompt = NAVIGATE_PROMPT.format(
                question=query,
                tree=json.dumps(slim, ensure_ascii=False),
            )
            try:
                raw = self.llm.complete(prompt, "")
                match = re.search(r"\{[\s\S]*\}", raw)
                data = json.loads(match.group(0) if match else raw)
                ids = [str(x) for x in data.get("node_list", [])]
                valid = {n["node_id"] for n in nodes}
                return [i for i in ids if i in valid][: self.max_nodes]
            except Exception:
                pass  # fall through to lexical fallback
        return self._lexical_select(nodes, query)

    def _lexical_select(self, nodes: List[Dict[str, Any]], query: str) -> List[str]:
        tokens = set(re.findall(r"\w+", query.lower()))
        # Depth heuristic: prefer specific child nodes over generic parents when
        # relevance ties, matching the LLM navigator's "层级优先" rule.
        # A dotted id ("n1.1") indicates a deeper, more specific node.
        scored = []
        for node in nodes:
            text = f"{node.get('title', '')} {node.get('summary', '')}".lower()
            score = sum(1 for t in tokens if t and t in text)
            if score:
                depth = str(node["node_id"]).count(".")
                scored.append((score, depth, node["node_id"]))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [nid for _, _, nid in scored[: self.max_nodes]]

    def retrieve(
        self,
        toc_paths: List[str],
        query: str,
        doc_names: Optional[Dict[str, str]] = None,
    ) -> List[Chunk]:
        """Retrieve nodes across documents and return them as Chunks."""
        doc_names = doc_names or {}
        chunks: List[Chunk] = []
        for toc_path in toc_paths:
            tree = self.tree_store.load(toc_path)
            if not tree:
                continue
            node_map = {n["node_id"]: n for n in iter_nodes(tree)}
            for node_id in self.select_node_ids(tree, query):
                node = node_map.get(node_id)
                if not node:
                    continue
                doc_name = doc_names.get(toc_path, toc_path)
                location = node.get("start_index") or node.get("line_num") or ""
                text = f"{node.get('title', '')}: {node.get('summary', '')}".strip()
                chunks.append(
                    Chunk(
                        id=f"pg::{doc_name}::{node_id}",
                        text=text,
                        score=1.0,  # tree hits are treated as high-precision signals
                        document_id=doc_name,
                        document_name=doc_name,
                        source=Source.PAGEINDEX,
                        extra={"node_id": node_id, "location": location, "toc_path": toc_path},
                    )
                )
        return chunks
