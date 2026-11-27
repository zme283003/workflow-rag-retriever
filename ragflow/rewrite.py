"""Query rewriting.

Two complementary strategies, both producing a list of rewritten queries that
are executed in parallel and then fused:

- MultiTurnRewriter: resolve pronouns using conversation history.
- SynonymRewriter: expand/standardize terms using a synonym table.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .ports import LLM

REWRITE_PROMPT = """Rewrite the user question so it is fully self-contained.
Use the conversation history to resolve pronouns and omitted subjects.
Return the rewritten question only, one line, no explanation.
History: {history}
Question: {query}"""


def _dedupe(queries: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


class MultiTurnRewriter:
    """Resolve the query against recent conversation turns via an LLM."""

    def __init__(self, llm: LLM, max_history_turns: int = 6):
        self.llm = llm
        self.max_history_turns = max_history_turns

    def rewrite(self, query: str, history: List[Dict[str, str]]) -> List[str]:
        if not history:
            return []
        recent = history[-self.max_history_turns :]
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent
        )
        try:
            raw = self.llm.complete(
                REWRITE_PROMPT.format(history=history_text, query=query), ""
            )
            return _dedupe([raw])
        except Exception:
            return []


class SynonymRewriter:
    """Rule-based synonym expansion; safe, deterministic, offline."""

    def __init__(self, synonyms: Optional[Dict[str, List[str]]] = None):
        # Demo vocabulary: generic technical terms, no business-specific data.
        self.synonyms = synonyms or {
            "maintenance": ["servicing", "upkeep"],
            "manual": ["handbook", "guide"],
        }

    def rewrite(self, query: str, max_count: int = 3) -> List[str]:
        results = [query]
        for term, variants in self.synonyms.items():
            if re.search(rf"\b{re.escape(term)}\b", query, flags=re.I):
                for variant in variants:
                    results.append(re.sub(rf"\b{re.escape(term)}\b", variant, query, flags=re.I))
            if len(results) >= max_count:
                break
        return _dedupe(results[:max_count])


class QueryRewriter:
    """Combine strategies and cap the total number of queries."""

    def __init__(
        self,
        multi_turn: Optional[MultiTurnRewriter] = None,
        synonym: Optional[SynonymRewriter] = None,
    ):
        self.multi_turn = multi_turn
        self.synonym = synonym

    def rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_count: int = 3,
    ) -> List[str]:
        candidates = [query]
        if self.multi_turn and history:
            candidates.extend(self.multi_turn.rewrite(query, history))
        if self.synonym:
            for base in list(candidates):
                candidates.extend(self.synonym.rewrite(base, max_count=max_count))
        return _dedupe(candidates)[:max_count]
