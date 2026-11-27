"""Schema-driven entity extraction and normalization.

This module is modelled on a production "element extraction" service. The
design has four layers, each independently useful:

1. ``SchemaItem`` / ``EntitySchema``: declarative, domain-configurable entity
   definitions with example patterns (few-shot).
2. ``SchemaEntityExtractor``: ask an LLM to extract entities that match the
   schema, returning exact text spans with their class.
3. Rule-based post-processing: normalize time entities and a regex fallback
   that recovers time expressions the LLM missed.
4. ``Normalizer``: canonicalize extracted surface forms (aliases, separators,
   case) so downstream metadata lookup is stable.

Everything here is dependency-free and uses only synthetic example data. A
production deployment supplies its own schema, alias map and LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .models import Entity
from .ports import EntityExtractor, LLM

# ---------------------------------------------------------------------------
# 1. Schema definitions
# ---------------------------------------------------------------------------


@dataclass
class SchemaItem:
    """One entity type to extract, with example surface forms for few-shot."""

    entity_type: str
    examples: str = ""


@dataclass
class ExampleData:
    """A few-shot example: input text and the entities it contains."""

    text: str
    extractions: List[tuple] = field(default_factory=list)  # (class, text)


ENTITY_EXTRACT_PROMPT = """You are an information extraction system. Extract all entities that match the schema.

## Entity schema
{schema}

{few_shot}
## Extraction rules
- Read the text left to right; extract in order of appearance.
- Use EXACT text spans from the input; never paraphrase or invent.
- Each span appears once; no overlapping entities.
- If a span fits multiple types, pick the most specific one.
- Do not extract vague time words (e.g. "recently", "early this year"); only concrete dates/times.

Return STRICT JSON only:
[{{"extraction_class":"type","extraction_text":"exact span"}}]
If none, return [].

Input text: {query}
Extracted entities (JSON):"""


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str):
    """Parse a JSON array/object from LLM output, tolerating fences/prose."""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else parts[0]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    match = re.search(r"[\[{][\s\S]*[\]}]", cleaned)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _format_schema(items: Sequence[SchemaItem]) -> str:
    lines = []
    for item in items:
        example = f" (e.g. {item.examples})" if item.examples else ""
        lines.append(f"- {item.entity_type}{example}")
    return "\n".join(lines)


def _format_examples(examples: Sequence[ExampleData]) -> str:
    if not examples:
        return ""
    blocks = ["## Few-shot examples"]
    for ex in examples:
        pairs = [{"extraction_class": c, "extraction_text": t} for c, t in ex.extractions]
        blocks.append(f"Input: {ex.text}\nExtracted: {json.dumps(pairs, ensure_ascii=False)}")
    return "\n".join(blocks) + "\n"


def _dedupe(entities: Iterable[Entity]) -> List[Entity]:
    seen = set()
    out: List[Entity] = []
    for e in entities:
        key = (e.type, e.text)
        if not e.text or key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------

TIME_PATTERNS = [
    (r"\d{4}-\d{1,2}-\d{1,2}", 12),
    (r"\d{4}年\d{1,2}月\d{1,2}日", 12),
    (r"\d{4}年\d{1,2}月", 8),
    (r"\d{4}年第[一二三四1-4]季度", 10),
    (r"\d{4}年", 5),
    (r"\d{1,2}月\d{1,2}日", 7),
]


class Normalizer:
    """Canonicalize extracted entity surface forms.

    ``aliases`` maps an uppercased surface form to its canonical id.
    ``separator`` normalizes common separators (spaces, underscores, Chinese
    dashes) to a single character, and ``strip_suffixes`` removes trailing
    words such as "report".
    """

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        separator: str = "-",
        strip_suffixes: Sequence[str] = (),
    ):
        self.aliases = {k.upper(): v for k, v in (aliases or {}).items()}
        self.separator = separator
        self.strip_suffixes = tuple(strip_suffixes)

    def normalize(self, text: str) -> str:
        value = text.strip()
        # Check the raw surface form against aliases first (e.g. "eq 1001").
        alias = self.aliases.get(value.upper())
        if alias is not None:
            return alias
        for suffix in self.strip_suffixes:
            if value.lower().endswith(suffix.lower()):
                value = value[: -len(suffix)].strip()
        value = re.sub(r"[\s_—–]+", self.separator, value)
        value = re.sub(rf"{re.escape(self.separator)}+", self.separator, value)
        return self.aliases.get(value.upper(), value)


def normalize_time_entities(entities: List[Entity]) -> List[Entity]:
    """Strip noise from time entities; split 'A to B' into two entities."""
    out: List[Entity] = []
    for e in entities:
        if e.type not in ("time", "date"):
            out.append(e)
            continue
        value = e.raw.replace(" ", "").replace("份", "").replace("度", "")
        parts = re.split(r"至|到", value)
        for part in parts:
            if part.strip():
                out.append(Entity(raw=part.strip(), text=part.strip(), type="date"))
    return out


def regex_time_fallback(query: str, entities: List[Entity]) -> List[Entity]:
    """Recover concrete time expressions the LLM missed."""
    existing = {e.text.replace(" ", "") for e in entities if e.type in ("time", "date")}
    spans: List[tuple] = []
    for pattern, _ in TIME_PATTERNS:
        for m in re.finditer(pattern, query):
            spans.append((m.start(), m.end(), m.group()))
    spans.sort(key=lambda x: x[0])
    kept: List[tuple] = []
    for start, end, text in spans:
        overlap = False
        for i, (s, en, _) in enumerate(kept):
            if start < en and end > s:
                if (end - start) > (en - s):
                    kept[i] = (start, end, text)
                overlap = True
                break
        if not overlap:
            kept.append((start, end, text))
    extra = []
    for _, _, text in kept:
        value = text.replace(" ", "")
        if value not in existing:
            extra.append(Entity(raw=value, text=value, type="date"))
            existing.add(value)
    return entities + extra


# ---------------------------------------------------------------------------
# 4. Extractors
# ---------------------------------------------------------------------------


class SchemaEntityExtractor:
    """Primary extractor: LLM + configurable schema + post-processing."""

    def __init__(
        self,
        llm: LLM,
        schema: Sequence[SchemaItem],
        examples: Sequence[ExampleData] = (),
        normalizer: Optional[Normalizer] = None,
        normalize_types: Sequence[str] = ("asset", "well"),
    ):
        self.llm = llm
        self.schema = list(schema)
        self.examples = list(examples)
        self.normalizer = normalizer
        self.normalize_types = set(normalize_types)

    def extract(self, query: str) -> List[Entity]:
        prompt = ENTITY_EXTRACT_PROMPT.format(
            schema=_format_schema(self.schema),
            few_shot=_format_examples(self.examples),
            query=query,
        )
        try:
            raw = self.llm.complete(prompt, "")
            data = _parse_json(raw)
            items = data if isinstance(data, list) else data.get("entities", [])
            entities: List[Entity] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("extraction_text", "")).strip()
                if not text:
                    continue
                etype = str(item.get("extraction_class", "entity"))
                canonical = text
                if self.normalizer and etype in self.normalize_types:
                    canonical = self.normalizer.normalize(text)
                entities.append(Entity(raw=text, text=canonical, type=etype))
        except Exception:
            entities = []

        entities = normalize_time_entities(entities)
        entities = regex_time_fallback(query, entities)
        return _dedupe(entities)


class RuleEntityExtractor:
    """Offline fallback: regex over configured ID patterns."""

    def __init__(
        self,
        patterns: Optional[Sequence[str]] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        self.patterns = [
            re.compile(p) for p in (patterns or [r"\b[A-Z]{1,4}-?\d+[A-Z0-9-]*\b"])
        ]
        self.normalizer = normalizer

    def extract(self, query: str) -> List[Entity]:
        found: List[str] = []
        for pattern in self.patterns:
            found.extend(m.group(0) for m in pattern.finditer(query))
        entities = []
        for text in found:
            canonical = self.normalizer.normalize(text) if self.normalizer else text
            entities.append(Entity(raw=text, text=canonical))
        entities = regex_time_fallback(query, entities)
        return _dedupe(entities)


# Backward-compatible alias used by older examples.
Standardizer = Normalizer
