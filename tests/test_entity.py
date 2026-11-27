from ragflow.entity import (
    ExampleData,
    Normalizer,
    RuleEntityExtractor,
    SchemaEntityExtractor,
    SchemaItem,
    normalize_time_entities,
    regex_time_fallback,
)
from ragflow.models import Entity


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete(self, system, user, temperature=0.0):
        return self.response


SCHEMA = [SchemaItem("asset", "EQ-1001, EQ-2002"), SchemaItem("date", "2024-05-12")]


def test_schema_extractor_parses_and_canonicalizes():
    llm = FakeLLM('[{"extraction_class":"asset","extraction_text":"eq 1001"}]')
    normalizer = Normalizer(separator="-")
    ex = SchemaEntityExtractor(llm, SCHEMA, normalizer=normalizer)
    entities = ex.extract("maintenance of eq 1001")
    assert any(e.text == "eq-1001" for e in entities)


def test_extractor_falls_back_on_bad_json():
    ex = SchemaEntityExtractor(FakeLLM("not json"), SCHEMA)
    assert ex.extract("nothing here") == []


def test_regex_time_fallback():
    out = regex_time_fallback("report from 2024-05-12 to 2024-06-01", [])
    assert len(out) == 2
    assert all(e.type == "date" for e in out)


def test_normalize_time_splits_range():
    entities = [Entity(raw="2024-05-12 到 2024-06-01", text="2024-05-12 到 2024-06-01", type="time")]
    out = normalize_time_entities(entities)
    assert len(out) == 2


def test_normalizer_aliases_and_separator():
    n = Normalizer(aliases={"EQ 1001": "EQ-1001"})
    assert n.normalize("eq 1001") == "EQ-1001"
    assert n.normalize("EQ 2002") == "EQ-2002"


def test_rule_extractor_offline():
    ex = RuleEntityExtractor()
    entities = ex.extract("Check EQ-1001 and EQ-2002 please")
    assert {e.text for e in entities} == {"EQ-1001", "EQ-2002"}
