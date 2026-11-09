# -*- coding: utf-8 -*-
"""Structural comparison of original vs draft JSON trees."""
from llm_conductor.cli.draft_validate import (
    compare_json_structure,
    format_draft_structure_errors,
)


def test_matching_trees_produce_no_issues():
    original = {"a": {"b": "x"}, "c": [1, "two"]}
    draft = {"a": {"b": "икс"}, "c": [1, "два"]}
    assert compare_json_structure(original, draft) == []


def test_missing_extra_and_type_mismatch():
    original = {"keep": "a", "gone": "b", "num": 1}
    draft = {"keep": "а", "extra": "z", "num": "1"}
    issues = compare_json_structure(original, draft)
    assert any("gone: missing in draft" in i for i in issues)
    assert any("extra: extra in draft" in i for i in issues)
    assert any("num: type mismatch" in i for i in issues)


def test_key_reorder_preserves_mapping():
    original = {"z": "1", "a": "2"}
    draft = {"a": "два", "z": "один"}
    assert compare_json_structure(original, draft) == []


def test_array_length_mismatch():
    issues = compare_json_structure({"x": [1, 2]}, {"x": [1]})
    assert any("array length mismatch" in i for i in issues)


def test_format_includes_target_index():
    text = format_draft_structure_errors(2, ["root.a: missing in draft"])
    assert "target 2:" in text
    assert "root.a: missing in draft" in text
