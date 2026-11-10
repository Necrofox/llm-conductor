# -*- coding: utf-8 -*-
"""Structural and marker comparison of original vs draft JSON trees."""
from llm_conductor.cli.draft_validate import (
    compare_draft_markers,
    compare_json_structure,
    format_draft_marker_errors,
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


def test_bool_int_type_mismatch():
    issues = compare_json_structure({"flag": True}, {"flag": 1})
    assert any("flag: type mismatch (original bool, draft int)" in i for i in issues)


def test_nested_missing_key_uses_full_path():
    issues = compare_json_structure({"a": {"b": "x"}}, {"a": {}})
    assert any("a.b: missing in draft" in i for i in issues)


def test_format_includes_target_index():
    text = format_draft_structure_errors(2, ["root.a: missing in draft"])
    assert "target 2:" in text
    assert "root.a: missing in draft" in text


def test_draft_markers_match_when_placeholders_preserved():
    original = {"line": "Open {0}.", "tag": "<b>Hi</b>"}
    draft = {"line": "Открой {0}.", "tag": "<b>Привет</b>"}
    assert compare_draft_markers(original, draft) == []


def test_draft_markers_detect_lost_placeholder():
    issues = compare_draft_markers({"line": "Open {0}."}, {"line": "Открой."})
    assert len(issues) == 1
    assert "marker mismatch" in issues[0]
    assert "missing in draft" in issues[0]
    assert "{0}" in issues[0]


def test_format_draft_marker_errors_includes_target_index():
    text = format_draft_marker_errors(3, ["line: marker mismatch (...)"])
    assert "target 3:" in text
    assert "draft markers do not match original" in text
    assert "line: marker mismatch" in text
