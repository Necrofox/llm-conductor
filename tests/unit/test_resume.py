# -*- coding: utf-8 -*-
"""Unit tests for stage-local resume helpers (llm_conductor/core/resume.py)."""
import pytest

from llm_conductor.core.resume import (
    build_run_fingerprint,
    overlay_resume_values,
    source_fingerprint,
)


def test_overlay_resume_values_sets_report_key_paths():
    source = {"ui": {"title": "Hello", "other": "x"}, "items": ["a", "b"]}
    key = '["ui","title"]'
    out = overlay_resume_values(source, {key: "RU(Hello)"})
    assert out["ui"]["title"] == "RU(Hello)"
    assert out["ui"]["other"] == "x"
    assert source["ui"]["title"] == "Hello"


def test_overlay_handles_lists_literal_keys_and_root_strings():
    source = {'flat.key': ['a', {'title': 'b'}]}
    assert overlay_resume_values(source, {'["flat.key",1,"title"]': 'translated'}) == {
        'flat.key': ['a', {'title': 'translated'}],
    }
    assert overlay_resume_values('source', {'[]': 'translated'}) == 'translated'


@pytest.mark.parametrize('path', ['not json', '{}', '[]', '[true]', '[2]', '["missing"]', '[0,"x"]'])
def test_overlay_ignores_invalid_or_incompatible_paths(path):
    assert overlay_resume_values(['source'], {path: 'bad'}) == ['source']


def test_source_fingerprint_changes_with_leaf_text():
    a = source_fingerprint({"t": "one"})
    b = source_fingerprint({"t": "two"})
    assert a != b


@pytest.mark.parametrize(('field', 'changed'), [
    ('operation_type', 'verification'), ('model', 'other'),
    ('system_prompt', 'new system'), ('prompt_template', 'new prompt'),
    ('mode', 'strict'), ('sampling_identity', '{"temperature": 1}'),
    ('api_url', 'http://other'), ('allowed_scripts', ('cyrillic',)),
    ('content_identity', ('changed', 'b')), ('source_digest', 'changed'),
])
def test_run_fingerprint_changes_with_generation_or_source_identity(field, changed):
    common = {
        "operation_type": "translation",
        "model": "m",
        "system_prompt": "",
        "prompt_template": "Translate: ",
        "mode": "style",
        "sampling_identity": "{}",
        "api_url": "http://x",
        "allowed_scripts": (),
        "content_identity": ("a", "b"),
        "source_digest": "abc",
    }
    one = build_run_fingerprint(**common)
    two = build_run_fingerprint(**{**common, field: changed})
    assert one != two
