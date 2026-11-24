# -*- coding: utf-8 -*-
"""Unit tests for stage-local resume helpers (llm_conductor/core/resume.py)."""
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


def test_source_fingerprint_changes_with_leaf_text():
    a = source_fingerprint({"t": "one"})
    b = source_fingerprint({"t": "two"})
    assert a != b


def test_run_fingerprint_changes_with_prompt_template():
    common = {
        "operation_type": "translation",
        "model": "m",
        "system_prompt": "",
        "mode": "style",
        "sampling_identity": "{}",
        "api_url": "http://x",
        "allowed_scripts": (),
        "content_identity": ("a", "b"),
        "source_digest": "abc",
    }
    one = build_run_fingerprint(prompt_template="Translate: ", **common)
    two = build_run_fingerprint(prompt_template="Translate differently: ", **common)
    assert one != two
