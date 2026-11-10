# -*- coding: utf-8 -*-
"""Unit tests for dry-run helpers (duplicate-key scan, table formatting)."""
from llm_conductor.cli.dry_run import (
    count_duplicate_json_keys,
    format_effective_table,
)
from llm_conductor.cli.glossary import GlossarySource
from llm_conductor.cli.runner import RunOptions
from llm_conductor.cli.targets import ResultLayout, RunTarget, TargetEntry
from llm_conductor.constants import RUN_STAGE_ALL, RUN_STAGE_TRANSLATE


def test_count_duplicate_json_keys_none():
    assert count_duplicate_json_keys('{"a": 1, "b": 2}') == 0


def test_count_duplicate_json_keys_top_level():
    # Last-wins: {"a": 1, "a": 2} → one duplicate occurrence.
    assert count_duplicate_json_keys('{"a": 1, "a": 2}') == 1


def test_count_duplicate_json_keys_nested():
    text = '{"outer": {"x": 1, "x": 2}, "outer2": {"y": 1}}'
    assert count_duplicate_json_keys(text) == 1


def test_format_effective_table_includes_columns_and_paths(tmp_path):
    original = str(tmp_path / "a.json")
    entry = TargetEntry(
        source_path=original,
        relative_path="a.json",
        index=1,
        stage=RUN_STAGE_TRANSLATE,
        strict=True,
        llm_config_path=str(tmp_path / "llm.toml"),
        glossary_path=None,
    )
    target = RunTarget(path=str(tmp_path / "run.toml"), entries=(entry,))
    layout = ResultLayout(str(tmp_path / "result" / "run.toml"), shared_by_name=True)
    llm_path = str(tmp_path / "llm.toml")
    # profile_paths values are unused when entry.llm_config_path is set.
    table = format_effective_table(
        target,
        layout,
        profile_paths={llm_path: object()},
        glossaries={None: GlossarySource()},
        default_options=RunOptions(stage=RUN_STAGE_ALL),
        bootstrap_llm=llm_path,
    )
    assert "index" in table
    assert "original" in table
    assert "result_paths" in table
    assert original in table
    assert "translate" in table
    assert "true" in table
