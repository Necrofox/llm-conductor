# -*- coding: utf-8 -*-
"""
Unit tests for the TOML run-manifest parser against the locked contract
(GH-13 / GH-14): happy path, conflicts, and ambiguous cases.
"""
import os

import pytest

from llm_conductor.cli.run_manifest import (
    RunManifestParser,
    is_legacy_list_path,
    is_run_manifest_path,
)
from llm_conductor.core.io import FileOperations
from llm_conductor.exceptions import TargetError
from tests.support import RecordingLogger


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return str(root)


@pytest.fixture
def conductor_dir(tmp_path):
    d = tmp_path / "app"
    d.mkdir()
    return str(d)


@pytest.fixture
def parser_deps():
    logger = RecordingLogger()
    return FileOperations(logger=logger), logger


def _parse(base, conductor_dir, text, file_ops, logger, name="run.toml"):
    path = os.path.join(base, name)
    return RunManifestParser(path, conductor_dir, file_ops, logger).parse_text(text)


def _raises(base, conductor_dir, text, file_ops, logger, match):
    with pytest.raises(TargetError, match=match):
        _parse(base, conductor_dir, text, file_ops, logger)


# ============================================================================
# Happy path / effective settings
# ============================================================================

def test_basic_targets_resolve_relative_to_manifest(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
[[targets]]
original = "menu/main.json"

[[targets]]
original = "hud.json"
""", file_ops, logger)

    assert len(settings.targets) == 2
    assert settings.targets[0].relative_path == os.path.join("menu", "main.json")
    assert settings.targets[0].index == 1
    assert settings.targets[1].original == os.path.join(base, "hud.json")
    assert settings.targets[0].stage == "all"
    assert settings.targets[0].strict is False
    assert settings.targets[0].llm_config == os.path.join(conductor_dir, "llm.toml")
    assert settings.result2target is False
    assert settings.cache2target is False
    assert settings.verb is None


def test_run_and_target_overrides(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
verb = "DEBUG"
result2target = true
cache2target = true
stage = "translate"
strict = true
llm_config = "profiles/base.toml"
glossary = "glossaries/base.json"

[[targets]]
original = "a.json"
stage = "verify"
strict = false
llm_config = "profiles/a.toml"
glossary = "glossaries/a.json"

[[targets]]
original = "b.json"
""", file_ops, logger)

    assert settings.verb == "DEBUG"
    assert settings.result2target is True
    assert settings.cache2target is True
    a, b = settings.targets
    assert (a.stage, a.strict) == ("verify", False)
    assert a.llm_config == os.path.join(base, "profiles", "a.toml")
    assert a.glossary == os.path.join(base, "glossaries", "a.json")
    assert (b.stage, b.strict) == ("translate", True)
    assert b.llm_config == os.path.join(base, "profiles", "base.toml")
    assert b.glossary == os.path.join(base, "glossaries", "base.json")


def test_target_override_wins_over_run_defaults(base, conductor_dir, parser_deps):
    """target ?? run ?? default — target wins even when run sets the opposite."""
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
stage = "translate"
strict = true

[[targets]]
original = "a.json"
stage = "all"
strict = false
""", file_ops, logger)
    assert (settings.targets[0].stage, settings.targets[0].strict) == ("all", False)


def test_inner_dot_dot_that_stays_inside_is_allowed(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
[[targets]]
original = "menu/../hud.json"
""", file_ops, logger)
    assert settings.targets[0].relative_path == "hud.json"


def test_both_slashes_in_original(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    # TOML literal string so backslash is not an escape (basic "\n" would be newline).
    settings = _parse(base, conductor_dir, """
[[targets]]
original = 'dialogs\\npc/guard.json'
""", file_ops, logger)
    assert settings.targets[0].relative_path == os.path.join("dialogs", "npc", "guard.json")


def test_implicit_glossary_beside_manifest(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    gloss = os.path.join(base, "glossary.json")
    with open(gloss, "w", encoding="utf-8") as fh:
        fh.write("{}")
    settings = _parse(base, conductor_dir, """
[[targets]]
original = "a.json"
""", file_ops, logger)
    assert settings.targets[0].glossary == gloss


def test_no_implicit_glossary_when_missing(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
[[targets]]
original = "a.json"
""", file_ops, logger)
    assert settings.targets[0].glossary is None


def test_entries_are_not_required_to_exist_on_disk(base, conductor_dir, parser_deps):
    """Missing original fails that file at run time, not at parse time (#13)."""
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
[[targets]]
original = "absent.json"
""", file_ops, logger)
    assert settings.targets[0].original == os.path.join(base, "absent.json")


# ============================================================================
# draft (#14) — implies verify; conflicts with target stage
# ============================================================================

def test_draft_implies_verify_and_ignores_run_level_stage(base, conductor_dir, parser_deps):
    """Run-level stage is unused for a draft target, not a conflict."""
    file_ops, logger = parser_deps
    settings = _parse(base, conductor_dir, """
stage = "all"

[[targets]]
original = "ui.json"
draft = "drafts/ui.ru.json"

[[targets]]
original = "other.json"
""", file_ops, logger)
    assert settings.targets[0].stage == "verify"
    assert settings.targets[0].draft == os.path.join(base, "drafts", "ui.ru.json")
    assert settings.targets[1].stage == "all"
    assert settings.targets[1].draft is None


def test_draft_plus_target_stage_is_an_error_with_index(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
[[targets]]
original = "ok.json"

[[targets]]
original = "ui.json"
draft = "drafts/ui.ru.json"
stage = "verify"
""", file_ops, logger)
    message = str(exc.value)
    assert "target 2:" in message
    assert "draft implies verify" in message


@pytest.mark.parametrize("stage", ["all", "translate", "verify"])
def test_draft_plus_any_target_stage_is_rejected(base, conductor_dir, parser_deps, stage):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, f"""
[[targets]]
original = "ui.json"
draft = "d.json"
stage = "{stage}"
""", file_ops, logger, match="draft implies verify")


def test_draft_path_escape_is_rejected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, """
[[targets]]
original = "ui.json"
draft = "../outside.ru.json"
""", file_ops, logger, match="outside the manifest directory")


# ============================================================================
# Schema / type / unknown keys / empty
# ============================================================================

def test_unknown_run_and_target_keys_are_collected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
extra = 1
dry_run = true

[[targets]]
original = "a.json"
bonus = true
""", file_ops, logger)
    message = str(exc.value)
    assert "unknown key 'extra'" in message
    assert "unknown key 'dry_run'" in message
    assert "target 1: unknown key 'bonus'" in message


def test_missing_targets_is_not_a_run_manifest(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, 'verb = "INFO"\n', file_ops, logger, match="not a run manifest")


def test_empty_targets_array_is_not_a_run_manifest(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    # tomllib: targets = [] with no [[targets]] tables
    _raises(base, conductor_dir, "targets = []\n", file_ops, logger, match="not a run manifest")


def test_missing_original_reports_target_index(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError, match="target 1: missing 'original'"):
        _parse(base, conductor_dir, """
[[targets]]
stage = "verify"
""", file_ops, logger)


def test_bad_types_are_rejected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
verb = 1
result2target = "yes"
strict = "true"
stage = "nope"

[[targets]]
original = 5
""", file_ops, logger)
    message = str(exc.value)
    assert "'verb' must be a string" in message
    assert "'result2target' must be a boolean" in message
    assert "'strict' must be a boolean" in message
    assert "'stage' must be one of" in message
    assert "target 1:" in message
    assert "'original' must be a non-empty string" in message


def test_invalid_stage_and_verb_values(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, """
verb = "LOUD"
stage = "both"

[[targets]]
original = "a.json"
""", file_ops, logger, match="'verb' must be one of")


def test_conductor_toml_shape_is_not_a_run_manifest(base, conductor_dir, parser_deps):
    """Passing conductor.toml / llm.toml as --target must exit 2."""
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, """
[logging]
level = "INFO"
""", file_ops, logger, match="not a run manifest")


def test_invalid_toml_is_not_a_run_manifest(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, "[[targets]\n", file_ops, logger, match="not a run manifest")


# ============================================================================
# Path / duplicate / collision conflicts
# ============================================================================

@pytest.mark.parametrize("original", [
    "../shared/x.json",
    "..\\shared\\x.json",
    "menu/../../x.json",
    ".",
    "menu/..",
])
def test_paths_leaving_manifest_directory_are_rejected(
        base, conductor_dir, parser_deps, original):
    file_ops, logger = parser_deps
    # Escape backslashes for TOML basic strings.
    escaped = original.replace("\\", "\\\\")
    _raises(base, conductor_dir, f"""
[[targets]]
original = "a.json"

[[targets]]
original = "{escaped}"
""", file_ops, logger, match="outside the manifest directory")


def test_absolute_posix_path_is_rejected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    _raises(base, conductor_dir, """
[[targets]]
original = "/etc/x.json"
""", file_ops, logger, match="absolute path")


def test_windows_drive_paths_are_rejected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    for original in ("D:/Other/x.json", "D:x.json"):
        _raises(base, conductor_dir, f"""
[[targets]]
original = "{original}"
""", file_ops, logger, match="absolute path")


def test_duplicate_originals_after_normalization(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
[[targets]]
original = "menu/main.json"

[[targets]]
original = "menu/./x/../main.json"
""", file_ops, logger)
    assert "target 2: duplicate original of target 1" in str(exc.value)


@pytest.mark.skipif(os.path.normcase("A") == "A", reason="case-sensitive file system")
def test_duplicate_originals_differing_only_in_case(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError, match="duplicate original"):
        _parse(base, conductor_dir, """
[[targets]]
original = "Hud.json"

[[targets]]
original = "hud.JSON"
""", file_ops, logger)


def test_result2target_input_under_result_is_rejected(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError, match="inside the result directory"):
        _parse(base, conductor_dir, """
result2target = true

[[targets]]
original = "result/final/a.json"
""", file_ops, logger)


def test_shared_layout_result_clash_when_manifest_beside_conductor(
        tmp_path, parser_deps):
    """When manifest dir == conductor dir, original under result/<name>/ is rejected."""
    file_ops, logger = parser_deps
    root = str(tmp_path / "app")
    os.makedirs(root, exist_ok=True)
    with pytest.raises(TargetError, match="inside the result directory"):
        _parse(root, root, """
[[targets]]
original = "result/run.toml/final/a.json"
""", file_ops, logger)


def test_llm_config_and_glossary_must_stay_inside_manifest_tree(
        base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
llm_config = "../secrets/llm.toml"
glossary = "../other/glossary.json"

[[targets]]
original = "a.json"
llm_config = "../../llm.toml"
""", file_ops, logger)
    message = str(exc.value)
    assert "llm_config:" in message
    assert "glossary:" in message
    assert "target 1: llm_config:" in message


def test_collect_all_path_violations_with_indices(base, conductor_dir, parser_deps):
    file_ops, logger = parser_deps
    with pytest.raises(TargetError) as exc:
        _parse(base, conductor_dir, """
[[targets]]
original = "ok.json"

[[targets]]
original = "../a.json"

[[targets]]
original = "/b.json"
""", file_ops, logger)
    message = str(exc.value)
    assert "target 2:" in message and "outside" in message
    assert "target 3:" in message and "absolute" in message


# ============================================================================
# Extension helpers
# ============================================================================

@pytest.mark.parametrize("path, expected", [
    ("run.toml", True), ("RUN.TOML", True), ("run.json", False), ("run.list", False),
])
def test_run_manifest_extension(path, expected):
    assert is_run_manifest_path(path) is expected


def test_legacy_list_extension():
    assert is_legacy_list_path("ui.list")
    assert is_legacy_list_path("UI.LIST")
    assert not is_legacy_list_path("ui.toml")
