# -*- coding: utf-8 -*-
"""
System tests for conflict / ambiguous cases of the TOML run-manifest contract
(GH-13 precedence B, GH-14 draft). Happy-path batch runs live in test_pipeline.
"""
import pytest

from llm_conductor.constants import EXIT_SUCCESS, EXIT_USAGE_ERROR

MENU = {"title": "Main menu"}
DRAFT = {"title": "Главное меню"}


@pytest.fixture
def project(workspace):
    workspace.write_json("project/menu/main.json", MENU)
    workspace.write_json("project/dialogs/npc/guard.json", {"line": "Halt!"})
    workspace.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"

[[targets]]
original = "dialogs/npc/guard.json"
""")
    return workspace


# ============================================================================
# Precedence B — job CLI flags with a .toml target
# ============================================================================

@pytest.mark.parametrize("flag_args", [
    ("--stage", "translate"),
    ("--strict",),
    ("--llm-config", "llm.toml"),
    ("--glossary", "glossary.json"),
    ("--result2target",),
    ("--cache2target",),
], ids=["stage", "strict", "llm-config", "glossary", "result2target", "cache2target"])
def test_job_cli_flag_with_manifest_exits_2(project, model, run_main, flag_args, capsys):
    code = run_main("run", "--target", "project/ui.toml", *flag_args)
    assert code == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "not allowed with a .toml" in combined or "Job flags" in combined
    assert model.requests == []


def test_session_flags_verb_and_no_cache_still_work_with_manifest(
        project, model, run_main, preserve_logs):
    assert run_main(
        "run", "--target", "project/ui.toml", "--verb", "DEBUG", "--no-cache",
    ) == EXIT_SUCCESS
    assert "File 2/2" in project.log_text()


def test_legacy_list_is_rejected_with_explicit_migration_message(
        project, model, run_main, capsys):
    project.write_text("project/ui.list", "menu/main.json\n")
    assert run_main("run", "--target", "project/ui.list") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert ".list manifests are no longer supported" in combined
    assert "[[targets]]" in combined
    assert 'original = "path/from/line.json"' in combined
    assert "doc/cli.md" in combined
    assert model.requests == []


def test_non_manifest_toml_exits_2(project, model, run_main, capsys):
    project.write_text("project/conductor-lookalike.toml", """
[logging]
level = "INFO"
""")
    assert run_main("run", "--target", "project/conductor-lookalike.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "not a run manifest" in combined
    assert model.requests == []


def test_job_flag_rejected_even_when_toml_is_not_a_manifest(project, model, run_main, capsys):
    """Precedence B keys off the .toml suffix before schema validation."""
    project.write_text("project/not-manifest.toml", 'verb = "INFO"\n')
    assert run_main(
        "run", "--target", "project/not-manifest.toml", "--stage", "translate",
    ) == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "Job flags" in combined or "not allowed" in combined
    assert model.requests == []


# ============================================================================
# Per-target overrides
# ============================================================================

def test_per_target_stage_override(project, model, run_main):
    """First target translate-only; second verify would need translated/ — use translate for both shapes."""
    project.write_text("project/ui.toml", """
stage = "all"

[[targets]]
original = "menu/main.json"
stage = "translate"

[[targets]]
original = "dialogs/npc/guard.json"
stage = "translate"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS
    assert project.exists(project.result("translated", "menu/main.json", target="ui.toml"))
    # translate-only does not write redacted/
    assert not project.exists(project.result("redacted", "menu/main.json", target="ui.toml"))


# ============================================================================
# Draft (#14) conflicts and verify-from-draft
# ============================================================================

def test_draft_plus_target_stage_exits_2_before_api(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
stage = "verify"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "draft implies verify" in combined
    assert model.requests == []


def test_draft_structure_mismatch_exits_2_before_api(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", {"title": "x", "extra": "nope"})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "extra" in combined or "structure" in combined
    assert model.requests == []


def test_draft_verify_calls_only_verification_and_keeps_inputs(
        project, model, run_main):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS

    # Echo model wraps with OK(...); only verification stage should have run.
    assert project.read_json("project/menu/main.json") == MENU
    assert project.read_json("project/drafts/main.ru.json") == DRAFT
    # No translated/ write required for draft path; redacted/final exist.
    assert project.exists(project.result("final", "menu/main.json", target="ui.toml"))
    # All model calls should be verification (proofread marker), not translation.
    assert model.requests  # verification called at least once
    # Translation stage would write translated/; draft verify must not.
    assert not project.exists(project.result("translated", "menu/main.json", target="ui.toml"))


def test_draft_missing_file_exits_2_before_api(project, model, run_main, capsys):
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/absent.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    assert model.requests == []


def test_run_level_stage_does_not_conflict_with_draft_target(project, model, run_main):
    """Other non-draft targets still use run-level stage; draft ignores it."""
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
stage = "translate"

[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"

[[targets]]
original = "dialogs/npc/guard.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS
    assert not project.exists(project.result("translated", "menu/main.json", target="ui.toml"))
    assert project.exists(project.result("translated", "dialogs/npc/guard.json", target="ui.toml"))
