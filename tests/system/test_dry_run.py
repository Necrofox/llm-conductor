# -*- coding: utf-8 -*-
"""
System tests for ``run --dry-run`` (GH-9).
"""
import pytest

from llm_conductor.constants import EXIT_SUCCESS, EXIT_USAGE_ERROR
from llm_conductor.core.locking import FileLock

MENU = {"title": "Main menu"}


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


def test_dry_run_manifest_prints_table_without_api_or_results(
        project, model, run_main, capsys):
    code = run_main("run", "--target", "project/ui.toml", "--dry-run")
    assert code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "index" in out
    assert "original" in out
    assert "menu/main.json" in out.replace("\\", "/") or "main.json" in out
    assert model.requests == []
    assert not (project.root / "result").exists()


def test_dry_run_json_target_exit_0(project, model, run_main, capsys):
    code = run_main("run", "--target", "project/menu/main.json", "--dry-run")
    assert code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "index" in out
    assert model.requests == []
    assert not project.exists(project.result("final", "main.json", target="main.json"))


def test_dry_run_contract_error_exit_2(project, model, run_main, capsys):
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
stage = "verify"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "draft implies verify" in combined
    assert model.requests == []


def test_dry_run_missing_original_exit_2(project, model, run_main, capsys):
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/missing.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert model.requests == []


def test_dry_run_duplicate_keys_warn_exit_0(project, model, run_main, capsys):
    # Write raw JSON with a duplicate key (json.dump cannot emit this).
    dup_path = project.root / "project" / "menu" / "dup.json"
    dup_path.write_text('{"title": "A", "title": "B"}', encoding="utf-8")
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/dup.json"
""")
    code = run_main("run", "--target", "project/ui.toml", "--dry-run", "--verb", "WARNING")
    assert code == EXIT_SUCCESS
    combined = "".join(capsys.readouterr())
    assert "duplicate key" in combined.lower()
    assert "last value wins" in combined.lower()
    assert model.requests == []


def test_dry_run_succeeds_while_target_lock_held(project, model, run_main):
    """Dry-run neither acquires nor checks locks (option 3)."""
    target = project.root / "project" / "menu" / "main.json"
    with FileLock(str(target)):
        assert run_main(
            "run", "--target", "project/menu/main.json", "--dry-run",
        ) == EXIT_SUCCESS
    assert model.requests == []


def test_dry_run_session_flag_allowed_with_manifest(project, model, run_main, capsys):
    assert run_main(
        "run", "--target", "project/ui.toml", "--dry-run", "--verb", "INFO",
    ) == EXIT_SUCCESS
    assert model.requests == []
    assert "index" in capsys.readouterr().out
