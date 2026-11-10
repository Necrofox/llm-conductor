# -*- coding: utf-8 -*-
"""
System tests for ``run --dry-run`` (GH-9, GH-19, GH-21).
"""
import pytest

from llm_conductor.constants import (
    CACHE_LOCK_FILENAME,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    RESULT_OWNER_FILENAME,
)
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


def test_dry_run_draft_structure_mismatch_exit_2(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", {"title": "x", "extra": "nope"})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "extra" in combined or "structure" in combined
    assert model.requests == []
    assert not (project.root / "result").exists()


def test_dry_run_draft_marker_mismatch_exit_2(project, model, run_main, capsys):
    project.write_json("project/menu/main.json", {"line": "Open {0}."})
    project.write_json("project/drafts/main.ru.json", {"line": "Открой."})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "marker" in combined
    assert model.requests == []
    assert not (project.root / "result").exists()


def test_dry_run_collects_preflight_errors_across_targets(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", {"title": "x", "extra": "nope"})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"

[[targets]]
original = "dialogs/missing.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "target 2:" in combined
    assert "missing.json" in combined.replace("\\", "/")
    assert model.requests == []


def test_dry_run_prints_draft_and_verify_stage_in_table(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", {"title": "Главное меню"})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--dry-run") == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "verify" in out
    assert "drafts" in out.replace("\\", "/") or "main.ru.json" in out
    assert model.requests == []


def test_dry_run_root_null_exit_2(project, model, run_main, capsys):
    """Root JSON null fails shared preflight (GH-19 / GH-21 scenario A)."""
    project.write_text("project/menu/null.json", "null")
    assert run_main(
        "run", "--target", "project/menu/null.json", "--dry-run",
    ) == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "null data" in combined
    assert model.requests == []
    assert not (project.root / "result").exists()


def test_dry_run_verify_with_translated_exit_0(project, model, run_main, capsys):
    """Verify-only dry-run succeeds when translated/ is present and matches."""
    project.write_json(
        project.result("translated", "main.json", target="main.json"),
        {"title": "Главное меню"},
    )
    code = run_main(
        "run", "--target", "project/menu/main.json", "--stage", "verify", "--dry-run",
    )
    assert code == EXIT_SUCCESS
    assert "verify" in capsys.readouterr().out
    assert model.requests == []
    assert not project.exists(f"result/main.json/{RESULT_OWNER_FILENAME}")
    assert not project.exists(f"result/main.json/{CACHE_LOCK_FILENAME}")


@pytest.mark.parametrize("mode", [(), ("--dry-run",)], ids=["real", "dry-run"])
def test_verify_refuses_translated_result_owned_by_another_target(
        project, model, run_main, capsys, mode):
    project.write_json("other/main.json", MENU)
    assert run_main(
        "run", "--target", "project/menu/main.json", "--stage", "translate",
    ) == EXIT_SUCCESS
    marker = project.root / "result" / "main.json" / RESULT_OWNER_FILENAME
    owner = marker.read_text(encoding="utf-8")
    model.requests.clear()

    assert run_main(
        "run", "--target", "other/main.json", "--stage", "verify", *mode,
    ) == EXIT_USAGE_ERROR
    assert "belongs to" in "".join(capsys.readouterr())
    assert marker.read_text(encoding="utf-8") == owner
    assert model.requests == []
