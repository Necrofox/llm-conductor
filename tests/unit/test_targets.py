# -*- coding: utf-8 -*-
"""
Unit tests of the manifest rules and the result layout (llm_conductor/cli/targets.py)
and of RunOptions (llm_conductor/cli/runner.py). Manifest and path-shape cases only
compute paths; claim_owner() uses a real tmp directory.
"""
import os

import pytest

from llm_conductor.cli.runner import RunOptions
from llm_conductor.cli.targets import (
    ManifestParser,
    ResultLayout,
    RunTarget,
    TargetEntry,
    is_inside,
    is_manifest_path,
)
from llm_conductor.constants import (
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    RESULT_OWNER_FILENAME,
)
from llm_conductor.core.io import FileOperations
from llm_conductor.exceptions import FileOperationError, TargetError
from tests.support import RecordingLogger


@pytest.fixture
def base(tmp_path):
    return str(tmp_path / "project")


def _parse(base, text):
    return ManifestParser(os.path.join(base, "ui.list")).parse(text)


# ============================================================================
# ManifestParser
# ============================================================================

def test_manifest_lines_resolve_from_the_manifest_directory(base):
    entries = _parse(base, "menu/main.json\nhud.json\n")

    assert entries == (
        TargetEntry(os.path.join(base, "menu", "main.json"), os.path.join("menu", "main.json")),
        TargetEntry(os.path.join(base, "hud.json"), "hud.json"),
    )


def test_blank_lines_comments_whitespace_and_bom_are_ignored(base):
    entries = _parse(base, "\ufeff# header\n\n   hud.json \t\n  # indented comment\n")

    assert [e.relative_path for e in entries] == ["hud.json"]


def test_both_slashes_separate_path_parts(base):
    (entry,) = _parse(base, "dialogs\\npc/guard.json")

    assert entry.relative_path == os.path.join("dialogs", "npc", "guard.json")


def test_inner_dot_dot_that_stays_inside_is_allowed(base):
    (entry,) = _parse(base, "menu/../hud.json")

    assert entry.relative_path == "hud.json"


@pytest.mark.parametrize("line", [
    "../shared/x.json",
    "..\\shared\\x.json",
    "menu/../../x.json",
    ".",
    "menu/..",
])
def test_paths_leaving_the_manifest_directory_are_rejected(base, line):
    with pytest.raises(TargetError, match="outside the manifest directory"):
        _parse(base, f"hud.json\n{line}\n")


@pytest.mark.parametrize("line", ["/etc/x.json", "\\x.json", "D:\\Other\\x.json", "D:x.json"])
def test_absolute_and_drive_paths_are_rejected(base, line):
    with pytest.raises(TargetError, match="absolute path"):
        _parse(base, line)


def test_every_violation_is_reported_at_once_with_line_numbers(base):
    with pytest.raises(TargetError) as exc:
        _parse(base, "# c\nhud.json\n../a.json\n/b.json\n")

    message = str(exc.value)
    assert "line 3: ../a.json" in message and "line 4: /b.json" in message


def test_duplicates_after_normalization_are_rejected(base):
    with pytest.raises(TargetError, match="line 3: .* duplicate of line 1"):
        _parse(base, "menu/main.json\nhud.json\nmenu/./x/../main.json\n")


@pytest.mark.skipif(os.path.normcase("A") == "A", reason="case-sensitive file system")
def test_duplicates_differing_only_in_case_are_rejected_on_windows(base):
    with pytest.raises(TargetError, match="duplicate"):
        _parse(base, "Hud.json\nhud.JSON\n")


@pytest.mark.parametrize("text", ["", "# only comments\n\n"])
def test_manifest_without_files_is_rejected(base, text):
    with pytest.raises(TargetError, match="lists no files"):
        _parse(base, text)


# ============================================================================
# Paths
# ============================================================================

@pytest.mark.parametrize("path, expected", [("ui.list", True), ("UI.LIST", True), ("ui.json", False),
                                            ("list", False)])
def test_manifest_is_recognized_by_extension(path, expected):
    assert is_manifest_path(path) is expected


def test_is_inside(base):
    assert is_inside(os.path.join(base, "a", "b.json"), base)
    assert is_inside(base, base)
    assert not is_inside(base + "-other", base)          # a common string prefix is not containment
    assert not is_inside(os.path.dirname(base), base)


def test_run_target_properties(base):
    target = RunTarget(os.path.join(base, "ui.list"), ())

    assert (target.name, target.directory, target.is_manifest) == ("ui.list", base, True)


# ============================================================================
# ResultLayout
# ============================================================================

def test_shared_layout_is_named_after_the_target(base, tmp_path):
    target = RunTarget(os.path.join(base, "ui.list"), ())
    entry = TargetEntry(os.path.join(base, "menu", "main.json"), os.path.join("menu", "main.json"))

    layout = ResultLayout.for_target(target, str(tmp_path / "app"), next_to_target=False)

    assert layout.shared_by_name is True
    assert layout.stage_path("final", entry) == str(tmp_path / "app" / "result" / "ui.list" / "final"
                                                    / "menu" / "main.json")


def test_result2target_layout_is_next_to_the_target(base):
    target = RunTarget(os.path.join(base, "strings.json"), ())

    layout = ResultLayout.for_target(target, "unused", next_to_target=True)

    assert layout.shared_by_name is False
    assert layout.root == os.path.join(base, "result")


def test_inputs_inside_the_result_root_are_rejected(base):
    inside = TargetEntry(os.path.join(base, "result", "final", "a.json"), "a.json")
    outside = TargetEntry(os.path.join(base, "a.json"), "a.json")
    layout = ResultLayout(os.path.join(base, "result"))

    layout.check_disjoint(RunTarget(os.path.join(base, "ui.list"), (outside,)))
    with pytest.raises(TargetError, match="would be overwritten"):
        layout.check_disjoint(RunTarget(os.path.join(base, "ui.list"), (outside, inside)))


# ============================================================================
# ResultLayout.claim_owner
# ============================================================================

@pytest.fixture
def file_ops():
    return FileOperations(logger=RecordingLogger())


def _canonical(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def test_claim_owner_writes_the_marker_on_first_use(tmp_path, file_ops):
    root = tmp_path / "result" / "strings.json"
    target = str(tmp_path / "proj" / "strings.json")
    layout = ResultLayout(str(root), shared_by_name=True)

    layout.claim_owner(target, file_ops)

    marker = root / RESULT_OWNER_FILENAME
    assert marker.read_text(encoding="utf-8").strip() == _canonical(target)


def test_claim_owner_accepts_a_matching_marker(tmp_path, file_ops):
    root = tmp_path / "result" / "strings.json"
    target = str(tmp_path / "proj" / "strings.json")
    layout = ResultLayout(str(root), shared_by_name=True)
    layout.claim_owner(target, file_ops)

    layout.claim_owner(target, file_ops)

    assert (root / RESULT_OWNER_FILENAME).read_text(encoding="utf-8").strip() == _canonical(target)


def test_claim_owner_rejects_a_foreign_target(tmp_path, file_ops):
    root = tmp_path / "result" / "strings.json"
    first = str(tmp_path / "a" / "strings.json")
    second = str(tmp_path / "b" / "strings.json")
    layout = ResultLayout(str(root), shared_by_name=True)
    layout.claim_owner(first, file_ops)

    with pytest.raises(TargetError, match="belongs to"):
        layout.claim_owner(second, file_ops)


@pytest.mark.parametrize("content", ["", "\n", "line1\nline2\n", "  \n  "])
def test_claim_owner_rejects_a_corrupt_marker(tmp_path, file_ops, content):
    root = tmp_path / "result" / "strings.json"
    root.mkdir(parents=True)
    (root / RESULT_OWNER_FILENAME).write_text(content, encoding="utf-8")
    layout = ResultLayout(str(root), shared_by_name=True)

    with pytest.raises(TargetError, match="corrupt owner marker"):
        layout.claim_owner(str(tmp_path / "proj" / "strings.json"), file_ops)


def test_claim_owner_is_a_noop_when_not_shared_by_name(tmp_path, file_ops):
    root = tmp_path / "project" / "result"
    layout = ResultLayout(str(root), shared_by_name=False)

    layout.claim_owner(str(tmp_path / "project" / "strings.json"), file_ops)

    assert not root.exists()


def test_claim_owner_maps_write_false_to_target_error(tmp_path, file_ops, monkeypatch):
    root = tmp_path / "result" / "strings.json"
    layout = ResultLayout(str(root), shared_by_name=True)
    monkeypatch.setattr(
        "llm_conductor.cli.targets.safe_write_file", lambda *args, **kwargs: False)

    with pytest.raises(TargetError, match="Cannot write the owner marker"):
        layout.claim_owner(str(tmp_path / "proj" / "strings.json"), file_ops)


def test_claim_owner_maps_write_file_operation_error_to_target_error(
        tmp_path, file_ops, monkeypatch):
    root = tmp_path / "result" / "strings.json"
    layout = ResultLayout(str(root), shared_by_name=True)

    def _raise(*args, **kwargs):
        raise FileOperationError("Failed to write marker, saved to fallback")

    monkeypatch.setattr("llm_conductor.cli.targets.safe_write_file", _raise)

    with pytest.raises(TargetError, match="Cannot write the owner marker"):
        layout.claim_owner(str(tmp_path / "proj" / "strings.json"), file_ops)


def test_claim_owner_compares_paths_case_folded(tmp_path, file_ops):
    """On case-insensitive FS, a differently cased absolute path still matches."""
    root = tmp_path / "result" / "strings.json"
    target = tmp_path / "proj" / "strings.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    layout = ResultLayout(str(root), shared_by_name=True)
    layout.claim_owner(str(target), file_ops)

    # Flip case of the drive letter / path where the platform folds case;
    # normcase makes both sides equal, so rematch must succeed.
    flipped = str(target)
    if os.name == "nt":
        flipped = flipped.swapcase()
    layout.claim_owner(flipped, file_ops)


# ============================================================================
# RunOptions
# ============================================================================

@pytest.mark.parametrize("stage, operations", [
    ("all", (OPERATION_TRANSLATION, OPERATION_VERIFICATION)),
    ("translate", (OPERATION_TRANSLATION,)),
    ("verify", (OPERATION_VERIFICATION,)),
])
def test_stage_selects_operations(stage, operations):
    assert RunOptions(stage=stage).operations == operations


def test_strict_selects_the_translation_mode():
    assert RunOptions(strict=True).translation_mode == "strict"
    assert RunOptions().translation_mode == "style"
