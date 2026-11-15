# -*- coding: utf-8 -*-
"""
Unit tests of path helpers, result layout (llm_conductor/cli/targets.py)
and of RunOptions (llm_conductor/cli/runner.py).
"""
import ntpath
import os
import posixpath
from types import SimpleNamespace

import pytest

from llm_conductor.cli import paths as path_helpers
from llm_conductor.cli import targets as target_helpers
from llm_conductor.cli.paths import (
    host_relative_path,
    is_inside,
    manifest_relative_identity,
    portable_relative_path,
    resolve_under_base,
)
from llm_conductor.cli.runner import RunOptions
from llm_conductor.cli.targets import (
    ResultLayout,
    RunTarget,
    TargetEntry,
    TargetResolver,
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

INVALID_FILE_IDENTITIES = (
    "", "///", "\\\\", ".", "..", "../outside.json", r"..\outside.json",
    "menu/../main.json", "menu/./main.json", "/main.json", r"\main.json",
    "C:main.json", "C:/main.json", "menu/C:/main.json", "//server/share/main.json",
)


@pytest.fixture
def base(tmp_path):
    return str(tmp_path / "project")


# ============================================================================
# Paths
# ============================================================================

def test_resolve_under_base_allows_inner_dot_dot(base):
    path = resolve_under_base(base, "menu/../hud.json")
    assert path == os.path.join(base, "hud.json")


@pytest.mark.parametrize("relative", [
    "../shared/x.json",
    "..\\shared\\x.json",
    "menu/../../x.json",
    ".",
    "menu/..",
])
def test_resolve_under_base_rejects_escape(base, relative):
    with pytest.raises(ValueError, match="outside the manifest directory"):
        resolve_under_base(base, relative)


@pytest.mark.parametrize("base_suffix", ["/", "//", "/nested/.."])
@pytest.mark.parametrize("relative", ["", ".", "menu/.."])
def test_resolve_under_base_rejects_base_with_noncanonical_spelling(base, base_suffix, relative):
    with pytest.raises(ValueError, match="outside the manifest directory"):
        resolve_under_base(base + base_suffix, relative)


@pytest.mark.parametrize("base_suffix", ["/", "//", "/nested/.."])
def test_resolve_under_base_normalizes_base_for_child(base, base_suffix):
    assert resolve_under_base(base + base_suffix, "menu/main.json") == os.path.join(
        base, "menu", "main.json")


def test_resolve_under_base_rejects_case_variant_of_base(base):
    relative = os.path.join("..", os.path.basename(base).upper())
    with pytest.raises(ValueError, match="outside the manifest directory"):
        resolve_under_base(base, relative)


@pytest.mark.parametrize("relative", ["/etc/x.json", "\\x.json", "D:\\Other\\x.json", "D:x.json"])
def test_resolve_under_base_rejects_absolute(base, relative):
    with pytest.raises(ValueError, match="absolute path"):
        resolve_under_base(base, relative)


@pytest.mark.parametrize("path, expected", [("ui.toml", True), ("UI.TOML", True), ("ui.json", False),
                                            ("ui.list", False), ("toml", False)])
def test_manifest_is_recognized_by_extension(path, expected):
    assert is_manifest_path(path) is expected


def test_is_inside(base):
    assert is_inside(os.path.join(base, "a", "b.json"), base)
    assert is_inside(base, base)
    assert not is_inside(base + "-other", base)
    assert not is_inside(os.path.dirname(base), base)


@pytest.mark.parametrize("raw, expected", [
    ("menu/main.json", "menu/main.json"),
    (r"menu\main.json", "menu/main.json"),
    (r"dialogs\npc/guard.json", "dialogs/npc/guard.json"),
    ("sample.json", "sample.json"),
])
def test_portable_relative_path_normalizes_separators(raw, expected):
    assert portable_relative_path(raw) == expected


def test_host_relative_path_uses_os_separators():
    assert host_relative_path("menu/main.json") == os.path.join("menu", "main.json")
    assert host_relative_path(r"dialogs\npc\guard.json") == os.path.join(
        "dialogs", "npc", "guard.json")


@pytest.mark.parametrize("path_module", [posixpath, ntpath], ids=["posix", "windows"])
@pytest.mark.parametrize("relative", INVALID_FILE_IDENTITIES)
def test_host_relative_path_rejects_unsafe_identities(monkeypatch, path_module, relative):
    monkeypatch.setattr(path_helpers, "os", SimpleNamespace(
        path=path_module, curdir=os.curdir, pardir=os.pardir,
    ))
    with pytest.raises(ValueError, match="relative file path"):
        host_relative_path(relative)


@pytest.mark.parametrize("path_module", [posixpath, ntpath], ids=["posix", "windows"])
@pytest.mark.parametrize("relative", ["menu/main.json", "menu//main.json", r"menu\main.json"])
def test_host_relative_path_keeps_valid_nested_paths(monkeypatch, path_module, relative):
    monkeypatch.setattr(path_helpers, "os", SimpleNamespace(
        path=path_module, curdir=os.curdir, pardir=os.pardir,
    ))
    assert host_relative_path(relative) == path_module.join("menu", "main.json")


@pytest.mark.parametrize("relative", INVALID_FILE_IDENTITIES)
def test_target_entry_rejects_unsafe_result_identity(base, relative):
    with pytest.raises(TargetError, match="Invalid result-relative path"):
        TargetEntry(os.path.join(base, "input.json"), relative)


def test_single_posix_file_with_separator_only_name_is_rejected(monkeypatch):
    """Model POSIX path semantics without creating a Windows-invalid filename."""
    monkeypatch.setattr(path_helpers, "os", SimpleNamespace(
        path=posixpath, curdir=os.curdir, pardir=os.pardir,
    ))
    monkeypatch.setattr(target_helpers, "os", SimpleNamespace(path=SimpleNamespace(
        abspath=posixpath.abspath, basename=posixpath.basename, isfile=lambda path: True,
    )))
    resolver = TargetResolver(FileOperations(), RecordingLogger())
    with pytest.raises(TargetError, match="Invalid result-relative path"):
        resolver.resolve("/inputs/" + "\\" * 2)


def test_manifest_relative_identity_is_portable(base):
    absolute = os.path.join(base, "input", "route-separate.json")
    assert manifest_relative_identity(absolute, base) == "input/route-separate.json"


def test_run_target_properties(base):
    target = RunTarget(os.path.join(base, "ui.toml"), ())

    assert (target.name, target.directory, target.is_manifest) == ("ui.toml", base, True)


# ============================================================================
# ResultLayout
# ============================================================================

def test_shared_layout_is_named_after_the_target(base, tmp_path):
    target = RunTarget(os.path.join(base, "ui.toml"), ())
    entry = TargetEntry(os.path.join(base, "menu", "main.json"), "menu/main.json")

    layout = ResultLayout.for_target(target, str(tmp_path / "app"), next_to_target=False)

    assert layout.shared_by_name is True
    assert layout.stage_path("final", entry) == str(tmp_path / "app" / "result" / "ui.toml" / "final"
                                                    / "menu" / "main.json")
    assert layout.leaves_path(entry) == str(
        tmp_path / "app" / "result" / "ui.toml" / "leaves" / "menu" / "main.json")


def test_result2target_layout_is_next_to_the_target(base):
    target = RunTarget(os.path.join(base, "strings.json"), ())

    layout = ResultLayout.for_target(target, "unused", next_to_target=True)

    assert layout.shared_by_name is False
    assert layout.root == os.path.join(base, "result")


def test_inputs_inside_the_result_root_are_rejected(base):
    inside = TargetEntry(os.path.join(base, "result", "final", "a.json"), "a.json")
    outside = TargetEntry(os.path.join(base, "a.json"), "a.json")
    layout = ResultLayout(os.path.join(base, "result"))

    layout.check_disjoint(RunTarget(os.path.join(base, "ui.toml"), (outside,)))
    with pytest.raises(TargetError, match="would be overwritten"):
        layout.check_disjoint(RunTarget(os.path.join(base, "ui.toml"), (outside, inside)))


def test_draft_inside_the_result_root_is_rejected(base):
    entry = TargetEntry(
        os.path.join(base, "a.json"), "a.json",
        draft_path=os.path.join(base, "result", "final", "a.json"),
    )
    layout = ResultLayout(os.path.join(base, "result"))
    with pytest.raises(TargetError, match="would be overwritten"):
        layout.check_disjoint(RunTarget(os.path.join(base, "ui.toml"), (entry,)))


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
