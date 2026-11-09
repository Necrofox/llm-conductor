# -*- coding: utf-8 -*-
"""
Integration tests of what a run reads before processing: the target or
manifest (TargetResolver), the glossary chain (GlossaryResolver) and the
result directories (ResultLayout.prepare), on real files in tmp_path.
"""
import json
import os

import pytest

from llm_conductor.cli.glossary import GlossaryResolver
from llm_conductor.cli.targets import (
    ResultLayout,
    RunTarget,
    TargetEntry,
    TargetResolver,
)
from llm_conductor.constants import (
    CACHE_LOCK_FILENAME,
    GLOSSARY_ORIGIN_OPTION,
    GLOSSARY_ORIGIN_TARGET,
    RESULT_OWNER_FILENAME,
)
from llm_conductor.core.io import FileOperations
from llm_conductor.core.locking import FileLock
from llm_conductor.exceptions import TargetError
from tests.support import RecordingLogger


@pytest.fixture
def logger():
    return RecordingLogger()


@pytest.fixture
def resolver(logger, tmp_path):
    return TargetResolver(
        FileOperations(logger=logger), logger, conductor_dir=str(tmp_path),
    )


@pytest.fixture
def glossaries(logger):
    return GlossaryResolver(FileOperations(logger=logger), logger)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


# ============================================================================
# TargetResolver
# ============================================================================

def test_single_file_is_its_own_only_entry(resolver, tmp_path):
    path = _write(tmp_path / "strings.json", {"a": "b"})

    target = resolver.resolve(str(path))

    assert target.entries == (TargetEntry(str(path), "strings.json"),)
    assert not target.is_manifest


def test_manifest_is_read_and_parsed(resolver, tmp_path, monkeypatch):
    _write(tmp_path / "project" / "ui.toml", """
[[targets]]
original = "menu/main.json"
""")
    monkeypatch.chdir(tmp_path)

    target = resolver.resolve(os.path.join("project", "ui.toml"))

    assert target.path == str(tmp_path / "project" / "ui.toml")
    assert [e.source_path for e in target.entries] == [
        str(tmp_path / "project" / "menu" / "main.json")
    ]
    assert target.is_manifest


def test_missing_target_is_a_target_error(resolver, tmp_path):
    with pytest.raises(TargetError, match="not found"):
        resolver.resolve(str(tmp_path / "absent.toml"))


def test_directory_target_is_a_target_error(resolver, tmp_path):
    with pytest.raises(TargetError, match="not found"):
        resolver.resolve(str(tmp_path))


def test_manifest_entries_are_not_checked_for_existence(resolver, tmp_path):
    """A missing listed file fails alone during the run, not the whole manifest."""
    path = _write(tmp_path / "ui.toml", """
[[targets]]
original = "absent.json"
""")

    assert len(resolver.resolve(str(path)).entries) == 1


def test_legacy_list_is_rejected(resolver, tmp_path):
    path = _write(tmp_path / "ui.list", "menu/main.json\n")
    with pytest.raises(TargetError, match=r"\.list manifests are no longer supported"):
        resolver.resolve(str(path))


# ============================================================================
# GlossaryResolver
# ============================================================================

def _target(directory):
    return RunTarget(str(directory / "strings.json"), ())


def test_explicit_glossary_wins(glossaries, tmp_path):
    _write(tmp_path / "glossary.json", {"Guild": "Гильдия"})
    explicit = _write(tmp_path / "terms" / "other.json", {"Guild": "Орден"})

    source = glossaries.resolve(str(explicit), _target(tmp_path))

    assert (source.terms, source.path, source.origin) == ({"Guild": "Орден"}, str(explicit), GLOSSARY_ORIGIN_OPTION)
    assert source.describe() == f"Glossary: {explicit} ({GLOSSARY_ORIGIN_OPTION}), 1 term(s)"


def test_glossary_next_to_the_target(glossaries, tmp_path):
    implicit = _write(tmp_path / "glossary.json", {"Guild": "Гильдия"})

    source = glossaries.resolve(None, _target(tmp_path))

    assert (source.path, source.origin) == (str(implicit), GLOSSARY_ORIGIN_TARGET)


def test_no_glossary_anywhere_is_an_empty_one(glossaries, tmp_path):
    source = glossaries.resolve(None, _target(tmp_path))

    assert source.terms == {} and source.path is None
    assert "working without terms" in source.describe()


def test_missing_explicit_glossary_is_a_target_error(glossaries, tmp_path):
    with pytest.raises(TargetError, match="not found"):
        glossaries.resolve(str(tmp_path / "absent.json"), _target(tmp_path))


@pytest.mark.parametrize("content", ["{broken", ["a", "list"], {"Guild": 5}])
def test_broken_implicit_glossary_is_a_target_error_not_a_silent_fallback(glossaries, tmp_path, content):
    _write(tmp_path / "glossary.json", content)

    with pytest.raises(TargetError):
        glossaries.resolve(None, _target(tmp_path))


# ============================================================================
# ResultLayout.prepare / claim_owner
# ============================================================================

def test_prepare_creates_the_nested_result_directory(tmp_path):
    layout = ResultLayout(str(tmp_path / "result"))
    entry = TargetEntry(str(tmp_path / "menu" / "main.json"), os.path.join("menu", "main.json"))

    path = layout.prepare("final", entry)

    assert path == str(tmp_path / "result" / "final" / "menu" / "main.json")
    assert os.path.isdir(os.path.dirname(path)) and not os.path.exists(path)


def test_claim_owner_on_real_files_refuses_a_second_target(tmp_path, logger):
    file_ops = FileOperations(logger=logger)
    root = tmp_path / "result" / "strings.json"
    first = str(tmp_path / "a" / "strings.json")
    second = str(tmp_path / "b" / "strings.json")
    layout = ResultLayout(str(root), shared_by_name=True)

    layout.claim_owner(first, file_ops, logger)
    with pytest.raises(TargetError, match="belongs to"):
        layout.claim_owner(second, file_ops, logger)

    assert (root / RESULT_OWNER_FILENAME).is_file()


def test_shared_result_root_lock_file_is_creatable(tmp_path):
    """The result-root lock uses the same filename as the cache lock."""
    lock_path = tmp_path / "result" / "sample.json" / CACHE_LOCK_FILENAME

    with FileLock(str(lock_path), create=True) as held:
        assert held.locked
        assert lock_path.is_file()
