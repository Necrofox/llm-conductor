# -*- coding: utf-8 -*-
"""
System tests through a real process: python llm-conductor.py ...

Only paths without network. They check what the in-process run replaces:
real interpreter exit codes, stdout/stderr, and an entry point started from
a foreign working directory. LLMC_CONFIG always points into the temporary
directory, so no test reads or writes configuration next to the repository's
entry point.
"""
import os
import subprocess
import sys
import tomllib

from llm_conductor.constants import (
    CONFIG_PATH_ENV_VAR,
    EXIT_CONFIG_ERROR,
    EXIT_TARGET_LOCKED,
    EXIT_USAGE_ERROR,
)
from llm_conductor.core.locking import FileLock
from tests.system.harness import ENTRY_POINT


def _run(tmp_path, *args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           CONFIG_PATH_ENV_VAR: str(tmp_path / "conductor.toml")}
    return subprocess.run(
        [sys.executable, str(ENTRY_POINT), *args],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
        check=False,      # the exit code is what is being checked
    )


def test_init_writes_both_templates(tmp_path):
    result = _run(tmp_path, "init")

    assert result.returncode == 0, result.stderr
    assert f"Created {tmp_path / 'conductor.toml'}" in result.stdout
    llm = tomllib.loads((tmp_path / "llm.toml").read_text(encoding="utf-8"))
    assert llm["translation"]["api"]["type"] == "ollama"
    assert "# " in (tmp_path / "conductor.toml").read_text(encoding="utf-8")


def test_init_refuses_to_overwrite(tmp_path):
    _run(tmp_path, "init")

    result = _run(tmp_path, "init")

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "Refusing to overwrite" in result.stderr


def test_check_accepts_fresh_templates(tmp_path):
    _run(tmp_path, "init")

    result = _run(tmp_path, "check")

    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_check_reports_a_misspelled_key(tmp_path):
    _run(tmp_path, "init")
    conductor = tmp_path / "conductor.toml"
    # Target [dedup]'s own "enabled" specifically, rather than the last
    # occurrence in the file - other sections ([validation]) have a setting
    # of the same name, and their relative order is not part of the contract.
    before_dedup, dedup_and_after = conductor.read_text(encoding="utf-8").split("[dedup]", 1)
    dedup_and_after = dedup_and_after.replace("enabled = true", "enabeld = true", 1)
    conductor.write_text(before_dedup + "[dedup]" + dedup_and_after, encoding="utf-8")

    result = _run(tmp_path, "check")

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "unknown key 'enabeld' in [dedup] (did you mean 'enabled'?)" in result.stderr
    assert "conductor.toml" in result.stderr and "Traceback" not in result.stderr


def test_no_arguments_prints_help_and_fails(tmp_path):
    result = _run(tmp_path)

    assert result.returncode == EXIT_USAGE_ERROR
    assert "usage:" in result.stdout


def test_missing_config_fails_without_traceback(tmp_path):
    result = _run(tmp_path, "run", "--target", "input.json")

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_lists_subcommands_and_run_flags(tmp_path):
    top = _run(tmp_path, "--help")
    run = _run(tmp_path, "run", "--help")

    assert top.returncode == run.returncode == 0
    for command in ("run", "init", "check"):
        assert command in top.stdout
    for flag in ("--target", "--stage", "--strict", "--no-cache", "--llm-config", "--glossary",
                 "--verb", "--result2target", "--cache2target"):
        assert flag in run.stdout


def test_version_flag_prints_package_version(tmp_path):
    from llm_conductor import __version__

    result = _run(tmp_path, "--version")

    assert result.returncode == 0
    # prog is fixed to "llm-conductor" (not argv0 basename / __main__.py).
    assert result.stdout.strip() == f"llm-conductor {__version__}"


def test_target_locked_by_another_process_exits_with_the_locked_code(tmp_path):
    """The lock is held by the pytest process; the run is a separate process."""
    _run(tmp_path, "init")
    target = tmp_path / "strings.json"
    target.write_text('{"a": "Hello"}', encoding="utf-8")

    with FileLock(str(target)):
        result = _run(tmp_path, "run", "--target", "strings.json")

    assert result.returncode == EXIT_TARGET_LOCKED, result.stdout + result.stderr
    assert f"{target} is locked by another run" in result.stdout
    (log,) = (tmp_path / "logs").glob("conductor_strings.json_*.log")
    assert "is locked by another run" in log.read_text(encoding="utf-8")
