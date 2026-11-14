# -*- coding: utf-8 -*-
"""
Smoke-test a frozen llm-conductor binary before attaching it to a release.

Runs the built exe away from the repo tree so config resolution uses the
directory of the executable (frozen ``resolve_app_dir``), not the source
checkout. Checks:

1. ``--version`` exits 0 and prints the exact expected version;
2. ``init`` writes ``conductor.toml`` and ``llm.toml`` next to the exe;
3. ``check`` succeeds when invoked with a *different* cwd (configs still
   found next to the executable).

Used by ``.github/workflows/release.yml`` after PyInstaller, and locally::

    python tools/smoke_frozen.py dist/llm-conductor.exe --expected-version 1.0.0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONDUCTOR_TOML = "conductor.toml"
LLM_TOML = "llm.toml"
EXE_NAME = "llm-conductor.exe"


def _run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command; capture combined stdout/stderr as text."""
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _fail(message: str) -> int:
    print(f"SMOKE FAIL: {message}", file=sys.stderr)
    return 1


def smoke(exe: Path, expected_version: str) -> int:
    """
    Copy ``exe`` into a temp directory and run the smoke sequence.

    Args:
        exe: Path to the frozen binary to test.
        expected_version: PEP 440 version to match exactly in ``--version`` output
            (same as ``llm_conductor.__version__`` / the release tag without ``v``).

    Returns:
        Process exit code (0 on success).
    """
    if not exe.is_file():
        return _fail(f"exe not found: {exe}")

    with tempfile.TemporaryDirectory(prefix="llmc-smoke-") as tmp:
        smoke_dir = Path(tmp)
        smoke_exe = smoke_dir / EXE_NAME
        shutil.copy2(exe, smoke_exe)

        # 1) --version
        version_proc = _run([str(smoke_exe), "--version"], cwd=smoke_dir)
        version_out = (version_proc.stdout or "") + (version_proc.stderr or "")
        print(f"--version exit={version_proc.returncode} output={version_out!r}")
        if version_proc.returncode != 0:
            return _fail(f"--version exited {version_proc.returncode}")
        if version_out.strip() != f"llm-conductor {expected_version}":
            return _fail(
                f"--version output did not match {expected_version!r}: {version_out!r}"
            )

        # 2) init → templates next to the exe
        init_proc = _run([str(smoke_exe), "init"], cwd=smoke_dir)
        init_out = (init_proc.stdout or "") + (init_proc.stderr or "")
        print(f"init exit={init_proc.returncode} output={init_out!r}")
        if init_proc.returncode != 0:
            return _fail(f"init exited {init_proc.returncode}")
        conductor_path = smoke_dir / CONDUCTOR_TOML
        llm_path = smoke_dir / LLM_TOML
        if not conductor_path.is_file() or not llm_path.is_file():
            return _fail(
                f"init did not write both configs next to the exe "
                f"(have conductor={conductor_path.is_file()}, llm={llm_path.is_file()})"
            )

        # 3) check from a foreign cwd — must still resolve configs via sys.executable
        foreign_cwd = smoke_dir / "foreign_cwd"
        foreign_cwd.mkdir()
        check_proc = _run([str(smoke_exe), "check"], cwd=foreign_cwd)
        check_out = (check_proc.stdout or "") + (check_proc.stderr or "")
        print(f"check exit={check_proc.returncode} output={check_out!r}")
        if check_proc.returncode != 0:
            return _fail(
                f"check from foreign cwd exited {check_proc.returncode} "
                f"(frozen resolve_app_dir likely broken): {check_out!r}"
            )

    print(f"SMOKE OK: {exe.name} version={expected_version}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args and run smoke."""
    parser = argparse.ArgumentParser(
        description="Smoke-test a frozen llm-conductor.exe before release publish.",
    )
    parser.add_argument(
        "exe",
        type=Path,
        help="Path to llm-conductor.exe (usually dist/llm-conductor.exe)",
    )
    parser.add_argument(
        "--expected-version",
        required=True,
        help="Exact PEP 440 version expected in --version output",
    )
    args = parser.parse_args(argv)
    return smoke(args.exe.resolve(), args.expected_version)


if __name__ == "__main__":
    sys.exit(main())
