"""Exercise SCM versions through real offline builds and runtime imports."""
from __future__ import annotations

import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Versions must come from the fixture, even if CI supplies SCM overrides.
    for name in list(env):
        if name.startswith(("SETUPTOOLS_SCM_", "VCS_VERSIONING_", "GIT_")):
            env.pop(name)
    result = subprocess.run(
        args, cwd=root, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120, check=False,
    )
    return result


def _success(root: Path, *args: str) -> str:
    result = _run(root, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    """Copy current sources without installed or generated version metadata."""
    root = tmp_path / "source"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore", ".gitattributes"):
        shutil.copy2(REPO / name, root / name)
    shutil.copytree(
        REPO / "llm_conductor", root / "llm_conductor",
        ignore=shutil.ignore_patterns("__pycache__", "_version.py"),
    )
    return root


def _commit(root: Path) -> None:
    _success(root, "git", "add", ".")
    _success(
        root, "git", "-c", "user.name=Packaging Test",
        "-c", "user.email=packaging@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Test snapshot",
    )


@pytest.mark.parametrize("state", ["release", "dev", "dirty"])
def test_built_versions_match_metadata_without_runtime_scm(source_tree: Path, state: str):
    """Tags, later commits and dirty trees retain their version in a wheel."""
    _success(source_tree, "git", "init")
    _commit(source_tree)
    _success(source_tree, "git", "tag", "v2.4.6")
    if state != "release":
        with (source_tree / "README.md").open("a", encoding="utf-8") as stream:
            stream.write("\nDevelopment snapshot.\n")
        _commit(source_tree)
    if state == "dirty":
        with (source_tree / "README.md").open("a", encoding="utf-8") as stream:
            stream.write("\nUncommitted change.\n")

    args = [] if state == "release" else ["--wheel"]
    _success(source_tree, sys.executable, "-m", "build", "--no-isolation", *args)
    if state == "release":
        # The default build creates its wheel from this sdist, outside Git.
        with tarfile.open(next((source_tree / "dist").glob("*.tar.gz"))) as archive:
            names = archive.getnames()
        assert any(name.endswith("/llm_conductor/_version.py") for name in names)
        assert not any(".git" in Path(name).parts for name in names)

    installed = source_tree.parent / "installed"
    with zipfile.ZipFile(next((source_tree / "dist").glob("*.whl"))) as wheel:
        wheel.extractall(installed)
    # -S excludes site-packages: setuptools-scm cannot be a runtime dependency.
    output = _success(
        source_tree.parent, sys.executable, "-I", "-S", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from llm_conductor import __version__; "
        "from importlib.metadata import version; "
        "assert __version__ == version('llm-conductor'); print(__version__)",
        str(installed),
    )
    if state == "release":
        assert output == "2.4.6"
    else:
        assert output.startswith("2.4.7.dev1+g")
        assert bool(re.search(r"\.d\d{8}$", output)) == (state == "dirty")


def test_uninstalled_checkout_reports_unknown_version(source_tree: Path):
    """A fresh checkout remains importable without Git or package metadata."""
    output = _success(
        source_tree, sys.executable, "-I", "-S", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from llm_conductor import __version__; print(__version__)",
        str(source_tree),
    )
    assert output == "0.0.0+unknown"


def test_build_without_scm_metadata_fails(source_tree: Path):
    """An unversioned source copy must not produce an apparent release."""
    result = _run(source_tree, sys.executable, "-m", "build", "--no-isolation", "--wheel")
    assert result.returncode != 0
    assert not list((source_tree / "dist").glob("*.whl"))
    assert "unable to detect version" in (result.stdout + result.stderr).lower()


def test_exe_spec_refreshes_stale_version_before_analysis(source_tree: Path):
    """Direct PyInstaller builds refresh metadata without an editable install."""
    _success(source_tree, "git", "init")
    _commit(source_tree)
    _success(source_tree, "git", "tag", "v2.4.6")
    generated = source_tree / "llm_conductor" / "_version.py"
    generated.write_text('__version__ = "0.0.0+stale"\n', encoding="utf-8")

    class AnalysisReached(Exception):
        pass

    def analysis(*args, **kwargs):
        raise AnalysisReached

    # Stop before actual bundling; the real spec must update the fixture even
    # though this process's cwd is a different checkout.
    with pytest.raises(AnalysisReached):
        runpy.run_path(
            str(REPO / "llm-conductor.spec"),
            init_globals={"SPECPATH": str(source_tree), "Analysis": analysis},
        )
    assert runpy.run_path(str(generated))["__version__"] == "2.4.6"
