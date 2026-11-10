"""Exercise mutation classification with real, isolated pytest subprocesses."""
import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    # Mutation copies deliberately omit tools; meta tests are deselected there.
    path = Path(__file__).resolve().parents[2] / "tools" / "mutation_check.py"
    spec = importlib.util.spec_from_file_location("mutation_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("test_source", "expected"), [
    ("import subject\ndef test_value():\n    assert subject.VALUE == 1\n", "KILLED"),
    ("import subject\ndef test_value():\n    assert subject.VALUE > 0\n", "SURVIVED"),
    ("import missing_synthetic_dependency\n", "HARNESS_ERROR"),
    ("def test_import():\n    import missing_synthetic_dependency\n", "HARNESS_ERROR"),
    (("import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError('broken setup')\n"
     "def test_setup(broken):\n    assert False\n"), "HARNESS_ERROR"),
    ("", "HARNESS_ERROR"),
])
def test_real_pytest_outcome(runner, tmp_path, test_source, expected):
    source = tmp_path / "subject.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\npythonpath = .\n", encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(test_source, encoding="utf-8")
    status, detail = runner._run_mutant(tmp_path, "subject.py", "VALUE = 1", "VALUE = 2")
    assert status == expected
    if expected != "SURVIVED":
        assert detail
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("pytest", 300), OSError("cannot launch pytest")])
def test_runner_failure_restores_source(runner, tmp_path, monkeypatch, failure):
    source = tmp_path / "subject.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runner.subprocess, "run", fail)
    status, detail = runner._run_mutant(tmp_path, "subject.py", "VALUE = 1", "VALUE = 2")
    assert status == "HARNESS_ERROR"
    assert detail
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_harness_failure_fails_summary_even_for_equivalent_mutant(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "MUTANTS", [("example", "subject.py", "old", "new", "synthetic", True)])
    monkeypatch.setattr(runner, "_prepare_copy", lambda work: None)
    monkeypatch.setattr(runner, "_run_mutant", lambda *args: ("HARNESS_ERROR", "collection failed"))
    assert runner.main(set()) == 1
    output = capsys.readouterr().out
    assert "HARNESS_ERROR" in output
    assert "0/1" in output


@pytest.mark.parametrize("report_text", [None, "not XML", "<testsuites><testsuite/></testsuites>"])
def test_missing_or_invalid_verdict_is_not_a_kill(runner, tmp_path, monkeypatch, report_text):
    source = tmp_path / "subject.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    def run(command, **kwargs):
        if report_text is not None:
            Path(command[-1]).write_text(report_text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "synthetic runner error", "diagnostic")

    monkeypatch.setattr(runner.subprocess, "run", run)
    status, detail = runner._run_mutant(tmp_path, "subject.py", "VALUE = 1", "VALUE = 2")
    assert status == "HARNESS_ERROR"
    assert "diagnostic" in detail
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
