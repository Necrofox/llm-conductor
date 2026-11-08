# -*- coding: utf-8 -*-
"""
Fixtures of the system tier: main() of llm_conductor.cli.main end to end.

main() runs in-process: sys.argv is replaced, HTTP is intercepted by
responses, and the global side effects of main() - signal.signal and
atexit.register - are captured so they do not leak into pytest itself.
LLMC_CONFIG points at the temporary directory, so a run never reads
configuration files next to the real entry point.
"""
import pytest
import responses

import llm_conductor.cli.main as entry_module
from llm_conductor.config import Config
from llm_conductor.constants import CONDUCTOR_CONFIG_FILENAME, CONFIG_PATH_ENV_VAR
from tests.support import OLLAMA_URL
from tests.system.harness import SAMPLE, EchoModel, Workspace


@pytest.fixture
def model():
    """EchoModel attached to OLLAMA_URL through responses."""
    echo = EchoModel()
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.add_callback(responses.POST, OLLAMA_URL, callback=echo)
        yield echo


@pytest.fixture
def preserve_logs(monkeypatch):
    """
    Disable the clean-run log auto-delete so a test can read the log file after main() returns.

    A run that logs no ERROR deletes its own log file at the end (close-of-run
    cleanup, see ConfigLogger.close_log_files()); tests whose actual concern
    is something else logged along the way (file naming, log level, glossary
    origin...) request this fixture instead of forcing an artificial failure
    just to keep the file around. The delete behavior itself is covered by
    tests/integration/test_config_files.py.
    """
    monkeypatch.setattr(Config, "close_log_files", lambda self, always_keep=False: None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """tmp_path as the cwd, with conductor.toml, llm.toml and sample.json."""
    monkeypatch.chdir(tmp_path)
    ws = Workspace(tmp_path)
    ws.write_configs()
    ws.write_json("sample.json", SAMPLE)
    return ws


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """
    Run main() with the given command line and return its exit code.

    The captured signal/atexit registrations are available as
    run_main.signals / run_main.atexit_handlers.
    """
    monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(tmp_path / CONDUCTOR_CONFIG_FILENAME))

    signals = {}
    atexit_handlers = []
    monkeypatch.setattr(entry_module.signal, "signal",
                        lambda signum, handler: signals.__setitem__(signum, handler))
    monkeypatch.setattr(entry_module.atexit, "register", atexit_handlers.append)

    def _run(*args) -> int:
        monkeypatch.setattr("sys.argv", ["llm-conductor", *args])
        try:
            entry_module.main()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        return 0

    _run.signals = signals
    _run.atexit_handlers = atexit_handlers
    return _run
