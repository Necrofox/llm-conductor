# -*- coding: utf-8 -*-
"""
System tests for stage-local resume (Issue #4 MVP): SIGINT flush, --continue,
fingerprint hard-fail, and degraded leaves staying out of the index.
"""
import json
import signal

from llm_conductor.constants import (
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    RESUME_SCHEMA_VERSION,
)
from llm_conductor.core import shutdown
from tests.system.harness import TRANSLATE_PROMPT

RUN = ("run", "--target", "sample.json")
TRANSLATE_NO_CACHE = (*RUN, "--stage", "translate", "--no-cache")


def _interrupt_on(model, call_number: int):
    def interrupt(n):
        if n == call_number:
            shutdown.signal_handler(signal.SIGINT, None)

    model.on_model_call = interrupt


def _reset_shutdown():
    """Allow a second run_main in the same test after an interrupt."""
    manager = shutdown._shutdown_manager
    with manager._lock:
        manager._shutdown_requested = False
        manager._shutdown_in_progress = False
        manager._config_instance = None
        manager._progress_flush = None


def test_sigint_with_no_cache_writes_partial_stage_and_resume_index(
        workspace, model, run_main):
    _interrupt_on(model, 3)

    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    assert len(model.model_calls) == 3
    assert workspace.exists(workspace.result("translated"))
    assert not workspace.exists(workspace.result("final"))
    resume_path = workspace.result("resume/translated")
    assert workspace.exists(resume_path)
    index = workspace.read_json(resume_path)
    assert index["schema_version"] == RESUME_SCHEMA_VERSION
    assert index["stage"] == "translated"
    assert len(index["leaves"]) == 2
    translated = workspace.read_json(workspace.result("translated"))
    # Accepted leaves are overlaid; the interrupted leaf stays English.
    assert translated["ui"]["title"].startswith("RU(")
    assert translated["ui"]["gold_counter"].startswith("RU(")


def test_continue_no_cache_skips_accepted_leaves(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED
    first_calls = len(model.model_calls)
    assert first_calls == 3

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE, "--continue") == EXIT_SUCCESS

    # SAMPLE has 7 eligible string leaves; two reused from resume → 5 API calls.
    assert first_calls == 3
    assert len(model.model_calls) == 5
    final = workspace.read_json(workspace.result("translated"))
    assert final["ui"]["title"] == "RU(Welcome to the Guild)"
    assert final["ui"]["gold_counter"] == "RU(You have {0} gold coins.)"
    assert final["options"][0] == "RU(Easy)"


def test_without_continue_does_not_reuse_resume_index(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_SUCCESS
    # Full rewalk — no silent auto-resume.
    assert len(model.model_calls) == 7


def test_continue_fingerprint_mismatch_refuses_stale_index(workspace, model, run_main):
    _interrupt_on(model, 3)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    model.on_model_call = None
    model.requests.clear()
    _reset_shutdown()
    # Prompt change → run_fingerprint mismatch.
    workspace.write_configs(
        translation__prompts__style=TRANSLATE_PROMPT + " (revised)",
    )

    code = run_main(*TRANSLATE_NO_CACHE, "--continue")
    assert code == EXIT_FAILURE_RATIO_EXCEEDED
    log = workspace.log_text()
    assert "fingerprint does not match" in log
    assert len(model.model_calls) == 0


def test_continue_retries_degraded_leaf_not_in_index(workspace, model, run_main):
    model.reject_leaf_substrings = ("Welcome to the Guild",)
    _interrupt_on(model, 4)
    assert run_main(*TRANSLATE_NO_CACHE) == EXIT_INTERRUPTED

    resume = workspace.read_json(workspace.result("resume/translated"))
    title_key = json.dumps(["ui", "title"], separators=(",", ":"))
    assert title_key not in resume["leaves"]
    # Call 1 rejected title; calls 2–3 accepted; call 4 interrupted.
    assert len(resume["leaves"]) == 2

    model.on_model_call = None
    model.reject_leaf_substrings = ()
    model.requests.clear()
    _reset_shutdown()

    assert run_main(*TRANSLATE_NO_CACHE, "--continue") == EXIT_SUCCESS
    title_calls = [
        body for body in model.model_calls
        if "Welcome to the Guild" in body["messages"][-1]["content"]
    ]
    assert len(title_calls) == 1
    translated = workspace.read_json(workspace.result("translated"))
    assert translated["ui"]["title"] == "RU(Welcome to the Guild)"
