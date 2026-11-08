# -*- coding: utf-8 -*-
"""
System tests: main() of llm-conductor.py end to end, in-process, with the
fake model (EchoModel) behind responses. They cover the run command
contract and exit codes (doc/cli.md) without real inference.
"""
import signal

import pytest
import responses

from llm_conductor.cli.runner import FilePipeline
from llm_conductor.constants import (
    CACHE_LOCK_FILENAME,
    EXIT_API_UNAVAILABLE,
    EXIT_CONFIG_ERROR,
    EXIT_CONSECUTIVE_FAILURES,
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERNAL_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    EXIT_TARGET_LOCKED,
    EXIT_USAGE_ERROR,
)
from llm_conductor.core import shutdown
from llm_conductor.core.locking import FileLock
from tests.support import write_configs
from tests.system.harness import (
    SAMPLE,
    STRICT_PROMPT,
    TRANSLATE_PROMPT,
    VERIFY_PROMPT,
    base_docs,
    transform_strings,
)

TRANSLATED = transform_strings(SAMPLE, lambda s: f"RU({s})")
FINAL = transform_strings(SAMPLE, lambda s: f"OK(RU({s}))")
RUN = ("run", "--target", "sample.json")


# ============================================================================
# Stages and modes
# ============================================================================

def test_full_pipeline_writes_every_stage_under_result(workspace, model, run_main):
    assert run_main(*RUN) == EXIT_SUCCESS

    assert workspace.read_json(workspace.result("translated")) == TRANSLATED
    assert workspace.read_json(workspace.result("redacted")) == FINAL
    assert workspace.read_json(workspace.result("final")) == FINAL
    assert workspace.exists("cache/translation_cache.json")
    assert workspace.exists("cache/verification_cache.json")
    assert not workspace.exists("sample_final.json")


def test_final_output_preserves_key_order_and_tags(workspace, model, run_main):
    run_main(*RUN)

    final = workspace.read_json(workspace.result("final"))
    assert list(final) == list(SAMPLE)
    assert list(final["ui"]) == list(SAMPLE["ui"])
    assert final["ui"]["warning"] == "OK(RU(<b>Warning:</b> this action cannot be undone.))"
    assert final["ui"]["multiline"] == "OK(RU(First line.\nSecond line.))"


def test_model_never_sees_raw_tags(workspace, model, run_main):
    run_main(*RUN)

    for payload in model.model_calls:
        content = payload["messages"][-1]["content"]
        assert "<b>" not in content and "{0}" not in content and "\n" not in content.split("the text: ")[-1]


def test_gender_hints_reach_the_model(workspace, model, run_main):
    run_main(*RUN, "--stage", "translate")

    assert len(model.calls_with("MASCULINE")) == 1
    assert len(model.calls_with("FEMININE")) == 1


def test_stage_translate_writes_final_from_translation(workspace, model, run_main):
    assert run_main(*RUN, "--stage", "translate") == EXIT_SUCCESS

    assert workspace.read_json(workspace.result("final")) == TRANSLATED
    assert not workspace.exists(workspace.result("redacted"))
    assert model.calls_with(VERIFY_PROMPT) == []
    assert workspace.exists("cache/translation_cache.json")
    assert not workspace.exists("cache/verification_cache.json")


def test_stage_verify_reads_the_translated_result_of_an_earlier_run(workspace, model, run_main):
    workspace.write_json(workspace.result("translated"), TRANSLATED)

    assert run_main(*RUN, "--stage", "verify") == EXIT_SUCCESS

    assert workspace.read_json(workspace.result("final")) == FINAL
    assert model.calls_with(TRANSLATE_PROMPT) == []


def test_stage_verify_without_translated_result_fails_the_file(workspace, model, run_main):
    assert run_main(*RUN, "--stage", "verify") == EXIT_FAILURE_RATIO_EXCEEDED

    assert not workspace.exists(workspace.result("final"))
    assert model.model_calls == []


@pytest.mark.parametrize("flag, prompt", [("--strict", STRICT_PROMPT), (None, TRANSLATE_PROMPT)])
def test_translation_mode_selects_prompt(workspace, model, run_main, flag, prompt):
    run_main(*[a for a in (*RUN, "--stage", "translate", flag) if a])

    translation_calls = [p for p in model.model_calls if VERIFY_PROMPT not in p["messages"][-1]["content"]]
    assert translation_calls
    assert all(prompt in p["messages"][-1]["content"] for p in translation_calls)


def test_stage_translate_reports_tokens(workspace, model, run_main, capsys):
    run_main(*RUN, "--stage", "translate")

    assert "TOKEN CONSUMPTION SUMMARY" in capsys.readouterr().out


# ============================================================================
# Command line
# ============================================================================

@pytest.mark.parametrize("argv", [
    ("run",),                                             # --target is required
    ("run", "sample.json"),                               # positional input file removed
    (*RUN, "--translate-only"),                           # replaced by --stage
    (*RUN, "--verify-only"),
    (*RUN, "--stage", "both"),
    (*RUN, "--translation-model", "x"),                   # model overrides live in profiles
    (*RUN, "--verb", "LOUD"),
], ids=lambda argv: " ".join(argv[1:]) or "no-target")
def test_invalid_command_lines_are_usage_errors(workspace, model, run_main, argv):
    assert run_main(*argv) == EXIT_USAGE_ERROR
    assert model.requests == []


def test_no_subcommand_prints_help(workspace, model, run_main, capsys):
    assert run_main() == EXIT_USAGE_ERROR
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("verb, level_line_logged", [("debug", True), ("warning", False)])
def test_verb_overrides_the_configured_log_level(workspace, model, run_main, preserve_logs, verb, level_line_logged):
    """conductor.toml says INFO; --verb wins in both directions."""
    run_main(*RUN, "--stage", "translate", "--verb", verb)

    log = workspace.log_text()
    assert (" - DEBUG - " in log) is (verb == "debug")
    assert (f"Log level: {verb.upper()}" in log) is level_line_logged   # an INFO line


def test_log_file_is_named_after_the_target(workspace, model, run_main, preserve_logs):
    run_main(*RUN, "--stage", "translate")

    (log,) = workspace.logs()
    assert log.name.startswith("conductor_sample.json_")


def test_llm_config_flag_switches_the_whole_profile(workspace, model, run_main):
    llm_doc, _ = base_docs(translation__api__model="t-profile", verification__api__model="v-profile",
                           meta__name="Other profile")
    write_configs(workspace.root / "profiles", llm_doc, {})
    (workspace.root / "profiles" / "llm.toml").rename(workspace.root / "profiles" / "other.toml")

    assert run_main(*RUN, "--llm-config", "profiles/other.toml") == EXIT_SUCCESS

    assert {p["model"] for p in model.calls_with(TRANSLATE_PROMPT)} == {"t-profile"}
    assert {p["model"] for p in model.calls_with(VERIFY_PROMPT)} == {"v-profile"}


def test_banner_names_the_profile(workspace, model, run_main, capsys):
    workspace.write_configs(meta__name="Nightly qwen")

    run_main(*RUN, "--stage", "translate")

    assert "Nightly qwen" in capsys.readouterr().out


# ============================================================================
# Cache
# ============================================================================

def test_second_run_is_served_entirely_from_cache(workspace, model, run_main):
    run_main(*RUN)
    first_calls = len(model.model_calls)

    assert run_main(*RUN) == EXIT_SUCCESS

    assert len(model.model_calls) == first_calls
    assert workspace.read_json(workspace.result("final")) == FINAL


def test_cached_rerun_of_rejected_leaves_explains_failure_ratio(workspace, model, run_main):
    """
    Rejected replies are never cached. A first run with one reject among many
    leaves stays under max_failure_ratio (exit 0); the cache rerun re-queries
    only that residual miss, exceeds the ratio (exit 5), and warns that the
    denominator is this run's requests only.
    """
    model.reject_leaf_substrings = ("Welcome to the Guild",)

    assert run_main(*RUN) == EXIT_SUCCESS
    first_calls = len(model.model_calls)

    assert run_main(*RUN) == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(model.model_calls) > first_calls

    log = workspace.log_text()
    assert "exceeds the [resilience].max_failure_ratio" in log
    assert "cache misses" in log
    assert "prior validation rejects" in log
    assert "--no-cache" in log


def test_model_change_invalidates_cache(workspace, model, run_main):
    run_main(*RUN, "--stage", "translate")
    first_calls = len(model.model_calls)

    workspace.write_configs(translation__api__model="another-model")
    run_main(*RUN, "--stage", "translate")

    new_calls = model.model_calls[first_calls:]
    assert new_calls and all(p["model"] == "another-model" for p in new_calls)


def test_no_cache_neither_reads_nor_writes_the_disk_cache(workspace, model, run_main):
    run_main(*RUN, "--stage", "translate")                 # a warm cache on disk
    cached_calls = len(model.model_calls)

    assert run_main(*RUN, "--stage", "translate", "--no-cache") == EXIT_SUCCESS

    assert len(model.model_calls) == 2 * cached_calls      # the warm cache was not read
    assert workspace.read_json(workspace.result("final")) == TRANSLATED


def test_no_cache_still_writes_heartbeat_without_cache_files(workspace, model, run_main):
    """Heartbeat uses cache_dir even when the disk cache is off; the lock is not taken."""
    assert run_main(*RUN, "--no-cache") == EXIT_SUCCESS

    assert workspace.exists("cache/progress_state.json")
    assert not workspace.exists("cache/translation_cache.json")
    assert not workspace.exists("cache/verification_cache.json")
    assert not workspace.exists(f"cache/{CACHE_LOCK_FILENAME}")
    snap = workspace.read_json("cache/progress_state.json")
    assert snap["status"] == "completed"
    assert snap["exit_code"] == EXIT_SUCCESS


def test_cache2target_keeps_the_cache_next_to_the_target(workspace, model, run_main):
    workspace.write_json("project/sample.json", SAMPLE)

    assert run_main("run", "--target", "project/sample.json", "--stage", "translate", "--cache2target") == 0

    assert workspace.exists("project/cache/translation_cache.json")
    assert not workspace.exists("cache")


# ============================================================================
# Result layout and locations
# ============================================================================

def test_result2target_writes_result_next_to_the_target(workspace, model, run_main):
    workspace.write_json("project/sample.json", SAMPLE)

    assert run_main("run", "--target", "project/sample.json", "--result2target") == EXIT_SUCCESS

    assert workspace.read_json("project/result/final/sample.json") == FINAL
    assert workspace.exists("project/result/translated/sample.json")
    assert not workspace.exists("result")


def test_cache_logs_and_results_live_next_to_the_config_not_the_cwd(workspace, model, run_main, preserve_logs,
                                                                     tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "sample.json").rename(work / "sample.json")
    monkeypatch.chdir(work)

    assert run_main(*RUN, "--stage", "translate") == EXIT_SUCCESS

    assert (tmp_path / "cache" / "translation_cache.json").exists()
    assert len(workspace.logs()) == 1
    assert workspace.read_json(workspace.result("final")) == TRANSLATED
    assert sorted(p.name for p in work.iterdir()) == ["sample.json"]


# ============================================================================
# Manifests
# ============================================================================

MENU = {"title": "Main menu"}
GUARD = {"line": "Halt!"}


@pytest.fixture
def project(workspace):
    """project/ with two inputs in nested directories and a manifest naming both."""
    workspace.write_json("project/menu/main.json", MENU)
    workspace.write_json("project/dialogs/npc/guard.json", GUARD)
    workspace.write_text("project/ui.list",
                         "# UI strings\n\n   menu/main.json   \ndialogs\\npc\\guard.json\n")
    return workspace


def test_manifest_results_mirror_the_nested_directories(project, model, run_main, preserve_logs):
    assert run_main("run", "--target", "project/ui.list") == EXIT_SUCCESS

    def final(relative):
        return project.read_json(project.result("final", relative, target="ui.list"))

    assert final("menu/main.json") == {"title": "OK(RU(Main menu))"}
    assert final("dialogs/npc/guard.json") == {"line": "OK(RU(Halt!))"}
    assert "File 2/2" in project.log_text()


def test_manifest_with_result2target_mirrors_next_to_the_manifest(project, model, run_main):
    assert run_main("run", "--target", "project/ui.list", "--result2target") == EXIT_SUCCESS

    assert project.read_json("project/result/final/menu/main.json") == {"title": "OK(RU(Main menu))"}


@pytest.mark.parametrize("bad_line", [
    "../outside.json",
    "menu/../../outside.json",
    "ABSOLUTE",
], ids=["parent", "parent-after-normalization", "absolute"])
def test_manifest_path_outside_its_directory_fails_before_anything_is_written(project, model, run_main,
                                                                               bad_line, capsys):
    outside = project.write_json("outside.json", {"secret": "Keep me"})
    line = str(outside) if bad_line == "ABSOLUTE" else bad_line
    project.write_text("project/ui.list", f"menu/main.json\n{line}\n")

    assert run_main("run", "--target", "project/ui.list", "--result2target") == EXIT_USAGE_ERROR

    assert "line 2" in capsys.readouterr().out
    assert model.requests == []
    assert not project.exists("result") and not project.exists("project/result")
    assert project.read_json("outside.json") == {"secret": "Keep me"}


def test_manifest_duplicate_after_normalization_is_an_error(project, model, run_main, capsys):
    project.write_text("project/ui.list", "menu/main.json\ndialogs\\..\\menu\\main.json\n")

    assert run_main("run", "--target", "project/ui.list") == EXIT_USAGE_ERROR

    assert "duplicate of line 1" in capsys.readouterr().out
    assert model.requests == []


def test_empty_manifest_is_an_error(project, model, run_main):
    project.write_text("project/ui.list", "# nothing yet\n\n")

    assert run_main("run", "--target", "project/ui.list") == EXIT_USAGE_ERROR


def test_input_inside_the_result_directory_is_rejected(project, model, run_main):
    project.write_json("project/result/final/menu/main.json", MENU)
    project.write_text("project/ui.list", "result/final/menu/main.json\n")

    assert run_main("run", "--target", "project/ui.list", "--result2target") == EXIT_USAGE_ERROR
    assert model.requests == []


@pytest.mark.parametrize("max_failure_ratio, expected", [(0.5, EXIT_SUCCESS), (0.0, EXIT_FAILURE_RATIO_EXCEEDED)])
def test_failed_file_does_not_stop_the_others(project, model, run_main, max_failure_ratio, expected):
    """One of two files is broken: 50% of files failed - an error only above the threshold."""
    project.write_configs(resilience__max_failure_ratio=max_failure_ratio)
    project.write_text("project/menu/main.json", "{broken")

    assert run_main("run", "--target", "project/ui.list") == expected

    assert project.read_json(project.result("final", "dialogs/npc/guard.json", target="ui.list"))
    assert not project.exists(project.result("final", "menu/main.json", target="ui.list"))
    assert "1 of 2 file(s) failed: " in project.log_text()


def test_missing_manifest_entry_is_a_failed_file(project, model, run_main):
    project.write_text("project/ui.list", "menu/main.json\nmenu/missing.json\n")

    assert run_main("run", "--target", "project/ui.list") == EXIT_SUCCESS

    assert project.exists(project.result("final", "menu/main.json", target="ui.list"))
    assert "missing.json" in project.log_text()
    snap = project.read_json("cache/progress_state.json")
    failed = {(item["path"].replace("\\", "/"), item["status"]) for item in snap["files"]["failed"]}
    done = {path.replace("\\", "/") for path in snap["files"]["done"]}
    assert ("menu/missing.json", "failed") in failed
    assert "menu/main.json" in done


def test_circuit_breaker_stops_the_remaining_files(project, model, run_main, no_sleep):
    project.write_configs(resilience__max_consecutive_failures=1)
    model.fail_model_calls = True

    assert run_main("run", "--target", "project/ui.list") == EXIT_CONSECUTIVE_FAILURES

    assert project.exists(project.result("final", "menu/main.json", target="ui.list"))
    assert not project.exists(project.result("final", "dialogs/npc/guard.json", target="ui.list"))
    assert "1 remaining file(s) not processed" in project.log_text()

    snap = project.read_json("cache/progress_state.json")
    failed = {(item["path"].replace("\\", "/"), item["status"]) for item in snap["files"]["failed"]}
    pending = {path.replace("\\", "/") for path in snap["files"]["pending"]}
    assert ("menu/main.json", "aborted") in failed
    assert snap["files"]["done"] == []
    assert "dialogs/npc/guard.json" in pending
    assert snap["exit_code"] == EXIT_CONSECUTIVE_FAILURES
    assert any(item["code"] == EXIT_CONSECUTIVE_FAILURES for item in snap["problems"])


def test_crashed_file_is_recorded_in_heartbeat_and_does_not_stop_others(
        project, model, run_main, monkeypatch):
    """Unexpected error in one file → status crashed, exit 1; the rest still run."""
    real_process = FilePipeline._process

    def boom(self, entry):
        if entry.relative_path.replace("\\", "/") == "menu/main.json":
            raise RuntimeError("injected crash")
        return real_process(self, entry)

    monkeypatch.setattr(FilePipeline, "_process", boom)

    assert run_main("run", "--target", "project/ui.list") == EXIT_INTERNAL_ERROR

    assert not project.exists(project.result("final", "menu/main.json", target="ui.list"))
    assert project.exists(project.result("final", "dialogs/npc/guard.json", target="ui.list"))

    snap = project.read_json("cache/progress_state.json")
    failed = {(item["path"].replace("\\", "/"), item["status"]) for item in snap["files"]["failed"]}
    done = {path.replace("\\", "/") for path in snap["files"]["done"]}
    assert ("menu/main.json", "crashed") in failed
    assert "dialogs/npc/guard.json" in done
    assert snap["exit_code"] == EXIT_INTERNAL_ERROR
    assert any(item["code"] == EXIT_INTERNAL_ERROR for item in snap["problems"])


# ============================================================================
# Glossary
# ============================================================================

def test_glossary_next_to_the_target_reaches_the_prompt(workspace, model, run_main, preserve_logs):
    workspace.write_json("glossary.json", {"Guild": "Гильдия"})

    run_main(*RUN, "--stage", "translate")

    (call,) = model.calls_with("Guild = Гильдия")
    assert call["messages"][-1]["content"].endswith("Welcome to the Guild")
    assert "glossary.json (next to the target), 1 term(s)" in workspace.log_text()


def test_glossary_in_the_working_directory_is_not_used(workspace, model, run_main, preserve_logs):
    workspace.write_json("project/sample.json", SAMPLE)
    workspace.write_json("glossary.json", {"Guild": "Гильдия"})

    run_main("run", "--target", "project/sample.json", "--stage", "translate")

    assert model.calls_with("Guild = Гильдия") == []
    assert "working without terms" in workspace.log_text()


def test_glossary_option_replaces_the_implicit_glossary(workspace, model, run_main):
    workspace.write_json("glossary.json", {"Guild": "Гильдия"})
    workspace.write_json("terms/other.json", {"Guild": "Орден"})

    run_main(*RUN, "--stage", "translate", "--glossary", "terms/other.json")

    assert model.calls_with("Guild = Орден")
    assert model.calls_with("Guild = Гильдия") == []


@pytest.mark.parametrize("glossary", [None, "{broken", '["not", "an object"]'])
def test_unusable_explicit_glossary_is_a_usage_error(workspace, model, run_main, glossary):
    if glossary is not None:
        workspace.write_text("terms.json", glossary)

    assert run_main(*RUN, "--glossary", "terms.json") == EXIT_USAGE_ERROR
    assert model.requests == []


# ============================================================================
# Locks
# ============================================================================

def test_locked_target_exits_with_the_locked_code(workspace, model, run_main, capsys):
    with FileLock(str(workspace.root / "sample.json")):
        assert run_main(*RUN) == EXIT_TARGET_LOCKED

    assert "sample.json is locked by another run" in capsys.readouterr().out
    assert model.requests == []
    assert not workspace.exists("result")


def test_locked_cache_exits_with_the_locked_code(workspace, model, run_main, capsys):
    with FileLock(str(workspace.root / "cache" / CACHE_LOCK_FILENAME), create=True):
        assert run_main(*RUN) == EXIT_TARGET_LOCKED

    assert f"{CACHE_LOCK_FILENAME} is locked by another run" in capsys.readouterr().out
    assert model.requests == []


def test_locked_result_root_exits_with_the_locked_code(workspace, model, run_main, capsys):
    with FileLock(str(workspace.root / "result" / "sample.json" / CACHE_LOCK_FILENAME), create=True):
        assert run_main(*RUN) == EXIT_TARGET_LOCKED

    assert f"{CACHE_LOCK_FILENAME} is locked by another run" in capsys.readouterr().out
    assert model.requests == []


def test_no_cache_run_does_not_lock_the_cache(workspace, model, run_main):
    with FileLock(str(workspace.root / "cache" / CACHE_LOCK_FILENAME), create=True):
        assert run_main(*RUN, "--no-cache") == EXIT_SUCCESS


def test_locks_are_released_after_the_run(workspace, model, run_main):
    run_main(*RUN)

    with FileLock(str(workspace.root / "sample.json")), \
            FileLock(str(workspace.root / "cache" / CACHE_LOCK_FILENAME)), \
            FileLock(str(workspace.root / "result" / "sample.json" / CACHE_LOCK_FILENAME)):
        pass


def test_same_basename_targets_cannot_share_a_result_directory(workspace, model, run_main, capsys):
    workspace.write_json("a/strings.json", SAMPLE)
    workspace.write_json("b/strings.json", SAMPLE)

    assert run_main("run", "--target", "a/strings.json", "--stage", "translate") == EXIT_SUCCESS
    assert run_main("run", "--target", "b/strings.json", "--stage", "translate") == EXIT_USAGE_ERROR

    out = capsys.readouterr().out
    assert "belongs to" in out
    assert workspace.exists("result/strings.json/.llmc.target")


def test_result2target_same_basename_targets_do_not_collide(workspace, model, run_main):
    workspace.write_json("a/sample.json", SAMPLE)
    workspace.write_json("b/sample.json", SAMPLE)

    assert run_main("run", "--target", "a/sample.json", "--result2target",
                    "--stage", "translate") == EXIT_SUCCESS
    assert run_main("run", "--target", "b/sample.json", "--result2target",
                    "--stage", "translate") == EXIT_SUCCESS

    assert workspace.exists("a/result/final/sample.json")
    assert workspace.exists("b/result/final/sample.json")
    assert not workspace.exists("a/result/.llmc.target")
    assert not workspace.exists("b/result/.llmc.target")


# ============================================================================
# Startup failures
# ============================================================================

@pytest.mark.parametrize("missing", ["conductor.toml", "llm.toml"])
def test_missing_config_exits_with_readable_error(workspace, model, run_main, capsys, missing):
    (workspace.root / missing).unlink()

    assert run_main(*RUN) == EXIT_CONFIG_ERROR

    err = capsys.readouterr().err
    assert "Error:" in err and "not found" in err
    assert "Traceback" not in err


def test_invalid_config_exits_with_validation_message(workspace, model, run_main, capsys):
    workspace.write_configs(translation__api__type="openai")

    assert run_main(*RUN) == EXIT_CONFIG_ERROR
    assert "api_key is required" in capsys.readouterr().err


def test_unknown_config_key_exits_with_a_suggestion(workspace, model, run_main, capsys):
    workspace.write_configs(cache__max_sise=5)

    assert run_main(*RUN) == EXIT_CONFIG_ERROR
    err = capsys.readouterr().err
    assert "did you mean 'max_size'" in err and "Traceback" not in err
    assert model.requests == []


def test_unavailable_api_stops_before_processing(workspace, run_main):
    """responses without registered URLs: ConnectionError on preflight, no real network."""
    with responses.RequestsMock():
        assert run_main(*RUN) == EXIT_API_UNAVAILABLE

    # The shared result root is claimed under its lock before preflight, so
    # the owner marker may exist; no stage outputs must have been written.
    assert not workspace.exists(workspace.result("translated"))
    assert not workspace.exists(workspace.result("final"))


def test_preflight_can_be_switched_off(workspace, model, run_main):
    workspace.write_configs(preflight__check_availability=False)

    assert run_main(*RUN, "--stage", "translate") == EXIT_SUCCESS

    assert len(model.requests) == len(model.model_calls)        # no "test" probe was sent


def test_missing_target_is_a_usage_error(workspace, model, run_main, capsys):
    assert run_main("run", "--target", "absent.json") == EXIT_USAGE_ERROR

    assert "absent.json not found" in capsys.readouterr().out
    assert model.requests == []


def test_broken_input_file_fails_the_run(workspace, model, run_main):
    (workspace.root / "sample.json").write_text("{broken", encoding="utf-8")

    assert run_main(*RUN) == EXIT_FAILURE_RATIO_EXCEEDED
    assert model.model_calls == []


def test_check_validates_both_files_without_side_effects(workspace, model, run_main, capsys):
    assert run_main("check") == EXIT_SUCCESS

    out = capsys.readouterr().out
    assert "conductor.toml" in out and "llm.toml" in out
    assert not workspace.exists("cache") and not workspace.exists("logs")
    assert model.requests == []


# ============================================================================
# Model failures
# ============================================================================

@pytest.mark.characterization(reason="policy §4.3: a failed leaf degrades to its source text")
def test_total_model_failure_keeps_original_text_and_exits_with_failure_ratio_code(
    workspace, model, run_main, no_sleep
):
    """
    Degradation to the original (§4.3) and the failure ratio threshold
    together: the pipeline does not crash, the final result holds the
    source text, and 100% of failures above [resilience].max_failure_ratio
    (default 0.5) is not reported as success.
    """
    model.fail_model_calls = True

    assert run_main(*RUN) == EXIT_FAILURE_RATIO_EXCEEDED
    assert workspace.read_json(workspace.result("final")) == SAMPLE


def test_circuit_breaker_trips_and_exits_with_consecutive_failures_code(workspace, model, run_main, no_sleep):
    """
    A short streak of consecutive failures aborts the stage instead of
    grinding through every remaining leaf against a dead model. With the
    threshold lowered to 2, only the first 2 leaves are attempted (each
    retried up to max_retries=2 times per base_docs()) before the breaker
    trips and the rest of the tree is left untouched. The breaker code wins
    over the failure ratio code that the same run also earns.
    """
    workspace.write_configs(resilience__max_consecutive_failures=2)
    model.fail_model_calls = True

    assert run_main(*RUN) == EXIT_CONSECUTIVE_FAILURES
    assert len(model.model_calls) == 4
    assert workspace.read_json(workspace.result("final")) == SAMPLE
    assert "exceeds the [resilience].max_failure_ratio" in workspace.log_text()


def test_task34_failures_are_reported_in_api_metrics(workspace, model, run_main, no_sleep, capsys):
    model.fail_model_calls = True

    run_main(*RUN)

    out = capsys.readouterr().out
    assert "Translation API:" in out
    assert "failed" in out.lower()


# ============================================================================
# Signals and ShutdownManager registration
# ============================================================================

def test_signal_handlers_and_atexit_are_registered(workspace, model, run_main):
    run_main(*RUN)

    registered = set(run_main.signals)
    assert {signal.SIGINT, signal.SIGTERM}.issubset(registered)
    # POSIX file timeouts install SIGALRM (SignalTimeoutStrategy); Windows has
    # no SIGALRM, so CI Ubuntu saw an extra signal that Windows never did.
    extra = registered - {signal.SIGINT, signal.SIGTERM}
    if hasattr(signal, "SIGALRM"):
        extra.discard(signal.SIGALRM)
    assert extra == set()
    assert len(run_main.atexit_handlers) == 1


@pytest.mark.parametrize("stage", ["all", "translate"])
def test_run_unregisters_config_from_shutdown_manager(workspace, model, run_main, stage):
    run_main(*RUN, "--stage", stage)

    assert shutdown._shutdown_manager._config_instance is None


def test_sigint_mid_run_exits_130_after_saving_caches(workspace, model, run_main):
    """
    The signal arrives inside a model call, i.e. inside the `except Exception`
    of the degradation policy - RunInterrupted must pass through it.
    """
    def interrupt(call_number):
        if call_number == 3:
            shutdown.signal_handler(signal.SIGINT, None)

    model.on_model_call = interrupt

    assert run_main(*RUN) == EXIT_INTERRUPTED

    assert len(model.model_calls) == 3
    assert workspace.exists("cache/translation_cache.json")
    assert not workspace.exists(workspace.result("final"))
    snap = workspace.read_json("cache/progress_state.json")
    assert snap["status"] == "interrupted"
    assert snap["exit_code"] == EXIT_INTERRUPTED
    log = workspace.log_text()
    assert "WARNING - SIGINT received: run interrupted, saving caches" in log
    assert f"exiting with code {EXIT_INTERRUPTED}" in log
    with FileLock(str(workspace.root / "sample.json")):     # the lock was released on the way out
        pass


def test_heartbeat_progress_state_written_on_success(workspace, model, run_main):
    assert run_main(*RUN, "--stage", "translate") == EXIT_SUCCESS

    snap = workspace.read_json("cache/progress_state.json")
    assert snap["schema_version"] == 1
    assert snap["status"] == "completed"
    assert snap["exit_code"] == EXIT_SUCCESS
    assert snap["target"]["kind"] == "file"
    assert snap["stages"] == ["translation"]
    assert snap["files"]["done"] == ["sample.json"]
    assert snap["files"]["failed"] == []
    assert snap["files"]["pending"] == []
    assert snap["progress"]["percent"] == 100.0
    # Compact (no pretty indent) on disk
    raw = (workspace.root / "cache" / "progress_state.json").read_text(encoding="utf-8")
    assert "\n" not in raw


def test_task46_token_report_printed_once(workspace, model, run_main, capsys):
    run_main(*RUN)

    assert capsys.readouterr().out.count("TOKEN CONSUMPTION SUMMARY") == 1
