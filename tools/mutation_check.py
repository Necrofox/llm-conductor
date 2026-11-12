# -*- coding: utf-8 -*-
"""
Manual mutation testing of the test shield.

Each mutation is a plausible bug (swapped condition, dropped cache-key
component, reversed traversal order...). The script copies llm_conductor/, tests/,
the entry point and pytest.ini into a temp directory, applies mutations
one at a time and runs ``pytest -x``. A mutation is "killed" if at least
one test fails during its call; "survived" means the tests would have missed the defect.
Collection, setup/teardown, import and runner failures are HARNESS_ERROR, never kills.

Why: check that tests catch errors, not only confirm the claimed
behaviour. mutmut does not work natively on Windows, so mutations are
listed by hand.

Run from the repository root (~3 min, does not touch working files):
    python tools/mutation_check.py
    python tools/mutation_check.py key_no_model dedup_by_value   # selective

Exit code 1 — unexpected survivors or mutations whose pattern is no
longer found in the code (SKIP: code changed — update the list), or any
HARNESS_ERROR (including timeout); these do not count as successful kills.

Equivalent mutations (5th tuple element = True) do not change behaviour
and are expected to survive; the list is for the history of findings:
  - unmask_order: [T1] is not a prefix of [T10] (closing bracket), so
    replacement order in unmask_tags() has no effect — doc/pipeline.md;
  - coord_lt: snapshot version numbers are unique, equality is unreachable.
"""
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from llm_conductor.core.io import FileOperations, safe_read_file
from llm_conductor.exceptions import FileOperationError

REPO = Path(__file__).resolve().parents[1]
COPIED_DIRS = ("llm_conductor", "tests", "doc")
COPIED_FILES = ("llm-conductor.py", "pytest.ini")

# (id, file, old, new, description[, equivalent])
MUTANTS = [
    # (id, file, old, new, description)
    ("unmask_order", "llm_conductor/utils.py", "key=len, reverse=True", "key=len, reverse=False", "unmask: [T1]/[T10] replacement order", True),
    ("mask_counter", "llm_conductor/utils.py", 'marker = f"[{prefix}{counter}]"', 'marker = f"[{prefix}0]"', "mask: marker counter does not increment"),
    ("mask_prefix_ignores_collision", "llm_conductor/utils.py",
     "if not re.search(r'\\[' + re.escape(prefix) + r'\\d+\\]', text):",
     "if True:",
     "mask: prefix selection ignores collision with a literal"),
    ("original_text_after_label", "llm_conductor/l10n/processor.py",
     "return prompt[:sentence_start] + original_text_block + prompt[sentence_start:]",
     "return prompt + original_text_block",
     "verify: [ORIGINAL] block is inserted after the text label, not before it"),
    ("mask_no_newline", "llm_conductor/constants.py", r"|(?:\n)|", "|", "regex does not mask \\n"),
    ("content_joins_fragments", "llm_conductor/l10n/content.py", "parts.extend((core[previous:start], CONTENT_MASK_REPLACEMENT))", "parts.extend((core[previous:start], ''))", "content eligibility joins text across protection"),
    ("content_gate_disabled", "llm_conductor/l10n/content.py", "return self._required.search(self.eligibility_text(text)) is not None", "return True", "content eligibility ignores the required pattern"),
    ("custom_mask_ignored", "llm_conductor/l10n/processor.py", "masked_text, tag_map = mask_tags(core, rules)", "masked_text, tag_map = mask_tags(core)", "model masking ignores the custom content policy"),
    ("profile_content_ignored", "llm_conductor/l10n/processor.py", "rules = runtime_config.get('processing', self.config.processing)", "rules = self.config.processing", "manifest processing uses the bootstrap profile"),
    ("key_no_content", "llm_conductor/core/cache.py", "api_url, sorted(set(allowed_scripts)), content_identity,", "api_url, sorted(set(allowed_scripts)),", "cache key ignores content policy"),
    ("key_no_model", "llm_conductor/core/cache.py", "CACHE_KEY_VERSION, operation_type, model, system_prompt,\n            enhanced_prompt, mode, value, sampling_identity,", "CACHE_KEY_VERSION, operation_type, system_prompt,\n            enhanced_prompt, mode, value, sampling_identity,", "cache key without model"),
    ("key_no_prompt", "llm_conductor/core/cache.py", "CACHE_KEY_VERSION, operation_type, model, system_prompt,\n            enhanced_prompt, mode, value, sampling_identity,", "CACHE_KEY_VERSION, operation_type, model, system_prompt,\n            mode, value, sampling_identity,", "cache key without assembled prompt"),
    ("key_no_sampling", "llm_conductor/core/cache.py", "CACHE_KEY_VERSION, operation_type, model, system_prompt,\n            enhanced_prompt, mode, value, sampling_identity,", "CACHE_KEY_VERSION, operation_type, model, system_prompt,\n            enhanced_prompt, mode, value,", "cache key without sampling identity"),
    ("capacity_gt", "llm_conductor/core/cache.py", "while len(self.data) >= self.max_size:", "while len(self.data) > self.max_size:", "eviction: size becomes max_size+1"),
    ("evict_lifo", "llm_conductor/core/cache.py", "cache_key, entry = self.data.popitem(last=False)", "cache_key, entry = self.data.popitem(last=True)", "evicts the fresh entry instead of the oldest"),
    ("expiry_inverted", "llm_conductor/core/cache.py", "if current_time - v['timestamp'] < expiry_seconds:\n                            entry = v", "if current_time - v['timestamp'] > expiry_seconds:\n                            entry = v", "load: expiry inverted"),
    ("no_version_check", "llm_conductor/core/cache.py", "if entry.get('cache_key_version') != CACHE_KEY_VERSION:", "if False:", "load: old key format is accepted"),
    ("no_deepcopy", "llm_conductor/core/cache.py", "return copy.deepcopy(value)", "return value", "get: without deepcopy"),
    ("deindex_skip", "llm_conductor/core/cache.py", "        self.deindex_key(cache_key, value_hash)\n        return cache_key", "        return cache_key", "eviction does not clean value_index"),
    ("coord_lt", "llm_conductor/core/cache.py", "if version <= self._last_written_version:", "if version < self._last_written_version:", "coordinator: rewrite of the same version", True),
    ("glossary_lru_off", "llm_conductor/core/cache.py", "                self._cache.move_to_end(cache_key)\n                self._hits += 1", "                self._hits += 1", "glossary LRU becomes FIFO"),
    ("inherit_priority", "llm_conductor/l10n/context.py", "priority=max(hint.priority - 5, 1)", "priority=hint.priority", "inherited hint without priority demotion"),
    ("hints_sort_asc", "llm_conductor/l10n/context.py", "hints.sort(reverse=True)", "hints.sort()", "hints sorted by ascending priority"),
    ("ancestor_order", "llm_conductor/l10n/context.py", "for ancestor_path in reversed(ancestors):", "for ancestor_path in ancestors:", "inheritance from distant ancestor, not nearest"),
    ("replace_first_marker", "llm_conductor/l10n/processor.py", "last_pos = text.rfind(marker)", "last_pos = text.find(marker)", "replaces the first marker instead of the last"),
    ("sentence_off_by_one", "llm_conductor/l10n/processor.py", "return single_newline + 1  # Skip past \\n.", "return single_newline", "sentence start: off-by-one"),
    ("dedup_by_value", "llm_conductor/l10n/processor.py", "                if cache_key in self._dedup_index:\n                    _, translated = self._dedup_index[cache_key]", "                if value in self._dedup_index:\n                    _, translated = self._dedup_index[value]", "dedup by value (full-key regression)"),
    ("dedup_store_value", "llm_conductor/l10n/processor.py", "self._dedup_index[cache_key] = (path, unmasked_result)", "self._dedup_index[value] = (path, unmasked_result)", "dedup stored under the value"),
    ("verify_no_original", "llm_conductor/l10n/processor.py", "                original_value = original_data.get(key)", "                original_value = None", "verification loses the original in a dict"),
    ("list_reversed", "llm_conductor/l10n/processor.py", "            result.append(\n", "            result.insert(0,\n", "list traversal reverses order"),
    ("invalid_verify_masked", "llm_conductor/l10n/processor.py", "                self.config.metrics.record_rejection(operation_type)\n\n                if self.config.circuit_breaker.record_failure(operation_type):\n                    self._log_circuit_breaker_trip(operation_type)\n\n                with self._stats_lock:\n                    self.stats['failed_leaves'] += 1\n\n                error_kind = validation.reason or FAILURE_KIND_REJECTED\n                outcomes.record(\n                    leaf_path, operation_type, make_degraded(value, error_kind)\n                )\n                # Reclassify by attempt id so concurrent leaves cannot race\n                # on “last success” when validation rejects a paid reply.\n                if outcome.usage_attempt_id is not None:\n                    self.config.usage_ledger.mark_attempt_rejected(\n                        outcome.usage_attempt_id, failure_kind=error_kind,\n                    )\n                return value", "                self.config.metrics.record_rejection(operation_type)\n\n                if self.config.circuit_breaker.record_failure(operation_type):\n                    self._log_circuit_breaker_trip(operation_type)\n\n                with self._stats_lock:\n                    self.stats['failed_leaves'] += 1\n\n                error_kind = validation.reason or FAILURE_KIND_REJECTED\n                outcomes.record(\n                    leaf_path, operation_type, make_degraded(value, error_kind)\n                )\n                # Reclassify by attempt id so concurrent leaves cannot race\n                # on “last success” when validation rejects a paid reply.\n                if outcome.usage_attempt_id is not None:\n                    self.config.usage_ledger.mark_attempt_rejected(\n                        outcome.usage_attempt_id, failure_kind=error_kind,\n                    )\n                return outcome.text", "rejected verification returns masked text"),
    ("usage_reject_skipped", "llm_conductor/l10n/processor.py", "                if outcome.usage_attempt_id is not None:\n                    self.config.usage_ledger.mark_attempt_rejected(\n                        outcome.usage_attempt_id, failure_kind=error_kind,\n                    )\n", "", "validation reject leaves usage outcome as success"),
    ("cache_masked_value", "llm_conductor/l10n/processor.py", "operation_type, cache_key, unmasked_result, metadata, source_value=value", "operation_type, cache_key, outcome.text, metadata, source_value=value", "cache stores the masked result (§4.4 invariant)"),
    ("autosave_gt", "llm_conductor/l10n/processor.py", "should_save = self._leaves_since_save >= interval", "should_save = self._leaves_since_save > interval", "autosave at N+1"),
    ("openai_no_max_tokens", "llm_conductor/core/api_client.py", "openai_params[tokens_key] = sampling_params['num_predict']", "pass", "OpenAI: num_predict is not mapped"),
    ("ollama_tokens_swapped", "llm_conductor/core/api_client.py", "prompt_tokens = response_json.get('prompt_eval_count', 0) if has_prompt else 0\n        completion_tokens = response_json.get('eval_count', 0) if has_completion else 0", "prompt_tokens = response_json.get('eval_count', 0) if has_prompt else 0\n        completion_tokens = response_json.get('prompt_eval_count', 0) if has_completion else 0", "Ollama: token counters swapped"),
    ("backoff_linear", "llm_conductor/core/api_client.py", "            else:\n                wait_time = backoff_factor ** attempt\n", "            else:\n                wait_time = backoff_factor * attempt\n", "linear backoff"),
    ("no_strip", "llm_conductor/core/api_client.py", "result_value = parsed.text.strip()", "result_value = parsed.text", "model reply is not stripped"),
    ("openai_no_auth", "llm_conductor/core/api_client.py", "if api_type == 'openai' and isinstance(api_key, str) and api_key.strip():", "if False:", "no Bearer in the request"),
    ("metrics_no_total", "llm_conductor/core/metrics.py", "            metrics['total_requests'] += 1\n\n            if failed:", "            if failed:", "total_requests stops growing"),
    ("cleanup_guard", "llm_conductor/core/shutdown.py", "            if self._shutdown_requested:\n                return", "            if False:\n                return", "atexit after a signal saves again"),
    ("shutdown_one_cache", "llm_conductor/core/shutdown.py", "                    self._config_instance.cache_manager.save(OPERATION_TRANSLATION)\n                    self._config_instance.cache_manager.save(OPERATION_VERIFICATION)\n                except Exception as e:  # noqa: BLE001 - signal path must finish and raise RunInterrupted", "                    self._config_instance.cache_manager.save(OPERATION_TRANSLATION)\n                except Exception as e:  # noqa: BLE001 - signal path must finish and raise RunInterrupted", "signal saves only the translation cache"),
    ("openai_key_optional", "llm_conductor/config.py", "if self.api_type == 'openai' and not self.api_key:", "if False:", "openai without a key passes validation"),
    ("strict_style_swap", "llm_conductor/config.py", "if mode == MODE_STRICT:\n            trans_config = self.config_data.translation_strict", "if mode != MODE_STRICT:\n            trans_config = self.config_data.translation_strict", "strict/style modes swapped"),
    ("num_ctx_unchecked", "llm_conductor/cli/validators.py", "if 'num_ctx' in runtime_config and runtime_config['num_ctx'] <= 0:", "if False:", "num_ctx <= 0 is skipped"),
    ("atomic_no_cleanup", "llm_conductor/core/io.py", "                if os.path.exists(temp_path):\n                    try:\n                        os.unlink(temp_path)", "                if False:\n                    try:\n                        os.unlink(temp_path)", "failed write leaves a .tmp behind"),
    ("depth_off", "llm_conductor/l10n/processor.py", "if current_depth > self.max_recursion_depth:", "if current_depth > self.max_recursion_depth + 1:", "depth limit is 1 too high"),
    ("unmask_silent_debug", "llm_conductor/utils.py", "if missing_markers and config.is_debug_mode:", "if False:", "lost markers are not logged"),
    ("glossary_by_translation", "llm_conductor/l10n/processor.py", "glossary_string = build_glossary_string(original_data, glossary, self.config)", "glossary_string = build_glossary_string(value, glossary, self.config)", "verification glossary keyed off the translation"),
    # Configuration layer: strict TOML schema, key lookup, path anchoring, init.
    ("schema_unknown_ignored", "llm_conductor/schema.py", "errors.append(f\"unknown key", "(f\"unknown key", "unknown TOML keys are silently accepted"),
    ("schema_bool_as_int", "llm_conductor/schema.py", "        if isinstance(value, bool):\n            return False\n", "", "true/false accepted where a number is expected"),
    ("key_env_ignored", "llm_conductor/config.py", "api_key = env_key if env_key.strip() else api[\"key\"]", "api_key = api[\"key\"]", "key_env is never consulted"),
    ("dir_not_anchored", "llm_conductor/config.py", "return os.path.normpath(os.path.join(base_dir, path))", "return path", "cache/log dirs resolved from the cwd again"),
    ("glossary_size_unchecked", "llm_conductor/config.py", "if self.glossary_patterns_cache_max_size <= 0:", "if False:", "glossary_patterns_max_size = 0 passes validation"),
    ("limits_timeout_ignored", "llm_conductor/config.py", "            timeout=config_data.limits.file_operation_timeout,\n", "", "[limits].file_operation_timeout has no effect"),
    ("llm_next_to_app_dir", "llm_conductor/cli/locations.py", "else os.path.join(os.path.dirname(conductor_path), LLM_CONFIG_FILENAME))", "else os.path.join(app_dir, LLM_CONFIG_FILENAME))", "default llm.toml ignores LLMC_CONFIG"),
    ("init_overwrites", "llm_conductor/cli/init.py", "    if existing:\n        raise", "    if False:\n        raise", "init overwrites an existing profile"),
    # The run command: stages, cache switches, manifests, locks, exit codes.
    ("preflight_always", "llm_conductor/cli/runner.py", "        if not config.preflight.check_availability:", "        if False:", "[preflight].check_availability is ignored"),
    ("stage_verify_uses_source", "llm_conductor/cli/runner.py", "                data = self._read_json(translated_path, \"translated data\")", "                data = source", "--stage verify proofreads the English source"),
    ("first_stage_cache_unsaved", "llm_conductor/cli/runner.py", "        for operation_type in self._union_operations():\n            self._config.cache_manager.save(operation_type)", "        for operation_type in self._union_operations()[1:]:\n            self._config.cache_manager.save(operation_type)", "the cache of the first stage is not saved"),
    ("file_failure_stops_run", "llm_conductor/cli/runner.py", "            logger.error(f\"{entry.relative_path}: {e}\")\n            return FileOutcome(entry, FileStatus.FAILED)", "            raise", "a failed file aborts the whole run"),
    ("file_ratio_ignored", "llm_conductor/cli/runner.py", "            if ratio > threshold:", "            if False:", "failed files never fail the run"),
    ("cache_not_locked", "llm_conductor/cli/runner.py", "        if self._config.cache_enabled:\n            locks.enter_context(", "        if False:\n            locks.enter_context(", "the cache directory is not locked"),
    ("no_cache_ignored", "llm_conductor/cli/main.py", "cache_enabled=False if args.no_cache else None,", "cache_enabled=None,", "--no-cache has no effect"),
    ("disabled_cache_saved", "llm_conductor/core/cache.py", "        if not self.persistent:\n            return True\n\n        cache_filename", "        cache_filename", "a disabled cache is still written"),
    ("glossary_from_cwd", "llm_conductor/cli/glossary.py", "candidate = os.path.join(target.directory, GLOSSARY_FILENAME)", "candidate = GLOSSARY_FILENAME", "the implicit glossary is read from the cwd again"),
    ("manifest_escape", "llm_conductor/cli/paths.py", "    if source_path == base_dir or not is_inside(source_path, base_dir):", "    if source_path == base_dir:", "manifest paths may leave the manifest directory"),
    ("manifest_absolute", "llm_conductor/cli/paths.py", "    if drive or MANIFEST_DRIVE_PATTERN.match(portable) or os.path.isabs(portable) or portable.startswith('/'):", "    if False:", "absolute manifest paths are accepted"),
    ("manifest_duplicates", "llm_conductor/cli/run_manifest.py", "            if original_key in first_original:", "            if False:", "duplicate manifest entries are accepted"),
    ("inside_by_prefix", "llm_conductor/cli/paths.py", "        return os.path.commonpath([path_key, dir_key]) == dir_key", "        return path_key.startswith(dir_key)", "a common string prefix counts as containment"),
    ("result_overlap_unchecked", "llm_conductor/cli/main.py", "    layout.check_disjoint(target)\n", "", "inputs inside result/ may be overwritten"),
    ("lock_over_content", "llm_conductor/constants.py", "LOCK_REGION_OFFSET = 0x7FFFFFF0", "LOCK_REGION_OFFSET = 0", "the Windows lock covers the file content (POSIX: equivalent)"),
    ("interrupt_is_exception", "llm_conductor/exceptions.py", "class RunInterrupted(BaseException):", "class RunInterrupted(Exception):", "Ctrl+C is swallowed by the degradation `except Exception`"),
    ("interrupt_exit_zero", "llm_conductor/cli/main.py", "        code = EXIT_INTERRUPTED\n    except Exception", "        code = EXIT_SUCCESS\n    except Exception", "an interrupted run exits with 0"),
    ("ratio_beats_breaker", "llm_conductor/constants.py", "    EXIT_CONSECUTIVE_FAILURES,\n    EXIT_FAILURE_RATIO_EXCEEDED,", "    EXIT_FAILURE_RATIO_EXCEEDED,\n    EXIT_CONSECUTIVE_FAILURES,", "exit code priority: 5 wins over 6"),
]


def _prepare_copy(work: Path) -> None:
    for name in COPIED_DIRS:
        shutil.copytree(REPO / name, work / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in COPIED_FILES:
        shutil.copy2(REPO / name, work / name)


def _run_mutant(work: Path, rel: str, old: str, new: str):
    """Return (status, first failed check)."""
    path = work / rel
    original = path.read_text(encoding="utf-8")
    if original.count(old) != 1:
        return "SKIP", f"pattern found {original.count(old)} time(s)"
    path.write_text(original.replace(old, new), encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="llmc-pytest-") as report_dir:
            report_path = Path(report_dir) / "results.xml"
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                     "-m", "not live and not meta", "--junitxml", str(report_path)],
                    cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=300, check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                return "HARNESS_ERROR", str(exc)
            if result.returncode == 0:
                return "SURVIVED", ""
            diagnostics = f"pytest exit {result.returncode}:\n{result.stdout}\n{result.stderr}".strip()
            if result.returncode != 1:
                return "HARNESS_ERROR", diagnostics
            try:
                report = ET.fromstring(safe_read_file(
                    str(report_path), "pytest report", FileOperations(), as_text=True,
                ))
            except (ET.ParseError, FileOperationError) as exc:
                return "HARNESS_ERROR", f"{exc}\n{diagnostics}"
            failures = report.findall(".//failure")
            if (report.find(".//error") is not None or not failures
                    or any(failure.get("message", "").startswith(("ImportError:", "ModuleNotFoundError:"))
                           for failure in failures)):
                return "HARNESS_ERROR", diagnostics
            failed = next((ln[len("FAILED "):] for ln in result.stdout.splitlines()
                           if ln.startswith("FAILED ")), failures[0].get("message", "test failure"))
            return "KILLED", failed.split(" - ")[0]
    finally:
        path.write_text(original, encoding="utf-8")


def main(selected) -> int:
    mutants = [m for m in MUTANTS if not selected or m[0] in selected]
    unexpected = []
    with tempfile.TemporaryDirectory(prefix="llmc-mutants-") as tmp:
        work = Path(tmp)
        _prepare_copy(work)
        for mutant_id, rel, old, new, description, *rest in mutants:
            equivalent = bool(rest and rest[0])
            status, detail = _run_mutant(work, rel, old, new)
            label = status + (" (equivalent)" if equivalent and status == "SURVIVED" else "")
            print(f"{label:26} {mutant_id:24} {description}  {detail}")
            if status in {"SKIP", "HARNESS_ERROR"} or (status == "SURVIVED" and not equivalent):
                unexpected.append(mutant_id)

    killed_total = len(mutants) - len(unexpected)
    print(f"\nMutations: {len(mutants)}; unexpectedly survived, skipped or harness errors: {unexpected or 'none'}")
    print(f"Killed or expected survivors: {killed_total}/{len(mutants)}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
