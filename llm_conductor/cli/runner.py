# -*- coding: utf-8 -*-
"""
The run command: every file of a target through the selected stages.

FilePipeline processes one file; TargetRunner locks the target and the
cache, checks the endpoints, walks the files - a failed file does not stop
the others - and turns everything that went wrong into one exit code.
"""
import os
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

from llm_conductor.cli.draft_validate import compare_json_structure
from llm_conductor.cli.exit_status import ExitStatus
from llm_conductor.cli.glossary import GlossaryResolver, GlossarySource
from llm_conductor.cli.heartbeat import HeartbeatWriter
from llm_conductor.cli.targets import ResultLayout, RunTarget, TargetEntry
from llm_conductor.cli.validators import (
    check_api_availability,
    validate_input_data,
    validate_runtime_config,
)
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
    HEARTBEAT_STATUS_COMPLETED,
    HEARTBEAT_STATUS_INTERRUPTED,
    MODE_STRICT,
    MODE_STYLE,
    MODE_VERIFICATION,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    RUN_REPORT_FILENAME,
    RUN_STAGE_ALL,
    RUN_STAGE_TRANSLATE,
    RUN_STAGE_VERIFY,
    STAGE_FINAL,
    STAGE_REDACTED,
    STAGE_TRANSLATED,
    USAGE_REPORT_FILENAME,
)
from llm_conductor.core.api_client import translate_key, verify_key
from llm_conductor.core.io import safe_read_file, safe_write_file
from llm_conductor.core.leaf_report import legacy_sidecar_v1_reason
from llm_conductor.core.locking import FileLock
from llm_conductor.exceptions import (
    ConfigValidationError,
    FileOperationError,
    LockUnavailableError,
    RunInterrupted,
)
from llm_conductor.interfaces import (
    APIConfigProtocol,
    ConfigProtocol,
    HeartbeatProgressProtocol,
    RuntimeConfigFactoryProtocol,
)
from llm_conductor.l10n.processor import process_stage_recursive


@dataclass(frozen=True)
class ProfileEndpoints:
    """Translation and verification endpoints for one llm.toml profile."""

    translation: APIConfigProtocol
    verification: APIConfigProtocol


@dataclass(frozen=True)
class RunOptions:
    """Pipeline choices for one file (or the default for a lone JSON target)."""

    stage: str = RUN_STAGE_ALL
    strict: bool = False

    @classmethod
    def for_entry(cls, entry: TargetEntry) -> 'RunOptions':
        """Build options from a resolved ``TargetEntry``."""
        return cls(stage=entry.stage, strict=entry.strict)

    @property
    def translates(self) -> bool:
        """True when the translation stage runs."""
        return self.stage in (RUN_STAGE_ALL, RUN_STAGE_TRANSLATE)

    @property
    def verifies(self) -> bool:
        """True when the verification stage runs."""
        return self.stage in (RUN_STAGE_ALL, RUN_STAGE_VERIFY)

    @property
    def translation_mode(self) -> str:
        """MODE_STRICT or MODE_STYLE."""
        return MODE_STRICT if self.strict else MODE_STYLE

    @property
    def operations(self) -> Tuple[str, ...]:
        """Operation types of the stages that run, in pipeline order."""
        selected = []
        if self.translates:
            selected.append(OPERATION_TRANSLATION)
        if self.verifies:
            selected.append(OPERATION_VERIFICATION)
        return tuple(selected)


class FileStatus(Enum):
    """How the processing of one file ended."""

    DONE = 'done'          # every selected stage ran; the final result is written
    FAILED = 'failed'      # unreadable input, invalid data or an unwritable result
    ABORTED = 'aborted'    # a stage was stopped by the circuit breaker; a partial final result is written
    CRASHED = 'crashed'    # an unexpected internal error


class FileOutcome(NamedTuple):
    """Result of processing one file."""

    entry: TargetEntry
    status: FileStatus
    tripped_operation: Optional[str] = None


StageFunction = Callable[..., Any]


class FilePipeline:
    """
    Runs the selected stages over one file and writes its results.

    Stateless between files: every call of process() works on its own data,
    the shared state (caches, metrics, circuit breaker) lives in the Config.
    """

    def __init__(self, config: ConfigProtocol, layout: ResultLayout,
                 heartbeat: Optional[HeartbeatProgressProtocol] = None):
        """
        Args:
            config: Configuration of the run (logger, caches, resilience).
            layout: Where results are written.
            heartbeat: Optional progress ticks for process_stage_recursive.
        """
        self._config = config
        self._layout = layout
        self._heartbeat = heartbeat

    def process(self, entry: TargetEntry, options: RunOptions,
                glossary: Dict[str, str],
                runtime_configs: Dict[str, dict]) -> FileOutcome:
        """
        Process one file; never raises for a problem of this file alone.

        Args:
            entry: The input file (and optional draft).
            options: Stage and mode selection for this file.
            glossary: Glossary {term: translation}.
            runtime_configs: Validated runtime_config per operation type that runs.

        Returns:
            The outcome; FAILED and CRASHED are already logged.
        """
        logger = self._config.logger
        try:
            return self._process(entry, options, glossary, runtime_configs)
        except (FileOperationError, OSError) as e:
            logger.error(f"{entry.relative_path}: {e}")
            return FileOutcome(entry, FileStatus.FAILED)
        except Exception as e:  # noqa: BLE001 - one broken file must not stop the others
            logger.error(f"{entry.relative_path}: unexpected {type(e).__name__}: {e}", include_traceback=True)
            return FileOutcome(entry, FileStatus.CRASHED)

    def _process(self, entry: TargetEntry, options: RunOptions,
                 glossary: Dict[str, str],
                 runtime_configs: Dict[str, dict]) -> FileOutcome:
        """
        The stages of one file.

        Raises:
            FileOperationError: If an input cannot be read or a result cannot be written.
            OSError: If a result directory cannot be created.
        """
        source = self._read_json(entry.source_path, "input data")
        if not validate_input_data(source, entry.source_path, self._config):
            raise FileOperationError(f"invalid input data in {entry.source_path}")

        self._config.leaf_outcomes.set_active_file(entry.relative_path)

        data = source
        if options.translates:
            data = self._run_stage(
                entry, STAGE_TRANSLATED, data, source, translate_key,
                OPERATION_TRANSLATION, glossary, runtime_configs)
            self._write_leaves_sidecar(entry)
            if self._config.circuit_breaker.is_tripped(OPERATION_TRANSLATION):
                return self._abort(entry, data, OPERATION_TRANSLATION)

        if options.verifies:
            if not options.translates:
                if entry.draft_path is not None:
                    data = self._read_json(entry.draft_path, "draft data")
                    self._config.logger.info(f"Loaded draft data from {entry.draft_path}")
                else:
                    translated_path = self._layout.stage_path(STAGE_TRANSLATED, entry)
                    data = self._read_json(translated_path, "translated data")
                    self._config.logger.info(f"Loaded translated data from {translated_path}")
                    self._reject_stale_translated_structure(entry, source, data)
                    self._load_leaves_sidecar(entry)
            data = self._run_stage(
                entry, STAGE_REDACTED, data, source, verify_key,
                OPERATION_VERIFICATION, glossary, runtime_configs)
            if options.translates or self._config.leaf_outcomes.sidecar_was_merged():
                self._write_leaves_sidecar(entry)
            if self._config.circuit_breaker.is_tripped(OPERATION_VERIFICATION):
                return self._abort(entry, data, OPERATION_VERIFICATION)

        final_path = self._write_final(entry, data)
        self._config.logger.info(f"Final output saved to {final_path}")
        return FileOutcome(entry, FileStatus.DONE)

    def _run_stage(self, entry: TargetEntry, stage: str, data: Any, source: Any,
                   process_func: StageFunction, operation_type: str,
                   glossary: Dict[str, str],
                   runtime_configs: Dict[str, dict]) -> Any:
        """Run one stage over the tree and write its stage result."""
        return process_stage_recursive(
            self._layout.prepare(stage, entry),
            stage,
            data,
            source,  # the English source: glossary terms are matched against it
            process_func,
            operation_type == OPERATION_VERIFICATION,
            self._config,
            glossary,
            operation_type,
            runtime_config=dict(runtime_configs[operation_type]),
            heartbeat=self._heartbeat,
        )

    def _write_leaves_sidecar(self, entry: TargetEntry) -> None:
        """Persist per-leaf stage statuses for this file under result/leaves/."""
        path = self._layout.prepare_leaves(entry)
        payload = self._config.leaf_outcomes.file_sidecar_dict(entry.relative_path)
        if not safe_write_file(
            path, payload, f"leaf status for {entry.relative_path}",
            self._config.file_ops, self._config.logger,
        ):
            self._config.logger.warning(f"Failed to write leaf status sidecar {path}")

    def _reject_stale_translated_structure(
            self, entry: TargetEntry, source: Any, translated: Any,
    ) -> None:
        """
        Fail verify-only when ``translated/`` no longer matches the original shape.

        Text-only leaf changes at the same path stay allowed (``source_stale``
        handles those). Structural drift must not reach the editor or overwrite
        ``final/``.
        """
        issues = compare_json_structure(source, translated)
        if not issues:
            return
        detail = "\n".join(f"  {issue}" for issue in issues)
        raise FileOperationError(
            f"{entry.relative_path}: translated structure does not match original; "
            f"re-run translate before verify\n{detail}"
        )

    def _load_leaves_sidecar(self, entry: TargetEntry) -> None:
        """
        Load translation statuses for verify-only gates (degraded / missing / stale).

        Missing sidecar: verification still runs; log once that skip protection
        is unavailable. Loading a translation record enables
        ``translation_missing`` skips for that file only.
        Schema v1 sidecars are rejected with an instruction to re-run translate.
        """
        path = self._layout.leaves_path(entry)
        if not os.path.isfile(path):
            if self._config.leaf_outcomes.note_sidecar_missing():
                self._config.logger.warning(
                    "Leaf status sidecar missing; verification cannot apply "
                    "translation_degraded / translation_missing gates (status "
                    "is not inferred from string equality)."
                )
            return
        try:
            payload = self._read_json(path, "leaf status sidecar")
        except FileOperationError as exc:
            self._config.logger.warning(f"Failed to load leaf status sidecar {path}: {exc}")
            return
        if not isinstance(payload, dict):
            return
        v1_reason = legacy_sidecar_v1_reason(payload)
        if v1_reason is not None:
            raise FileOperationError(
                f"{entry.relative_path}: leaf status sidecar uses schema v1 "
                f"({v1_reason}); re-run translate to regenerate schema v2 "
                f"sidecars before verify"
            )
        self._config.leaf_outcomes.merge_sidecar(entry.relative_path, payload)

    def _abort(self, entry: TargetEntry, data: Any, operation_type: str) -> FileOutcome:
        """Write the partial final result of a stage stopped by the circuit breaker."""
        self._config.logger.error(
            f"{operation_type} stage aborted by the circuit breaker "
            f"([resilience].max_consecutive_failures); writing partial results."
        )
        final_path = self._write_final(entry, data)
        self._config.logger.info(f"Final output (partial) saved to {final_path}")
        return FileOutcome(entry, FileStatus.ABORTED, operation_type)

    def _write_final(self, entry: TargetEntry, data: Any) -> str:
        """
        Write the final result of a file.

        Returns:
            Its path.

        Raises:
            FileOperationError: If it cannot be written.
        """
        path = self._layout.prepare(STAGE_FINAL, entry)
        if not safe_write_file(path, data, "final output", self._config.file_ops, self._config.logger):
            raise FileOperationError(f"failed to write {path}")
        return path

    def _read_json(self, path: str, description: str) -> Any:
        """
        Read a JSON input.

        Raises:
            FileOperationError: If it is missing, unreadable or not valid JSON.
        """
        return safe_read_file(path, description, self._config.file_ops, self._config.logger,
                              critical=True, as_text=False)


class TargetRunner:
    """
    Executes the run command for a resolved target.

    Holds exclusive OS-level locks on the target file, on the cache
    directory (when used) and on the shared result root (when results go
    under result/<target name>/) for the whole run, so a second run of the
    same target, cache or result root exits with EXIT_TARGET_LOCKED instead
    of interleaving writes. The shared result root is also claimed by an
    owner marker so a different same-named target cannot reuse it.

    Writes ``{cache_dir}/progress_state.json`` for the whole run (even when
    the disk cache is off). On ``RunInterrupted`` the heartbeat is finalized
    with status ``interrupted`` / exit 130, then the exception is re-raised
    so the entry point can exit; locks unwind on the way out.
    """

    def __init__(self, config: ConfigProtocol, target: RunTarget,
                 layout: ResultLayout,
                 factories: Mapping[str, RuntimeConfigFactoryProtocol],
                 endpoints: Mapping[str, ProfileEndpoints],
                 glossaries: Dict[Optional[str], GlossarySource],
                 default_options: Optional[RunOptions] = None):
        """
        Args:
            config: Bootstrap configuration (logger, caches, resilience).
            target: The resolved target.
            layout: Where results are written.
            factories: Runtime config factories keyed by absolute llm.toml path.
            endpoints: API endpoints keyed by the same llm.toml paths.
            glossaries: Glossary sources keyed by absolute path (``None`` → empty).
            default_options: Fallback stage/strict when an entry has no overrides
                (lone JSON path); unused when every entry carries its own settings.
        """
        self._config = config
        self._target = target
        self._layout = layout
        self._factories = factories
        self._endpoints = endpoints
        self._glossaries = glossaries
        self._default_options = default_options or RunOptions()

    def run(self) -> int:
        """
        Run the pipeline over every file of the target.

        Returns:
            The exit code of the run (EXIT_CODE_PRIORITY decides between problems).
        """
        status = ExitStatus()
        self._log_parameters()
        union_ops = self._union_operations()
        # Heartbeat profile/glossary: bootstrap profile and first non-empty glossary.
        first_glossary = next(
            (g for g in self._glossaries.values() if g.terms),
            GlossarySource(),
        )
        heartbeat = HeartbeatWriter(
            cache_dir=self._config.cache_dir,
            interval=self._config.heartbeat_interval,
            file_ops=self._config.file_ops,
            logger=self._config.logger,
            target=self._target,
            profile_name=self._config.profile_name,
            stages=union_ops,
            glossary=first_glossary,
        )
        try:
            with ExitStack() as locks:
                self._acquire_locks(locks)
                self._execute(status, heartbeat)
        except RunInterrupted:
            heartbeat.set_problems(status.records)
            heartbeat.finalize(HEARTBEAT_STATUS_INTERRUPTED, EXIT_INTERRUPTED)
            raise
        except LockUnavailableError as e:
            self._config.logger.error(f"{e}; exiting - a later run will pick the work up.")
            status.record(EXIT_TARGET_LOCKED, str(e))
        except ConfigValidationError as e:
            self._config.logger.error(f"Runtime configuration is invalid: {e}")
            status.record(EXIT_CONFIG_ERROR, str(e))

        code = status.code
        if heartbeat.started and not heartbeat.finalized:
            heartbeat.set_problems(status.records)
            heartbeat.finalize(HEARTBEAT_STATUS_COMPLETED, code)
        log = self._config.logger.info if code == EXIT_SUCCESS else self._config.logger.error
        log(f"Run finished with exit code {code}")
        return code

    def _options_for(self, entry: TargetEntry) -> RunOptions:
        """Effective stage/strict for one entry."""
        if self._target.is_manifest:
            return RunOptions.for_entry(entry)
        return self._default_options

    def _profile_path_for(self, entry: TargetEntry) -> str:
        """Absolute llm.toml path for one entry."""
        if entry.llm_config_path is not None:
            return entry.llm_config_path
        # Bootstrap profile: the only key when a lone JSON uses CLI --llm-config.
        return next(iter(self._factories))

    def _glossary_for(self, entry: TargetEntry) -> GlossarySource:
        """Glossary source for one entry."""
        return self._glossaries.get(entry.glossary_path, GlossarySource())

    def _union_operations(self) -> Tuple[str, ...]:
        """Union of operation types across all entries, in pipeline order."""
        need_translate = False
        need_verify = False
        for entry in self._target.entries:
            options = self._options_for(entry)
            need_translate = need_translate or options.translates
            need_verify = need_verify or options.verifies
        selected = []
        if need_translate:
            selected.append(OPERATION_TRANSLATION)
        if need_verify:
            selected.append(OPERATION_VERIFICATION)
        return tuple(selected)

    def _log_parameters(self) -> None:
        """Log what the run is about to do."""
        logger = self._config.logger
        kind = f"manifest, {len(self._target.entries)} file(s)" if self._target.is_manifest else "file"
        logger.info(f"  - Target: {self._target.path} ({kind})")
        if self._target.is_manifest:
            logger.info("  - Stages: per-target (see each file)")
        else:
            options = self._options_for(self._target.entries[0]) if self._target.entries else self._default_options
            logger.info(f"  - Stages: {', '.join(options.operations)}")
            if options.translates:
                logger.info(f"  - Translation mode: {options.translation_mode.upper()}")
        logger.info(f"  - Results: {self._layout.root}")
        cache = self._config.cache_dir if self._config.cache_enabled else "disabled"
        logger.info(f"  - Cache: {cache}")
        if self._target.is_manifest:
            logger.info("  - Glossary: per-target")
        else:
            entry = self._target.entries[0] if self._target.entries else None
            glossary = self._glossary_for(entry) if entry else GlossarySource()
            logger.info(f"  - {glossary.describe()}")
        logger.info(f"  - Log level: {self._config.log_level}")

    def _acquire_locks(self, locks: ExitStack) -> None:
        """
        Lock the target, then the cache directory (when the disk cache is
        used), then the shared result root (when results are under
        result/<target name>/). The result-root owner marker is claimed
        under that lock so two same-named targets cannot race.

        Raises:
            LockUnavailableError: If a lock is held by another run.
            TargetError: If the shared result root belongs to another target.
        """
        locks.enter_context(FileLock(self._target.path))
        if self._config.cache_enabled:
            locks.enter_context(FileLock(os.path.join(self._config.cache_dir, CACHE_LOCK_FILENAME), create=True))
        if self._layout.shared_by_name:
            locks.enter_context(FileLock(
                os.path.join(self._layout.root, CACHE_LOCK_FILENAME), create=True))
            self._layout.claim_owner(
                self._target.path, self._config.file_ops, self._config.logger)

    def _execute(self, status: ExitStatus, heartbeat: HeartbeatWriter) -> None:
        """
        Everything that happens under the locks.

        Raises:
            ConfigValidationError: If a runtime configuration is invalid.
        """
        # Original/draft JSON preflight runs in cli.main._execute_run before
        # locks (and before --dry-run) via validate_target_inputs.
        heartbeat.start()

        if not self._endpoints_available():
            status.record(EXIT_API_UNAVAILABLE, "an API endpoint is not available")
            return

        # Both caches are loaded even for one stage: a signal saves both.
        self._config.cache_manager.load(OPERATION_TRANSLATION)
        self._config.cache_manager.load(OPERATION_VERIFICATION)

        outcomes = self._process_files(status, heartbeat)

        for operation_type in self._union_operations():
            self._config.cache_manager.save(operation_type)
        self._config.logger.info("Cache saved.")

        self._report(outcomes, status)

    def _runtime_configs_for(self, entry: TargetEntry, options: RunOptions) -> Dict[str, dict]:
        """
        Validated runtime configurations of the stages that run for one entry.

        Raises:
            ConfigValidationError: If one is invalid.
        """
        factory = self._factories[self._profile_path_for(entry)]
        configs: Dict[str, dict] = {}
        if options.translates:
            translation = factory.get_translation_config_for_mode(options.translation_mode)
            translation['mode'] = options.translation_mode
            configs[OPERATION_TRANSLATION] = validate_runtime_config(
                translation, self._config, OPERATION_TRANSLATION)
        if options.verifies:
            verification = factory.get_verification_config()
            verification['mode'] = MODE_VERIFICATION
            configs[OPERATION_VERIFICATION] = validate_runtime_config(
                verification, self._config, OPERATION_VERIFICATION)
        return configs

    def _endpoints_available(self) -> bool:
        """
        Preflight check of every endpoint required by any entry.

        Skipped when [preflight].check_availability is false.

        Returns:
            False if an endpoint failed the check (the reason is logged).
        """
        config = self._config
        if not config.preflight.check_availability:
            config.logger.info("API availability check disabled ([preflight].check_availability = false)")
            return True

        checked: List[Tuple[str, str]] = []
        for entry in self._target.entries:
            options = self._options_for(entry)
            ends = self._endpoints[self._profile_path_for(entry)]
            pairs = []
            if options.translates:
                pairs.append((OPERATION_TRANSLATION, ends.translation))
            if options.verifies:
                pairs.append((OPERATION_VERIFICATION, ends.verification))
            for operation_type, api in pairs:
                if (api.url, api.model) in checked:
                    continue
                config.logger.info(f"Checking {operation_type} API availability "
                                   f"({os.path.basename(self._profile_path_for(entry))})...")
                if not check_api_availability(api.url, api.model, api.api_type, config,
                                              api_key=api.api_key, timeout=config.preflight.timeout,
                                              max_tokens_param=api.max_tokens_param):
                    config.logger.error(f"{operation_type.capitalize()} API is not available. "
                                        f"Check the profile and the API server.")
                    return False
                checked.append((api.url, api.model))
        return True

    def _process_files(self, status: ExitStatus,
                       heartbeat: HeartbeatWriter) -> List[FileOutcome]:
        """
        Process the files in order; the circuit breaker stops the rest.

        Returns:
            The outcomes of the processed files.
        """
        pipeline = FilePipeline(self._config, self._layout, heartbeat=heartbeat)
        entries = self._target.entries
        outcomes: List[FileOutcome] = []

        for index, entry in enumerate(entries, 1):
            options = self._options_for(entry)
            glossary = self._glossary_for(entry)
            runtime_configs = self._runtime_configs_for(entry, options)

            if self._target.is_manifest:
                self._config.logger.info("=" * 70)
                self._config.logger.info(f"File {index}/{len(entries)}: {entry.relative_path}")
                self._config.logger.info(
                    f"  stage={options.stage} strict={options.strict} "
                    f"llm_config={entry.llm_config_path or '(bootstrap)'} "
                    f"glossary={glossary.path or '(none)'}"
                    + (f" draft={entry.draft_path}" if entry.draft_path else "")
                )
                self._config.logger.info("=" * 70)

            heartbeat.on_file_start(entry, index)
            outcome = pipeline.process(entry, options, glossary.terms, runtime_configs)
            outcomes.append(outcome)
            heartbeat.on_file_done(entry.relative_path, outcome.status.value)

            if outcome.status is FileStatus.CRASHED:
                status.record(EXIT_INTERNAL_ERROR, f"internal error while processing {entry.relative_path}")
            if outcome.status is FileStatus.ABORTED:
                status.record(EXIT_CONSECUTIVE_FAILURES,
                              f"{outcome.tripped_operation} stopped by the circuit breaker")
                skipped = len(entries) - index
                if skipped:
                    self._config.logger.error(f"Circuit breaker: {skipped} remaining file(s) not processed.")
                break

        return outcomes

    def _report(self, outcomes: List[FileOutcome], status: ExitStatus) -> None:
        """Print the run metrics and record failure shares above [resilience].max_failure_ratio."""
        config = self._config
        config.print_api_metrics()
        if any(g.terms for g in self._glossaries.values()):
            config.print_glossary_cache_stats()
        config.print_token_metrics()
        config.print_leaf_outcome_banner()
        self._write_run_artifacts()

        threshold = config.resilience.max_failure_ratio
        for operation_type, ratio in config.failure_ratio_violations():
            message = (f"{operation_type}: failure ratio {ratio:.1%} exceeds the "
                       f"[resilience].max_failure_ratio threshold of {threshold:.1%}")
            config.logger.error(message)
            status.record(EXIT_FAILURE_RATIO_EXCEEDED, message)
            self._warn_residual_cache_miss_ratio(operation_type, ratio)

        failed = [o.entry.relative_path for o in outcomes if o.status is FileStatus.FAILED]
        if failed:
            ratio = len(failed) / len(outcomes)
            config.logger.error(f"{len(failed)} of {len(outcomes)} file(s) failed: {', '.join(failed)}")
            if ratio > threshold:
                message = (f"file failure ratio {ratio:.1%} exceeds the "
                           f"[resilience].max_failure_ratio threshold of {threshold:.1%}")
                config.logger.error(message)
                status.record(EXIT_FAILURE_RATIO_EXCEEDED, message)

        if config.resilience.fail_on_any_error and config.leaf_outcomes.has_untreated_error():
            message = (
                "one or more leaves ended degraded "
                "([resilience].fail_on_any_error is enabled)"
            )
            config.logger.error(message)
            status.record(EXIT_FAILURE_RATIO_EXCEEDED, message)

    def _write_run_artifacts(self) -> None:
        """Write run_report.json and usage.json under the result root."""
        root = self._layout.root
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as exc:
            self._config.logger.warning(f"Cannot create result root for reports: {exc}")
            return

        report_path = os.path.join(root, RUN_REPORT_FILENAME)
        usage_path = os.path.join(root, USAGE_REPORT_FILENAME)
        file_ops = self._config.file_ops
        logger = self._config.logger

        if safe_write_file(
            report_path, self._config.leaf_outcomes.build_run_report(),
            "run report", file_ops, logger,
        ):
            logger.info(f"Run report written to {report_path}")
        else:
            logger.warning(f"Failed to write run report {report_path}")

        if safe_write_file(
            usage_path, self._config.usage_ledger.build_export(),
            "usage report", file_ops, logger,
        ):
            logger.info(f"Usage report written to {usage_path}")
        else:
            logger.warning(f"Failed to write usage report {usage_path}")

    def _warn_residual_cache_miss_ratio(self, operation_type: str, ratio: float) -> None:
        """
        Explain exit 5 when this run only re-queried cache misses.

        Rejected replies are never cached, so a second run with a warm cache
        may call the model only for those residual leaves. Failure ratio is
        then measured against that smaller request set — a prior successful
        run can still exit 5 if those leaves fail again. Cache hits > 0 is
        the signal that this stage was a partial rerun, not a cold start.
        """
        cache_stats = self._config.cache_manager.get_stats(operation_type)
        if cache_stats.get('hits', 0) <= 0:
            return

        total = (self._config.metrics.get_snapshot()
                 .get(operation_type, {})
                 .get('total_requests', 0))
        self._config.logger.warning(
            f"{operation_type}: {total} model request(s) were cache misses "
            f"(new values or prior validation rejects - rejects are never cached). "
            f"Failure ratio {ratio:.1%} is measured against this run's requests only; "
            f"a prior successful run can still exit with code {EXIT_FAILURE_RATIO_EXCEEDED} "
            f"on a cache rerun if those residual leaves fail again. "
            f"Fix the prompt/model, or remeasure with --no-cache."
        )


def load_glossaries_for_target(
        target: RunTarget,
        file_ops,
        logger,
        *,
        cli_glossary: Optional[str] = None,
) -> Dict[Optional[str], GlossarySource]:
    """
    Load every glossary referenced by the target (plus CLI glossary for JSON).

    Args:
        target: Resolved run target.
        file_ops: File operations.
        logger: Logger.
        cli_glossary: ``--glossary`` for a lone JSON target.

    Returns:
        Map of absolute glossary path → source; ``None`` key is the empty glossary.
    """
    resolver = GlossaryResolver(file_ops, logger)
    found: Dict[Optional[str], GlossarySource] = {None: GlossarySource()}

    if not target.is_manifest:
        source = resolver.resolve(cli_glossary, target)
        found[source.path] = source
        # Lone JSON entries leave glossary_path as None; map None to the
        # resolved CLI / beside-target glossary (or empty).
        found[None] = source
        return found

    for entry in target.entries:
        path = entry.glossary_path
        if path in found:
            continue
        if path is None:
            continue
        found[path] = resolver.load_path(path)
    return found
