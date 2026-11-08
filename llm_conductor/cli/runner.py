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
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from llm_conductor.cli.exit_status import ExitStatus
from llm_conductor.cli.glossary import GlossarySource
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
    RUN_STAGE_ALL,
    RUN_STAGE_TRANSLATE,
    RUN_STAGE_VERIFY,
    STAGE_FINAL,
    STAGE_REDACTED,
    STAGE_TRANSLATED,
)
from llm_conductor.core.api_client import translate_key, verify_key
from llm_conductor.core.io import safe_read_file, safe_write_file
from llm_conductor.core.locking import FileLock
from llm_conductor.exceptions import (
    ConfigValidationError,
    FileOperationError,
    LockUnavailableError,
    RunInterrupted,
)
from llm_conductor.interfaces import ConfigProtocol, HeartbeatProgressProtocol
from llm_conductor.l10n.processor import process_stage_recursive


@dataclass(frozen=True)
class RunOptions:
    """Pipeline choices of the run command that apply to every file."""

    stage: str = RUN_STAGE_ALL
    strict: bool = False

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

    def __init__(self, config: ConfigProtocol, options: RunOptions, layout: ResultLayout,
                 glossary: Dict[str, str], runtime_configs: Dict[str, dict],
                 heartbeat: Optional[HeartbeatProgressProtocol] = None):
        """
        Args:
            config: Configuration of the run.
            options: Stage and mode selection.
            layout: Where results are written.
            glossary: Glossary {term: translation}.
            runtime_configs: Validated runtime_config per operation type that runs.
            heartbeat: Optional progress ticks for process_stage_recursive.
        """
        self._config = config
        self._options = options
        self._layout = layout
        self._glossary = glossary
        self._runtime_configs = runtime_configs
        self._heartbeat = heartbeat

    def process(self, entry: TargetEntry) -> FileOutcome:
        """
        Process one file; never raises for a problem of this file alone.

        Args:
            entry: The input file.

        Returns:
            The outcome; FAILED and CRASHED are already logged.
        """
        logger = self._config.logger
        try:
            return self._process(entry)
        except (FileOperationError, OSError) as e:
            logger.error(f"{entry.relative_path}: {e}")
            return FileOutcome(entry, FileStatus.FAILED)
        except Exception as e:  # noqa: BLE001 - one broken file must not stop the others
            logger.error(f"{entry.relative_path}: unexpected {type(e).__name__}: {e}", include_traceback=True)
            return FileOutcome(entry, FileStatus.CRASHED)

    def _process(self, entry: TargetEntry) -> FileOutcome:
        """
        The stages of one file.

        Raises:
            FileOperationError: If an input cannot be read or a result cannot be written.
            OSError: If a result directory cannot be created.
        """
        source = self._read_json(entry.source_path, "input data")
        if not validate_input_data(source, entry.source_path, self._config):
            raise FileOperationError(f"invalid input data in {entry.source_path}")

        data = source
        if self._options.translates:
            data = self._run_stage(entry, STAGE_TRANSLATED, data, source, translate_key, OPERATION_TRANSLATION)
            if self._config.circuit_breaker.is_tripped(OPERATION_TRANSLATION):
                return self._abort(entry, data, OPERATION_TRANSLATION)

        if self._options.verifies:
            if not self._options.translates:
                translated_path = self._layout.stage_path(STAGE_TRANSLATED, entry)
                data = self._read_json(translated_path, "translated data")
                self._config.logger.info(f"Loaded translated data from {translated_path}")
            data = self._run_stage(entry, STAGE_REDACTED, data, source, verify_key, OPERATION_VERIFICATION)
            if self._config.circuit_breaker.is_tripped(OPERATION_VERIFICATION):
                return self._abort(entry, data, OPERATION_VERIFICATION)

        final_path = self._write_final(entry, data)
        self._config.logger.info(f"Final output saved to {final_path}")
        return FileOutcome(entry, FileStatus.DONE)

    def _run_stage(self, entry: TargetEntry, stage: str, data: Any, source: Any,
                   process_func: StageFunction, operation_type: str) -> Any:
        """Run one stage over the tree and write its stage result."""
        return process_stage_recursive(
            self._layout.prepare(stage, entry),
            stage,
            data,
            source,  # the English source: glossary terms are matched against it
            process_func,
            operation_type == OPERATION_VERIFICATION,
            self._config,
            self._glossary,
            operation_type,
            runtime_config=dict(self._runtime_configs[operation_type]),
            heartbeat=self._heartbeat,
        )

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

    def __init__(self, config: ConfigProtocol, options: RunOptions, target: RunTarget,
                 layout: ResultLayout, glossary: GlossarySource):
        """
        Args:
            config: Configuration of the run.
            options: Stage and mode selection.
            target: The resolved target.
            layout: Where results are written.
            glossary: The glossary of the run.
        """
        self._config = config
        self._options = options
        self._target = target
        self._layout = layout
        self._glossary = glossary

    def run(self) -> int:
        """
        Run the pipeline over every file of the target.

        Returns:
            The exit code of the run (EXIT_CODE_PRIORITY decides between problems).
        """
        status = ExitStatus()
        self._log_parameters()
        heartbeat = HeartbeatWriter(
            cache_dir=self._config.cache_dir,
            interval=self._config.heartbeat_interval,
            file_ops=self._config.file_ops,
            logger=self._config.logger,
            target=self._target,
            profile_name=self._config.profile_name,
            stages=self._options.operations,
            glossary=self._glossary,
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

    def _log_parameters(self) -> None:
        """Log what the run is about to do."""
        logger = self._config.logger
        kind = f"manifest, {len(self._target.entries)} file(s)" if self._target.is_manifest else "file"
        logger.info(f"  - Target: {self._target.path} ({kind})")
        logger.info(f"  - Stages: {', '.join(self._options.operations)}")
        if self._options.translates:
            logger.info(f"  - Translation mode: {self._options.translation_mode.upper()}")
        logger.info(f"  - Results: {self._layout.root}")
        cache = self._config.cache_dir if self._config.cache_enabled else "disabled"
        logger.info(f"  - Cache: {cache}")
        logger.info(f"  - {self._glossary.describe()}")
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
        heartbeat.start()
        runtime_configs = self._runtime_configs()

        if not self._endpoints_available():
            status.record(EXIT_API_UNAVAILABLE, "an API endpoint is not available")
            return

        # Both caches are loaded even for one stage: a signal saves both.
        self._config.cache_manager.load(OPERATION_TRANSLATION)
        self._config.cache_manager.load(OPERATION_VERIFICATION)

        outcomes = self._process_files(runtime_configs, status, heartbeat)

        for operation_type in self._options.operations:
            self._config.cache_manager.save(operation_type)
        self._config.logger.info("Cache saved.")

        self._report(outcomes, status)

    def _runtime_configs(self) -> Dict[str, dict]:
        """
        Validated runtime configurations of the stages that run.

        Raises:
            ConfigValidationError: If one is invalid.
        """
        configs: Dict[str, dict] = {}
        if self._options.translates:
            translation = self._config.get_translation_config_for_mode(self._options.translation_mode)
            translation['mode'] = self._options.translation_mode
            configs[OPERATION_TRANSLATION] = validate_runtime_config(
                translation, self._config, OPERATION_TRANSLATION)
        if self._options.verifies:
            verification = self._config.get_verification_config()
            verification['mode'] = MODE_VERIFICATION
            configs[OPERATION_VERIFICATION] = validate_runtime_config(
                verification, self._config, OPERATION_VERIFICATION)
        return configs

    def _endpoints_available(self) -> bool:
        """
        Preflight check of the endpoints of the stages that run.

        Skipped when [preflight].check_availability is false. The
        verification endpoint is checked only when it differs from the
        translation one that was already checked.

        Returns:
            False if an endpoint failed the check (the reason is logged).
        """
        config = self._config
        if not config.preflight.check_availability:
            config.logger.info("API availability check disabled ([preflight].check_availability = false)")
            return True

        checked: List[Tuple[str, str]] = []
        for operation_type, api in ((OPERATION_TRANSLATION, config.translation_api),
                                    (OPERATION_VERIFICATION, config.verification_api)):
            if operation_type not in self._options.operations or (api.url, api.model) in checked:
                continue
            config.logger.info(f"Checking {operation_type} API availability...")
            if not check_api_availability(api.url, api.model, api.api_type, config,
                                          api_key=api.api_key, timeout=config.preflight.timeout,
                                          max_tokens_param=api.max_tokens_param):
                config.logger.error(f"{operation_type.capitalize()} API is not available. "
                                    f"Check the profile and the API server.")
                return False
            checked.append((api.url, api.model))
        return True

    def _process_files(self, runtime_configs: Dict[str, dict], status: ExitStatus,
                       heartbeat: HeartbeatWriter) -> List[FileOutcome]:
        """
        Process the files in order; the circuit breaker stops the rest.

        Returns:
            The outcomes of the processed files.
        """
        pipeline = FilePipeline(self._config, self._options, self._layout,
                                self._glossary.terms, runtime_configs,
                                heartbeat=heartbeat)
        entries = self._target.entries
        outcomes: List[FileOutcome] = []

        for index, entry in enumerate(entries, 1):
            if self._target.is_manifest:
                self._config.logger.info("=" * 70)
                self._config.logger.info(f"File {index}/{len(entries)}: {entry.relative_path}")
                self._config.logger.info("=" * 70)

            heartbeat.on_file_start(entry, index)
            outcome = pipeline.process(entry)
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
        if self._glossary.terms:
            config.print_glossary_cache_stats()
        config.print_token_metrics()

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
