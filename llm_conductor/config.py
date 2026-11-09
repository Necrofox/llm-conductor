# -*- coding: utf-8 -*-
"""
Configuration layer and composition root.

Config is the only class that knows the concrete components; everything
else depends on the protocols of llm_conductor/interfaces.py. The data classes are
frozen: a loaded configuration never changes during a run, so any number of
threads may read it without locking.
"""
import logging
import logging.handlers
import os
import sys
import time
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

from llm_conductor.constants import (
    DEBUG_LOG_LEVELS,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_EXTENDED_DIAGNOSTIC,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_RETENTION_DAYS,
    DEFAULT_MAX_TOKENS_PARAM,
    LOG_FILE_PREFIX,
    LOG_TIMESTAMP_FORMAT,
    MAX_OPENAI_STOP_SEQUENCES,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    MODE_STRICT,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    TRACE_LOG_FILE_PREFIX,
    TRACE_LOG_LEVEL,
    TRACE_LOG_LEVEL_NAME,
    VALID_ALLOWED_SCRIPTS,
    VALID_API_TYPES,
    VALID_LATIN_POLICIES,
    VALID_LOG_LEVELS,
    VALID_MAX_TOKENS_PARAMS,
)
from llm_conductor.core.cache import CacheManager, GlossaryPatternCache
from llm_conductor.core.circuit_breaker import ConsecutiveFailureTracker
from llm_conductor.core.io import FileOperations, safe_read_file
from llm_conductor.core.leaf_report import LeafOutcomeRecorder
from llm_conductor.core.metrics import MetricsCollector
from llm_conductor.core.usage import UsageLedger
from llm_conductor.exceptions import ConfigValidationError
from llm_conductor.interfaces import (
    CacheManagerProtocol,
    CircuitBreakerProtocol,
    FileOperationsProtocol,
    GlossaryPatternCacheProtocol,
    LeafOutcomeRecorderProtocol,
    Logger,
    MetricsProtocol,
    UsageLedgerProtocol,
)
from llm_conductor.schema import CONDUCTOR_SCHEMA, LLM_SCHEMA

SectionValues = Mapping[str, Any]

logging.addLevelName(TRACE_LOG_LEVEL, TRACE_LOG_LEVEL_NAME)


class _ErrorTrackingHandler(logging.Handler):
    """
    Flags whether an ERROR-or-above record was emitted on the logger it is attached to.

    Attached at ERROR level, so emit() is only called for records that
    qualify - it never has to re-check the level itself. Used by
    ConfigLogger.close_log_files() to tell a clean run from a problem one.
    """

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.had_errors = False

    def emit(self, record: logging.LogRecord) -> None:
        self.had_errors = True


class ConfigLogger(Logger):
    """Logger implementation on top of the standard logging.Logger."""

    def __init__(self, logger: logging.Logger, is_debug_mode: bool = False,
                 error_tracker: Optional[_ErrorTrackingHandler] = None,
                 log_file_paths: Optional[List[str]] = None):
        self._logger = logger
        self.is_debug_mode = is_debug_mode
        self._error_tracker = error_tracker
        self._log_file_paths = log_file_paths or []

    def trace(self, message: str):
        self._logger.log(TRACE_LOG_LEVEL, message)

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str, include_traceback: bool = False):
        if include_traceback:
            self._logger.exception(message)
        else:
            self._logger.error(message)

    def setLevel(self, level: int):
        """
        Change the level of the underlying logger and keep is_debug_mode in sync.

        Args:
            level: A logging level number.
        """
        self._logger.setLevel(level)

        if level == logging.DEBUG:
            self.is_debug_mode = True
        elif level > logging.DEBUG:
            self.is_debug_mode = False

    @property
    def had_errors(self) -> bool:
        """True if an ERROR-or-above record was logged since create()."""
        return self._error_tracker.had_errors if self._error_tracker else False

    def close_log_files(self, always_keep: bool = False, header: Optional[str] = None) -> None:
        """
        Detach this run's handlers; keep, condense or delete its log files.

        A "clean" run (no ERROR-or-above record, always_keep False) has
        nothing worth keeping in full, but is not necessarily left with
        nothing: header, when given, replaces the removed detail with a
        condensed run summary instead of deleting the file outright (Config
        supplies it only for [logging].level VERBOSE). A problem run's files
        are always kept as written, header or not. Safe to call on a logger
        created with log_to_file=False: there is then nothing to touch.

        Args:
            always_keep: Keep the files exactly as written even on a clean
                run (the caller knows something else went wrong, e.g. a
                non-zero exit code with no matching log message, or wants
                every byte kept regardless - DEBUG level, extended_diagnostic).
            header: Replacement content for a clean run's file(s) instead of
                deleting them; ignored (nothing is touched) when the run is
                kept in full.
        """
        keep_as_written = always_keep or self.had_errors

        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

        if keep_as_written:
            return

        for path in self._log_file_paths:
            try:
                if header is None:
                    os.remove(path)
                else:
                    # Direct write, not safe_write_file: this is the log
                    # file's own post-mortem handling right after its
                    # handler was detached above, not application data -
                    # the same exception the os.remove() branch above
                    # already relies on for the same file.
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(header)
            except OSError:
                pass

    @staticmethod
    def build_log_filename(log_name: Optional[str], started: datetime,
                            prefix: str = LOG_FILE_PREFIX) -> str:
        """
        File name of a run's log: <prefix>[_<log_name>]_<start time>.log.

        Args:
            log_name: Name of the run target, or None for a run without one.
            started: Start time of the run.
            prefix: LOG_FILE_PREFIX for the main log, TRACE_LOG_FILE_PREFIX
                for the extended-diagnostic one.

        Returns:
            The bare file name.
        """
        parts = [prefix]
        if log_name:
            parts.append(log_name)
        parts.append(started.strftime(LOG_TIMESTAMP_FORMAT))
        return "_".join(parts) + ".log"

    @staticmethod
    def create(log_level: str = DEFAULT_LOG_LEVEL, log_to_file: bool = True,
               log_dir: str = DEFAULT_LOG_DIR, log_name: Optional[str] = None,
               retention_days: int = DEFAULT_LOG_RETENTION_DAYS,
               extended_diagnostic: bool = DEFAULT_LOG_EXTENDED_DIAGNOSTIC) -> 'ConfigLogger':
        """
        Configure the process-wide 'conductor' logger and wrap it.

        Previous handlers of that logger are dropped, so calling this again
        replaces the output channels instead of duplicating them. Log files
        rotate daily (TimedRotatingFileHandler) and keep retention_days of
        history; extended_diagnostic adds a second file capturing TRACE
        records (full prompts, cache-key hashes) that never reach the
        console or the main file.

        Args:
            log_level: Level name from [logging].level.
            log_to_file: Also write a DEBUG-level log file.
            log_dir: Directory of the log file(s); created if missing.
            log_name: Name of the run target, made part of the log file name.
            retention_days: Rotated log files kept, [logging].retention_days.
            extended_diagnostic: Also write the TRACE-level file, [logging].extended_diagnostic.

        Returns:
            The wrapping ConfigLogger.
        """
        logger = logging.getLogger('conductor')
        console_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(TRACE_LOG_LEVEL if extended_diagnostic else console_level)
        logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file_paths: List[str] = []

        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            started = datetime.now().astimezone()  # local wall clock for the log file name

            log_filename = os.path.join(log_dir, ConfigLogger.build_log_filename(log_name, started))
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_filename, when='midnight', backupCount=retention_days, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            log_file_paths.append(log_filename)

            if extended_diagnostic:
                trace_filename = os.path.join(
                    log_dir, ConfigLogger.build_log_filename(log_name, started, prefix=TRACE_LOG_FILE_PREFIX))
                trace_handler = logging.handlers.TimedRotatingFileHandler(
                    trace_filename, when='midnight', backupCount=retention_days, encoding='utf-8')
                trace_handler.setLevel(TRACE_LOG_LEVEL)
                trace_handler.setFormatter(formatter)
                logger.addHandler(trace_handler)
                log_file_paths.append(trace_filename)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        error_tracker = _ErrorTrackingHandler()
        logger.addHandler(error_tracker)

        return ConfigLogger(logger, log_level.upper() in DEBUG_LOG_LEVELS,
                             error_tracker=error_tracker, log_file_paths=log_file_paths)


@dataclass(frozen=True)
class APIConfig:
    """One API endpoint: transport, model, system prompt and key."""

    url: str
    api_type: str
    timeout: int
    max_retries: int
    retry_backoff_factor: float
    model: str
    system_prompt: str = ""
    api_key: Optional[str] = None
    key_env: str = ""
    keep_alive: str = ""
    max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM

    @classmethod
    def from_section(cls, api: SectionValues, system_prompt: str,
                     environ: Mapping[str, str]) -> 'APIConfig':
        """
        Build an endpoint from a resolved [<stage>.api] table.

        The key comes from the environment variable named by key_env when it
        is set and non-empty, otherwise from the literal key.

        Args:
            api: Resolved values of the [<stage>.api] table.
            system_prompt: System prompt of the stage.
            environ: Environment to look key_env up in.

        Returns:
            The endpoint configuration (not yet validated).
        """
        key_env = api["key_env"]
        env_key = environ.get(key_env, "") if key_env else ""
        api_key = env_key if env_key.strip() else api["key"]
        return cls(
            url=api["url"],
            api_type=api["type"].lower(),
            timeout=api["timeout"],
            max_retries=api["max_retries"],
            retry_backoff_factor=api["retry_backoff_factor"],
            model=api["model"],
            system_prompt=system_prompt,
            api_key=api_key or None,
            key_env=key_env,
            keep_alive=api["keep_alive"],
            max_tokens_param=api["max_tokens_param"],
        )

    def get_headers(self) -> Dict[str, str]:
        """
        Build the HTTP headers of a request to this endpoint.

        Returns:
            Content type plus a Bearer token for 'openai' endpoints with a key.
        """
        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key and self.api_key.strip() and self.api_type == 'openai':
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"

        return headers

    def validate(self) -> None:
        """
        Check the endpoint invariants.

        Raises:
            ConfigValidationError: On an empty URL or model, an unknown
                api_type, a non-positive timeout, a missing key for 'openai',
                or an unrecognized max_tokens_param.
        """
        if not self.url:
            raise ConfigValidationError("API URL cannot be empty")

        if self.api_type not in VALID_API_TYPES:
            raise ConfigValidationError(
                f"Invalid api_type: '{self.api_type}'. "
                f"Must be one of {VALID_API_TYPES}"
            )

        if self.timeout <= 0:
            raise ConfigValidationError(
                f"timeout must be positive, got {self.timeout}"
            )

        if not self.model:
            raise ConfigValidationError("model cannot be empty")

        if self.api_type == 'openai' and not self.api_key:
            source = (f"environment variable '{self.key_env}' is unset or empty and key is empty"
                      if self.key_env else "set key_env or key")
            raise ConfigValidationError(
                f"api_key is required for api_type='openai' "
                f"(OpenAI, DeepSeek, OpenRouter): {source}"
            )

        if self.max_tokens_param not in VALID_MAX_TOKENS_PARAMS:
            raise ConfigValidationError(
                f"max_tokens_param must be one of {VALID_MAX_TOKENS_PARAMS}, "
                f"got '{self.max_tokens_param}'"
            )


@dataclass(frozen=True)
class TranslationConfig:
    """Prompt and sampling parameters of one mode (strict, style or verification)."""

    prompt: str
    temp: float
    num_predict: int
    num_ctx: int
    top_p: float
    top_k: int
    repeat_penalty: float
    frequency_penalty: float
    presence_penalty: float
    stop: Tuple[str, ...] = ()

    @classmethod
    def from_sections(cls, prompt: str, mode: SectionValues, shared: SectionValues) -> 'TranslationConfig':
        """
        Build a mode from its own sampling table and the table shared between modes.

        Args:
            prompt: User prompt of the mode.
            mode: Resolved per-mode sampling (temperature, top_p, top_k,
                repeat_penalty, num_predict, num_ctx).
            shared: Resolved shared sampling (frequency_penalty,
                presence_penalty, stop); may be the same table as mode.

        Returns:
            The mode configuration (not yet validated).
        """
        return cls(
            prompt=prompt,
            temp=mode["temperature"],
            num_predict=mode["num_predict"],
            num_ctx=mode["num_ctx"],
            top_p=mode["top_p"],
            top_k=mode["top_k"],
            repeat_penalty=mode["repeat_penalty"],
            frequency_penalty=shared["frequency_penalty"],
            presence_penalty=shared["presence_penalty"],
            stop=shared["stop"],
        )

    def validate(self) -> None:
        """
        Check the sampling invariants.

        Raises:
            ConfigValidationError: On temperature or top_p out of range, or a
                non-positive num_predict / num_ctx.
        """
        if not MIN_TEMPERATURE <= self.temp <= MAX_TEMPERATURE:
            raise ConfigValidationError(
                f"temp must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {self.temp}"
            )

        if not MIN_TOP_P <= self.top_p <= MAX_TOP_P:
            raise ConfigValidationError(
                f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}, got {self.top_p}"
            )

        if self.num_predict <= 0:
            raise ConfigValidationError(
                f"num_predict must be positive, got {self.num_predict}"
            )

        if self.num_ctx <= 0:
            raise ConfigValidationError(
                f"num_ctx must be positive, got {self.num_ctx}"
            )


@dataclass(frozen=True)
class PreflightConfig:
    """Endpoint check performed before the run ([preflight] of llm.toml)."""

    check_availability: bool
    timeout: int

    def validate(self) -> None:
        """
        Raises:
            ConfigValidationError: On a non-positive timeout.
        """
        if self.timeout <= 0:
            raise ConfigValidationError(f"preflight timeout must be positive, got {self.timeout}")


@dataclass(frozen=True)
class LimitsConfig:
    """Safety limits ([limits] of conductor.toml)."""

    max_recursion_depth: int
    large_file_threshold_mb: int
    file_operation_timeout: int
    max_error_display: int
    dedup_log_interval: int

    def validate(self) -> None:
        """
        Raises:
            ConfigValidationError: If any limit is not positive.
        """
        for name in ("max_recursion_depth", "large_file_threshold_mb", "file_operation_timeout",
                     "max_error_display", "dedup_log_interval"):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigValidationError(f"limits.{name} must be positive, got {value}")


@dataclass(frozen=True)
class HeartbeatConfig:
    """Progress-state rewrite cadence ([heartbeat] of conductor.toml)."""

    interval: int

    def validate(self) -> None:
        """
        Raises:
            ConfigValidationError: If interval is negative.
        """
        if self.interval < 0:
            raise ConfigValidationError(
                f"heartbeat.interval must be non-negative, got {self.interval}"
            )


@dataclass(frozen=True)
class ResilienceConfig:
    """Consecutive-failure circuit breaker settings ([resilience] of conductor.toml)."""

    max_consecutive_failures: int
    max_failure_ratio: float
    fail_on_any_error: bool = False

    def validate(self) -> None:
        """
        Raises:
            ConfigValidationError: If max_consecutive_failures is negative,
                or max_failure_ratio is outside [0.0, 1.0].
        """
        if self.max_consecutive_failures < 0:
            raise ConfigValidationError(
                f"max_consecutive_failures must be non-negative, got {self.max_consecutive_failures}"
            )
        if not 0.0 <= self.max_failure_ratio <= 1.0:
            raise ConfigValidationError(
                f"max_failure_ratio must be between 0.0 and 1.0, got {self.max_failure_ratio}"
            )


@dataclass(frozen=True)
class ValidationConfig:
    """
    Model-reply sanity check settings.

    Thresholds, latin_text_policy, and save_rejected_replies come from
    [validation] of conductor.toml; allowed_scripts comes from [validation]
    of llm.toml (travels with the language pair in the profile).
    """

    enabled: bool
    max_length_ratio: float
    max_length_slack: int
    latin_text_policy: str
    save_rejected_replies: bool = False
    allowed_scripts: Tuple[str, ...] = ()

    def validate(self) -> None:
        """
        Raises:
            ConfigValidationError: If max_length_ratio or max_length_slack
                is negative, latin_text_policy is not a recognized value, or
                allowed_scripts contains an unknown writing-system name.
        """
        if self.max_length_ratio < 0:
            raise ConfigValidationError(
                f"max_length_ratio must be non-negative, got {self.max_length_ratio}"
            )
        if self.max_length_slack < 0:
            raise ConfigValidationError(
                f"max_length_slack must be non-negative, got {self.max_length_slack}"
            )
        if self.latin_text_policy not in VALID_LATIN_POLICIES:
            raise ConfigValidationError(
                f"latin_text_policy must be one of {VALID_LATIN_POLICIES}, got '{self.latin_text_policy}'"
            )
        unknown = [name for name in self.allowed_scripts if name not in VALID_ALLOWED_SCRIPTS]
        if unknown:
            raise ConfigValidationError(
                f"allowed_scripts contains unknown name(s) {unknown!r}; "
                f"known: {VALID_ALLOWED_SCRIPTS}"
            )


@dataclass(frozen=True)
class ConfigData:
    """
    Complete, validated configuration of a run.

    Holds data and invariants only; files are read by ConfigFileLoader and
    turned into objects by from_toml().
    """

    translation_api: APIConfig
    verification_api: APIConfig
    translation_strict: TranslationConfig
    translation_style: TranslationConfig
    verification: TranslationConfig
    preflight: PreflightConfig
    limits: LimitsConfig
    heartbeat: HeartbeatConfig
    resilience: ResilienceConfig
    validation: ValidationConfig
    profile_name: str
    profile_description: str
    log_level: str
    log_dir: str
    log_retention_days: int
    log_extended_diagnostic: bool
    cache_enabled: bool
    cache_dir: str
    cache_max_size: int
    cache_expiry_days: int
    cache_auto_save_interval: int
    glossary_patterns_cache_max_size: int
    value_deduplication_enabled: bool

    @property
    def is_debug_mode(self) -> bool:
        """True when the log level enables debug diagnostics."""
        return self.log_level in DEBUG_LOG_LEVELS

    @classmethod
    def from_toml(
        cls,
        llm_doc: Mapping[str, Any],
        conductor_doc: Mapping[str, Any],
        base_dir: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        llm_source: Optional[str] = None,
        conductor_source: Optional[str] = None,
    ) -> 'ConfigData':
        """
        Build a validated configuration from the two parsed TOML documents.

        Absent keys take their schema defaults; unknown keys and wrongly
        typed values are errors.

        Args:
            llm_doc: Parsed llm.toml.
            conductor_doc: Parsed conductor.toml.
            base_dir: Directory that relative [logging].dir and [cache].dir
                are resolved from (the directory of conductor.toml); None
                leaves them relative to the working directory.
            environ: Environment for key_env lookups; os.environ by default.
            llm_source: File name of llm_doc for error messages.
            conductor_source: File name of conductor_doc for error messages.

        Returns:
            The validated configuration.

        Raises:
            ConfigValidationError: On a schema violation or a broken invariant.
        """
        llm = LLM_SCHEMA.resolve(llm_doc, llm_source)
        conductor = CONDUCTOR_SCHEMA.resolve(conductor_doc, conductor_source)
        env = os.environ if environ is None else environ

        cache = conductor["cache"]
        data = cls(
            translation_api=APIConfig.from_section(
                llm["translation.api"], llm["translation.prompts"]["system"], env),
            verification_api=APIConfig.from_section(
                llm["verification.api"], llm["verification.prompts"]["system"], env),
            translation_strict=TranslationConfig.from_sections(
                llm["translation.prompts"]["strict"], llm["translation.sampling.strict"],
                llm["translation.sampling"]),
            translation_style=TranslationConfig.from_sections(
                llm["translation.prompts"]["style"], llm["translation.sampling.style"],
                llm["translation.sampling"]),
            verification=TranslationConfig.from_sections(
                llm["verification.prompts"]["edit"], llm["verification.sampling"],
                llm["verification.sampling"]),
            preflight=PreflightConfig(**llm["preflight"]),
            limits=LimitsConfig(**conductor["limits"]),
            heartbeat=HeartbeatConfig(**conductor["heartbeat"]),
            resilience=ResilienceConfig(**conductor["resilience"]),
            validation=ValidationConfig(
                **conductor["validation"],
                allowed_scripts=tuple(llm["validation"]["allowed_scripts"]),
            ),
            profile_name=llm["meta"]["name"],
            profile_description=llm["meta"]["description"],
            log_level=conductor["logging"]["level"].upper(),
            log_dir=cls._resolve_dir(conductor["logging"]["dir"], base_dir),
            log_retention_days=conductor["logging"]["retention_days"],
            log_extended_diagnostic=conductor["logging"]["extended_diagnostic"],
            cache_enabled=cache["enabled"],
            cache_dir=cls._resolve_dir(cache["dir"], base_dir),
            cache_max_size=cache["max_size"],
            cache_expiry_days=cache["expiry_days"],
            cache_auto_save_interval=cache["auto_save_interval"],
            glossary_patterns_cache_max_size=cache["glossary_patterns_max_size"],
            value_deduplication_enabled=conductor["dedup"]["enabled"],
        )
        data.validate()
        return data

    @staticmethod
    def _resolve_dir(path: str, base_dir: Optional[str]) -> str:
        """Anchor a relative directory at base_dir; empty paths stay empty for validate()."""
        if not path or base_dir is None or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))

    def validate(self) -> None:
        """
        Check every invariant of the configuration.

        Raises:
            ConfigValidationError: On the first broken invariant.
        """
        for endpoint in (self.translation_api, self.verification_api):
            endpoint.validate()

        for mode in (self.translation_strict, self.translation_style, self.verification):
            mode.validate()

        self._validate_stop_limits()
        self.preflight.validate()
        self.limits.validate()
        self.heartbeat.validate()
        self.resilience.validate()
        self.validation.validate()

        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigValidationError(
                f"log level must be one of {VALID_LOG_LEVELS}, got '{self.log_level}'")

        if self.log_retention_days < 1:
            raise ConfigValidationError(
                f"log_retention_days must be at least 1, got {self.log_retention_days}")

        if self.cache_max_size <= 0:
            raise ConfigValidationError(f"cache_max_size must be positive, got {self.cache_max_size}")

        if self.cache_expiry_days <= 0:
            raise ConfigValidationError(f"cache_expiry_days must be positive, got {self.cache_expiry_days}")

        if self.cache_auto_save_interval < 0:
            raise ConfigValidationError(
                f"cache_auto_save_interval must be non-negative, got {self.cache_auto_save_interval}"
            )

        if self.glossary_patterns_cache_max_size <= 0:
            raise ConfigValidationError(
                f"glossary_patterns_max_size must be positive, got {self.glossary_patterns_cache_max_size}"
            )

        if not self.cache_dir:
            raise ConfigValidationError("cache_dir cannot be empty")

        if not self.log_dir:
            raise ConfigValidationError("log_dir cannot be empty")

    def _validate_stop_limits(self) -> None:
        """
        Reject stop lists longer than the OpenAI API accepts.

        The cap is a property of the endpoint type, so it is checked here
        rather than in TranslationConfig, which does not know its endpoint.
        Ollama has no such limit.

        Raises:
            ConfigValidationError: If an 'openai' endpoint is paired with more
                than MAX_OPENAI_STOP_SEQUENCES stop sequences.
        """
        pairs = (
            (self.translation_api, 'translation stop', self.translation_strict),
            (self.translation_api, 'translation stop', self.translation_style),
            (self.verification_api, 'verification stop', self.verification),
        )
        for api, key, translation_config in pairs:
            if api.api_type == 'openai' and len(translation_config.stop) > MAX_OPENAI_STOP_SEQUENCES:
                raise ConfigValidationError(
                    f"{key} has {len(translation_config.stop)} sequences; "
                    f"api_type='openai' allows at most {MAX_OPENAI_STOP_SEQUENCES}"
                )


@dataclass(frozen=True)
class RunOverrides:
    """
    Command-line values that replace conductor.toml settings for one run.

    None leaves the file value in place - the command line wins only where
    it says something (CLI > files > defaults).
    """

    log_level: Optional[str] = None
    cache_enabled: Optional[bool] = None
    cache_dir: Optional[str] = None

    def apply(self, data: ConfigData) -> ConfigData:
        """
        Build a configuration with the overridden values.

        Args:
            data: Configuration loaded from the files.

        Returns:
            A new validated ConfigData; data itself when nothing is overridden.

        Raises:
            ConfigValidationError: If an overridden value breaks an invariant.
        """
        changes: Dict[str, Any] = {}
        if self.log_level is not None:
            changes['log_level'] = self.log_level.upper()
        if self.cache_enabled is not None:
            changes['cache_enabled'] = self.cache_enabled
        if self.cache_dir is not None:
            changes['cache_dir'] = os.path.abspath(self.cache_dir)
        if not changes:
            return data

        overridden = replace(data, **changes)
        overridden.validate()
        return overridden


class RuntimeConfigFactory:
    """
    Turns ConfigData into the flat runtime dictionaries of the API call.

    Every call returns a new dictionary, so callers may extend it (the
    entry point adds 'mode') without touching shared state.
    """

    def __init__(self, config_data: ConfigData):
        self.config_data = config_data

    @staticmethod
    def _build(api: APIConfig, sampling: TranslationConfig) -> dict:
        """Flatten one endpoint and one mode into the runtime_config contract of call_model_api()."""
        return {
            'api_url': api.url,
            'api_type': api.api_type,
            'api_key': api.api_key,
            'api_timeout': api.timeout,
            'max_retries': api.max_retries,
            'retry_backoff_factor': api.retry_backoff_factor,
            'model': api.model,
            'system_prompt': api.system_prompt,
            'keep_alive': api.keep_alive,
            'max_tokens_param': api.max_tokens_param,
            'prompt': sampling.prompt,
            'temp': sampling.temp,
            'num_predict': sampling.num_predict,
            'num_ctx': sampling.num_ctx,
            'top_p': sampling.top_p,
            'top_k': sampling.top_k,
            'repeat_penalty': sampling.repeat_penalty,
            'frequency_penalty': sampling.frequency_penalty,
            'presence_penalty': sampling.presence_penalty,
            'stop': list(sampling.stop),
        }

    def get_translation_config_for_mode(self, mode: str) -> dict:
        """
        Runtime configuration of the translation stage.

        Args:
            mode: MODE_STRICT; any other value selects the style mode.

        Returns:
            A new runtime_config dictionary.
        """
        if mode == MODE_STRICT:
            trans_config = self.config_data.translation_strict
        else:
            trans_config = self.config_data.translation_style
        return self._build(self.config_data.translation_api, trans_config)

    def get_verification_config(self) -> dict:
        """
        Runtime configuration of the verification stage.

        Returns:
            A new runtime_config dictionary.
        """
        return self._build(self.config_data.verification_api, self.config_data.verification)


class ConfigFileLoader:
    """
    Reads conductor.toml and llm.toml and builds a validated ConfigData.

    Stateless apart from its collaborators; the 'check' command uses it
    directly to validate the files without creating logs or caches.
    """

    def __init__(self, logger: Optional[Logger] = None,
                 file_ops: Optional[FileOperationsProtocol] = None,
                 environ: Optional[Mapping[str, str]] = None):
        """
        Args:
            logger: Logger for read errors; a console-only logger by default.
            file_ops: File operations for reading; default FileOperations.
            environ: Environment for key_env lookups; os.environ by default.
        """
        self._logger = logger or ConfigLogger.create(log_to_file=False)
        self._file_ops = file_ops or FileOperations(logger=self._logger)
        self._environ = environ

    def read_document(self, path: str, description: str) -> Dict[str, Any]:
        """
        Read and parse one TOML file.

        Args:
            path: Path to the file.
            description: Human-readable name of the file for messages.

        Returns:
            The parsed document.

        Raises:
            FileOperationError: If the file cannot be read.
            ConfigValidationError: If the file is not valid TOML.
        """
        # critical=True: safe_read_file either returns the text or raises.
        text = cast(str, safe_read_file(path, description, self._file_ops, self._logger,
                                        critical=True, as_text=True))
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigValidationError(f"Invalid {path}: not valid TOML ({e})") from e

    def load(self, conductor_path: str, llm_path: str,
             overrides: Optional[RunOverrides] = None) -> ConfigData:
        """
        Load both files into a validated configuration.

        Relative directories of conductor.toml are anchored at its own directory.

        Args:
            conductor_path: Path to conductor.toml.
            llm_path: Path to the model profile.
            overrides: Command-line values applied on top of the files.

        Returns:
            The validated configuration.

        Raises:
            FileOperationError: If a file cannot be read.
            ConfigValidationError: If a file or an overridden value is invalid.
        """
        conductor_doc = self.read_document(conductor_path, f"configuration {conductor_path}")
        llm_doc = self.read_document(llm_path, f"model profile {llm_path}")
        data = ConfigData.from_toml(
            llm_doc, conductor_doc,
            base_dir=os.path.dirname(os.path.abspath(conductor_path)),
            environ=self._environ,
            llm_source=llm_path,
            conductor_source=conductor_path,
        )
        return overrides.apply(data) if overrides is not None else data


MetricsFactory = Callable[[], MetricsProtocol]
LeafOutcomesFactory = Callable[[], LeafOutcomeRecorderProtocol]
UsageLedgerFactory = Callable[[], UsageLedgerProtocol]
CacheManagerFactory = Callable[['Config'], CacheManagerProtocol]
GlossaryCacheFactory = Callable[[int], GlossaryPatternCacheProtocol]
CircuitBreakerFactory = Callable[[int], CircuitBreakerProtocol]


class Config:
    """
    Configuration facade and DI container of a run.

    Builds every manager in the constructor through replaceable factories,
    so a Config is never observed half-initialized; components receive it
    through narrow protocols and never import this module.
    """

    def __init__(
        self,
        config_data: ConfigData,
        logger: Logger,
        metrics_factory: Optional[MetricsFactory] = None,
        leaf_outcomes_factory: Optional[LeafOutcomesFactory] = None,
        usage_ledger_factory: Optional[UsageLedgerFactory] = None,
        cache_manager_factory: Optional[CacheManagerFactory] = None,
        glossary_cache_factory: Optional[GlossaryCacheFactory] = None,
        circuit_breaker_factory: Optional[CircuitBreakerFactory] = None,
    ):
        """
        Args:
            config_data: Validated configuration.
            logger: Logger of the run.
            metrics_factory: Builds the metrics collector; MetricsCollector by default.
            leaf_outcomes_factory: Builds the per-leaf outcome recorder;
                LeafOutcomeRecorder by default.
            usage_ledger_factory: Builds the per-attempt usage ledger;
                UsageLedger by default.
            cache_manager_factory: Builds the reply cache from this Config
                (its logger, file_ops and cache settings are ready by then);
                _default_cache_manager_factory by default.
            glossary_cache_factory: Builds the glossary pattern cache from
                its maximum size; _default_glossary_cache_factory by default.
            circuit_breaker_factory: Builds the consecutive-failure tracker
                from [resilience].max_consecutive_failures;
                _default_circuit_breaker_factory by default.
        """
        self.data = config_data
        self.logger = logger
        self._started_at = time.monotonic()

        self.metrics: MetricsProtocol = (metrics_factory or MetricsCollector)()
        self.leaf_outcomes: LeafOutcomeRecorderProtocol = (
            leaf_outcomes_factory or LeafOutcomeRecorder)()
        self.usage_ledger: UsageLedgerProtocol = (
            usage_ledger_factory or UsageLedger)()
        self.circuit_breaker: CircuitBreakerProtocol = (
            circuit_breaker_factory or Config._default_circuit_breaker_factory)(
            config_data.resilience.max_consecutive_failures)
        self.file_ops: FileOperationsProtocol = FileOperations(
            logger=logger,
            timeout=config_data.limits.file_operation_timeout,
            large_file_threshold_mb=config_data.limits.large_file_threshold_mb,
        )
        self.runtime_factory = RuntimeConfigFactory(config_data)

        self.cache_manager: CacheManagerProtocol = (
            cache_manager_factory or Config._default_cache_manager_factory)(self)
        self.glossary_patterns_cache: GlossaryPatternCacheProtocol = (
            glossary_cache_factory or Config._default_glossary_cache_factory)(
            config_data.glossary_patterns_cache_max_size)

    @staticmethod
    def _default_cache_manager_factory(config: 'Config') -> CacheManagerProtocol:
        """
        Build the production reply cache.

        Args:
            config: The Config under construction; supplies CacheConfigProtocol.

        Returns:
            A CacheManager.
        """
        return CacheManager(config=config)

    @staticmethod
    def _default_glossary_cache_factory(max_size: int) -> GlossaryPatternCacheProtocol:
        """
        Build the production glossary pattern cache.

        Args:
            max_size: Maximum number of compiled patterns.

        Returns:
            A GlossaryPatternCache.
        """
        return GlossaryPatternCache(max_size=max_size)

    @staticmethod
    def _default_circuit_breaker_factory(max_consecutive_failures: int) -> CircuitBreakerProtocol:
        """
        Build the production consecutive-failure tracker.

        Args:
            max_consecutive_failures: See [resilience].max_consecutive_failures.

        Returns:
            A ConsecutiveFailureTracker.
        """
        return ConsecutiveFailureTracker(max_consecutive_failures)

    @property
    def is_debug_mode(self) -> bool:
        """Delegates to ConfigData."""
        return self.data.is_debug_mode

    @property
    def log_level(self) -> str:
        """Delegates to ConfigData."""
        return self.data.log_level

    @property
    def log_dir(self) -> str:
        """Delegates to ConfigData."""
        return self.data.log_dir

    @property
    def log_retention_days(self) -> int:
        """Delegates to ConfigData."""
        return self.data.log_retention_days

    @property
    def log_extended_diagnostic(self) -> bool:
        """Delegates to ConfigData."""
        return self.data.log_extended_diagnostic

    def close_log_files(self, always_keep: bool = False) -> None:
        """
        Close this run's log file handlers; keep, condense or delete its log files.

        [logging].level decides what survives a clean run (no ERROR logged,
        always_keep False): DEBUG or extended_diagnostic keep every file
        exactly as written - a deliberately verbose run is exactly the one
        whose detail must not disappear; VERBOSE keeps a condensed run
        summary (_render_run_summary()) in its place; quieter levels remove
        the file(s) entirely, as before VERBOSE existed. A no-op when
        self.logger was not built by ConfigLogger.create() (e.g. a test
        double), since there is then nothing to close.

        Args:
            always_keep: See ConfigLogger.close_log_files().
        """
        if not isinstance(self.logger, ConfigLogger):
            return

        keep_everything = always_keep or self.log_level == 'DEBUG' or self.log_extended_diagnostic
        header = None if keep_everything else (
            self._render_run_summary() if self.log_level == 'VERBOSE' else None
        )
        self.logger.close_log_files(keep_everything, header)

    def _render_run_summary(self) -> str:
        """
        Condensed header left in place of a VERBOSE clean run's removed detail.

        Covers what the removed detail would otherwise be the only record
        of: the run's parameters, its wall-clock duration since this Config
        was built, and the per-stage cache/request/token numbers
        print_api_metrics()/print_token_metrics()/CacheManager.print_stats()
        log as the run goes - shown here even when zero, unlike those
        methods, since a short summary that is sometimes silently empty
        would be confusing on its own.

        Returns:
            A short multi-line text block.
        """
        elapsed = time.monotonic() - self._started_at
        metrics_snapshot = self.metrics.get_snapshot()

        lines = [
            "RUN SUMMARY (detail removed - a clean run at VERBOSE keeps only this)",
            f"Profile: {self.profile_name}",
            f"Translation model: {self.translation_api.model} ({self.translation_api.api_type})",
            f"Verification model: {self.verification_api.model} ({self.verification_api.api_type})",
            f"Duration: {elapsed:.2f}s",
        ]

        for operation_type, label in (
            (OPERATION_TRANSLATION, "Translation"),
            (OPERATION_VERIFICATION, "Verification"),
        ):
            metrics = metrics_snapshot.get(operation_type, {})
            cache_stats = self.cache_manager.get_stats(operation_type)
            lines.append(
                f"{label}: {metrics.get('total_requests', 0)} requests, "
                f"{metrics.get('failed_requests', 0)} failed, "
                f"{metrics.get('rejected_responses', 0)} rejected, "
                f"{metrics.get('total_tokens', 0):,} tokens; "
                f"cache {cache_stats['hits']}/{cache_stats['total_requests']} hits"
            )

        return "\n".join(lines)

    @property
    def cache_enabled(self) -> bool:
        """Delegates to ConfigData."""
        return self.data.cache_enabled

    @property
    def cache_max_size(self) -> int:
        """Delegates to ConfigData."""
        return self.data.cache_max_size

    @property
    def cache_expiry_days(self) -> int:
        """Delegates to ConfigData."""
        return self.data.cache_expiry_days

    @property
    def cache_auto_save_interval(self) -> int:
        """Delegates to ConfigData."""
        return self.data.cache_auto_save_interval

    @property
    def cache_dir(self) -> str:
        """Delegates to ConfigData."""
        return self.data.cache_dir

    @property
    def value_deduplication_enabled(self) -> bool:
        """Delegates to ConfigData."""
        return self.data.value_deduplication_enabled

    @property
    def glossary_patterns_cache_max_size(self) -> int:
        """Delegates to ConfigData."""
        return self.data.glossary_patterns_cache_max_size

    @property
    def max_recursion_depth(self) -> int:
        """Delegates to ConfigData.limits."""
        return self.data.limits.max_recursion_depth

    @property
    def dedup_log_interval(self) -> int:
        """Delegates to ConfigData.limits."""
        return self.data.limits.dedup_log_interval

    @property
    def max_error_display(self) -> int:
        """Delegates to ConfigData.limits."""
        return self.data.limits.max_error_display

    @property
    def profile_name(self) -> str:
        """Delegates to ConfigData."""
        return self.data.profile_name

    @property
    def preflight(self) -> PreflightConfig:
        """Delegates to ConfigData."""
        return self.data.preflight

    @property
    def heartbeat_interval(self) -> int:
        """Delegates to ConfigData.heartbeat."""
        return self.data.heartbeat.interval

    @property
    def resilience(self) -> ResilienceConfig:
        """Delegates to ConfigData."""
        return self.data.resilience

    @property
    def validation(self) -> ValidationConfig:
        """Delegates to ConfigData."""
        return self.data.validation

    @property
    def translation_api(self) -> APIConfig:
        """Delegates to ConfigData."""
        return self.data.translation_api

    @property
    def verification_api(self) -> APIConfig:
        """Delegates to ConfigData."""
        return self.data.verification_api

    def get_translation_config_for_mode(self, mode: str) -> dict:
        """Delegates to RuntimeConfigFactory."""
        return self.runtime_factory.get_translation_config_for_mode(mode)

    def get_verification_config(self) -> dict:
        """Delegates to RuntimeConfigFactory."""
        return self.runtime_factory.get_verification_config()

    def print_api_metrics(self) -> None:
        """
        Log the API performance summary for both pipeline stages.

        total_requests counts every attempt, successful or not, so a stage
        prints its block as soon as anything was attempted - a run where
        every request failed still reports a request count and a failure
        line instead of staying silent. A failure line is logged at
        WARNING to stand out from the plain request count.
        """
        metrics_snapshot = self.metrics.get_snapshot()

        for operation_type, label in (
            (OPERATION_TRANSLATION, "Translation"),
            (OPERATION_VERIFICATION, "Verification"),
        ):
            metrics = metrics_snapshot.get(operation_type, {})
            total = metrics.get('total_requests', 0)
            if total == 0:
                continue

            successful = metrics.get('successful_requests', 0)
            failed = metrics.get('failed_requests', 0)
            avg_time = metrics['total_time'] / successful if successful > 0 else 0.0

            self.logger.info(
                f"{label} API: {total} requests, "
                f"avg time: {avg_time:.4f}s, "
                f"failed: {failed}"
            )

            if failed > 0:
                self.logger.warning(
                    f"{label} API: {failed}/{total} request(s) failed "
                    f"(network: {metrics.get('network_failures', 0)}, "
                    f"parse: {metrics.get('parse_failures', 0)}, "
                    f"truncated: {metrics.get('truncated_responses', 0)}, "
                    f"refusal: {metrics.get('refusal_responses', 0)})"
                )

            rejected = metrics.get('rejected_responses', 0)
            if rejected > 0:
                self.logger.warning(
                    f"{label} API: {rejected} reply/replies rejected by content validation "
                    f"(not cached, degraded to the original value; a later cache "
                    f"rerun re-queries only those leaves and measures failure "
                    f"ratio against that smaller request set)"
                )

    def failure_ratio_violations(self) -> List[Tuple[str, float]]:
        """
        Find stages whose failure share exceeds [resilience].max_failure_ratio.

        A reply the content validator rejected counts as a failure here even
        though MetricsCollector keeps it in a separate rejected_responses
        counter rather than failed_requests (record_rejection() does not
        double-count total_requests) - from the operator's point of view a
        rejected reply degraded to the original value exactly like a network
        or parse failure would.

        Returns:
            (operation_type, actual_ratio) pairs for every stage that made
            at least one request and exceeded the threshold.
        """
        threshold = self.data.resilience.max_failure_ratio
        violations: List[Tuple[str, float]] = []

        for operation_type, metrics in self.metrics.get_snapshot().items():
            total = metrics.get('total_requests', 0)
            if total == 0:
                continue

            failed = metrics.get('failed_requests', 0) + metrics.get('rejected_responses', 0)
            ratio = failed / total
            if ratio > threshold:
                violations.append((operation_type, ratio))

        return violations

    def print_glossary_cache_stats(self) -> None:
        """Log the hit rate of the glossary pattern cache, if it was used."""
        stats = self.glossary_patterns_cache.get_stats()

        if stats['total_requests'] > 0:
            self.logger.info(
                f"Glossary pattern cache: {stats['cache_size']}/{stats['max_size']} entries, "
                f"hit rate: {stats['hit_rate']:.2f}%, "
                f"evictions: {stats['evictions']}"
            )

    def print_token_metrics(self) -> None:
        """
        Report the token consumption of both stages and of the whole session.

        Prefer UsageLedger stage totals (every HTTP attempt, including
        retries) when the ledger has recorded attempts; otherwise fall back
        to MetricsCollector so older test doubles still work.

        The report goes through the logger only (console handler + log file).
        """
        ledger_totals = self.usage_ledger.stage_token_totals()
        metrics_snapshot = self.metrics.get_snapshot()
        t_metrics = metrics_snapshot.get(OPERATION_TRANSLATION, {})
        v_metrics = metrics_snapshot.get(OPERATION_VERIFICATION, {})

        def _stage(operation: str, metrics: dict) -> dict:
            ledger = ledger_totals.get(operation, {})
            if (ledger.get('prompt_tokens', 0) or ledger.get('completion_tokens', 0)
                    or ledger.get('total_tokens', 0)):
                return ledger
            return metrics

        t_view = _stage(OPERATION_TRANSLATION, t_metrics)
        v_view = _stage(OPERATION_VERIFICATION, v_metrics)
        total_consumed = t_view.get('total_tokens', 0) + v_view.get('total_tokens', 0)

        report = (
            "\n"
            "====================================================================\n"
            "                    TOKEN CONSUMPTION SUMMARY                      \n"
            "====================================================================\n"
            f" TRANSLATION STAGE (step 1 - {self.translation_api.api_type.upper()}):\n"
            f"   - Prompt Tokens:     {t_view.get('prompt_tokens', 0):,}\n"
            f"   - Completion Tokens: {t_view.get('completion_tokens', 0):,}\n"
            f"   - Total Tokens:      {t_view.get('total_tokens', 0):,}\n"
            "--------------------------------------------------------------------\n"
            f" VERIFICATION STAGE (step 2 - {self.verification_api.api_type.upper()}):\n"
            f"   - Prompt Tokens:     {v_view.get('prompt_tokens', 0):,}\n"
            f"   - Completion Tokens: {v_view.get('completion_tokens', 0):,}\n"
            f"   - Total Tokens:      {v_view.get('total_tokens', 0):,}\n"
            "====================================================================\n"
            f" SESSION TOTAL:         {total_consumed:,} tokens\n"
            "===================================================================="
        )

        self.logger.info(report)

    def print_leaf_outcome_banner(self) -> None:
        """Log untranslated / degraded / verify-skipped counts for the run."""
        lines = self.leaf_outcomes.banner_lines()
        box = (
            "\n"
            "====================================================================\n"
            + "\n".join(lines)
            + "\n===================================================================="
        )
        self.logger.info(box)

    @staticmethod
    def create_from_files(
        conductor_path: str,
        llm_path: str,
        log_to_file: bool = True,
        metrics_factory: Optional[MetricsFactory] = None,
        cache_manager_factory: Optional[CacheManagerFactory] = None,
        glossary_cache_factory: Optional[GlossaryCacheFactory] = None,
        circuit_breaker_factory: Optional[CircuitBreakerFactory] = None,
        environ: Optional[Mapping[str, str]] = None,
        overrides: Optional[RunOverrides] = None,
        log_name: Optional[str] = None,
    ) -> 'Config':
        """
        Load both configuration files and build the run's Config.

        Args:
            conductor_path: Path to conductor.toml.
            llm_path: Path to the model profile.
            log_to_file: Also write a log file into [logging].dir.
            metrics_factory: See Config.__init__.
            cache_manager_factory: See Config.__init__.
            glossary_cache_factory: See Config.__init__.
            circuit_breaker_factory: See Config.__init__.
            environ: Environment for key_env lookups; os.environ by default.
            overrides: Command-line values applied on top of the files.
            log_name: Name of the run target, made part of the log file name.

        Returns:
            A fully initialized Config.

        Raises:
            FileOperationError: If a file is missing or unreadable.
            ConfigValidationError: If a file or an overridden value is invalid.
        """
        config_data = ConfigFileLoader(environ=environ).load(conductor_path, llm_path, overrides)

        logger = ConfigLogger.create(log_level=config_data.log_level, log_to_file=log_to_file,
                                     log_dir=config_data.log_dir, log_name=log_name,
                                     retention_days=config_data.log_retention_days,
                                     extended_diagnostic=config_data.log_extended_diagnostic)

        return Config(
            config_data,
            logger,
            metrics_factory=metrics_factory,
            cache_manager_factory=cache_manager_factory,
            glossary_cache_factory=glossary_cache_factory,
            circuit_breaker_factory=circuit_breaker_factory,
        )
