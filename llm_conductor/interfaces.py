# -*- coding: utf-8 -*-
"""
Logger ABC and the narrow Protocol interfaces components depend on.

Members a component only reads are declared as read-only properties: a
plain attribute and a @property both satisfy them, so the Config facade can
expose its fields as delegating properties and test doubles can use plain
attributes.
"""
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable


# ============================================================================
# LOGGING INTERFACE (Logger ABC)
# ============================================================================
class Logger(ABC):
    """Abstract base class of the logging subsystem."""

    @abstractmethod
    def trace(self, message: str) -> None:
        """Finest diagnostic level; written only when extended diagnostics are on."""

    @abstractmethod
    def debug(self, message: str) -> None:
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str, include_traceback: bool = False) -> None:
        pass

# ============================================================================
# API CONFIGURATION PROTOCOLS
# ============================================================================
@runtime_checkable
class APIEndpointProtocol(Protocol):
    """Minimal description of one API endpoint, used by the API clients."""

    @property
    def url(self) -> str: ...
    @property
    def api_type(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def timeout(self) -> int: ...
    @property
    def max_retries(self) -> int: ...
    @property
    def retry_backoff_factor(self) -> float: ...

@runtime_checkable
class APIConfigProtocol(APIEndpointProtocol, Protocol):
    """Endpoint description together with its system prompt and key."""

    @property
    def system_prompt(self) -> str: ...
    @property
    def api_key(self) -> Optional[str]: ...
    @property
    def max_tokens_param(self) -> str: ...

    def validate(self) -> None: ...

# ============================================================================
# METRICS PROTOCOL
# ============================================================================
@runtime_checkable
class MetricsProtocol(Protocol):
    """Thread-safe collector of API request metrics."""

    def update(self, operation_type: str, duration: float, failed: bool,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               failure_kind: Optional[str] = None) -> None: ...

    def record_rejection(self, operation_type: str) -> None: ...

    def get_snapshot(self) -> dict: ...


@runtime_checkable
class LeafOutcomeRecorderProtocol(Protocol):
    """Per-leaf stage outcomes for run reports and verify-skip."""

    def set_active_file(self, relative_path: str) -> None: ...
    def note_sidecar_missing(self) -> bool: ...
    def record(self, leaf_path: str, stage: str, record: Any) -> None: ...
    def is_translation_degraded(self, leaf_path: str) -> bool: ...
    def merge_sidecar(self, relative_path: str, payload: dict) -> None: ...
    def file_sidecar_dict(self, relative_path: str) -> dict: ...
    def aggregates(self) -> dict: ...
    def has_untreated_error(self) -> bool: ...
    def build_run_report(self) -> dict: ...
    def banner_lines(self) -> List[str]: ...


@runtime_checkable
class UsageLedgerProtocol(Protocol):
    """Per-attempt token usage for cost-estimation exports."""

    def record_attempt(
        self,
        *,
        provider: str,
        model: str,
        stage: str,
        attempt: int,
        outcome: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        failure_kind: Optional[str] = None,
        usage_unknown: bool = False,
    ) -> int: ...

    def record_non_api(self, stage: str, kind: str, count: int = 1) -> None: ...
    def mark_attempt_rejected(
        self,
        attempt_id: int,
        *,
        failure_kind: Optional[str] = None,
    ) -> bool: ...
    def stage_token_totals(self) -> dict: ...
    def build_export(self) -> dict: ...

# ============================================================================
# FILE OPERATIONS
# ============================================================================
@runtime_checkable
class FileOperationsProtocol(Protocol):
    """
    File operations used by the managers and the entry point.

    Describes the public surface of FileOperations (llm_conductor/core/io.py).
    """

    @property
    def large_file_threshold_mb(self) -> int: ...

    def read_json(self, file_path: str) -> dict: ...
    def read_text_with_retry(self, file_path: str, encoding: str = 'utf-8',
                              max_retries: int = 3, retry_delay: float = 1.0) -> str: ...
    def write_json_atomic(self, file_path: str, data: dict, encoding: str = 'utf-8') -> None: ...
    def write_text_atomic(self, file_path: str, data: str, encoding: str = 'utf-8') -> None: ...
    def write_json_with_retry(self, file_path: str, data: dict,
                               max_retries: int = 3, retry_delay: float = 1.0) -> bool: ...
    def write_text_with_retry(self, file_path: str, data: str,
                               max_retries: int = 3, retry_delay: float = 1.0) -> bool: ...

@runtime_checkable
class HasLogger(Protocol):
    """Component with a logger."""

    @property
    def logger(self) -> Logger: ...

@runtime_checkable
class HasDebugMode(Protocol):
    """Component aware of the debug mode."""

    @property
    def is_debug_mode(self) -> bool: ...

@runtime_checkable
class HasFileOps(Protocol):
    """Component with file operations."""

    @property
    def file_ops(self) -> FileOperationsProtocol: ...

# ============================================================================
# CACHE MANAGER PROTOCOLS
# ============================================================================
@runtime_checkable
class CacheManagerProtocol(Protocol):
    """
    Public surface of the disk cache of model replies.

    Describes CacheManager (llm_conductor/core/cache.py) to the extent used by
    RecursiveProcessor, ShutdownManager and the entry point (ISP). The
    manager receives its FileOperations once, through CacheConfigProtocol.
    """
    def get(self, cache_type: str, cache_key: str) -> Optional[Any]: ...
    def set(self, cache_type: str, cache_key: str, value: Any,
            metadata: Optional[dict] = None, source_value: Optional[str] = None) -> None: ...
    def delete(self, operation_type: str, key: str) -> bool: ...
    def load(self, cache_type: str) -> bool: ...
    def save(self, cache_type: str) -> bool: ...
    def create_cache_key(self, operation_type: str, model: str, system_prompt: str,
                          enhanced_prompt: str, mode: str, value: str,
                          sampling_identity: str) -> str: ...
    def compute_fingerprints(self, operation_type: str, model: str, system_prompt: str,
                              prompt_template: str, hints_text: str, glossary_string: str,
                              mode: str, sampling_identity: str) -> dict: ...
    def diagnose_miss(self, cache_type: str, value: str, fingerprints: dict) -> str: ...
    def print_stats(self, cache_type: str) -> None: ...
    def get_stats(self, cache_type: str) -> dict: ...

@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Thread-safe tracker of consecutive model-call failures per operation type.

    Describes ConsecutiveFailureTracker (llm_conductor/core/circuit_breaker.py) to the
    extent used by RecursiveProcessor (ISP).
    """
    def record_success(self, operation_type: str) -> None: ...
    def record_failure(self, operation_type: str) -> bool: ...
    def is_tripped(self, operation_type: str) -> bool: ...

@runtime_checkable
class ValidationConfigProtocol(Protocol):
    """
    Settings of the model-reply sanity checks.

    Thresholds, latin_text_policy, and save_rejected_replies come from
    conductor.toml [validation]; allowed_scripts from llm.toml [validation].
    """

    @property
    def enabled(self) -> bool: ...
    @property
    def max_length_ratio(self) -> float: ...
    @property
    def max_length_slack(self) -> int: ...
    @property
    def latin_text_policy(self) -> str: ...
    @property
    def save_rejected_replies(self) -> bool: ...
    @property
    def allowed_scripts(self) -> tuple: ...

@runtime_checkable
class GlossaryPatternCacheProtocol(Protocol):
    """
    Public surface of the cache of compiled glossary regexes.

    Describes GlossaryPatternCache (llm_conductor/core/cache.py) to the extent used by
    LinguisticContext and Config.print_glossary_cache_stats() (ISP).
    """
    def get_or_compile(self, term: str, use_word_boundaries: bool = True) -> re.Pattern: ...
    def get_stats(self) -> dict: ...

# ============================================================================
# MANAGER CONFIGURATION PROTOCOLS
# ============================================================================
@runtime_checkable
class CacheConfigProtocol(Protocol):
    """Settings and services CacheManager reads once in its constructor."""

    @property
    def logger(self) -> Logger: ...
    @property
    def is_debug_mode(self) -> bool: ...
    @property
    def cache_enabled(self) -> bool: ...
    @property
    def cache_max_size(self) -> int: ...
    @property
    def cache_expiry_days(self) -> int: ...
    @property
    def cache_auto_save_interval(self) -> int: ...
    @property
    def cache_dir(self) -> str: ...
    @property
    def file_ops(self) -> FileOperationsProtocol: ...

@runtime_checkable
class GlossaryConfigProtocol(Protocol):
    """Settings used by glossary term matching."""

    @property
    def glossary_patterns_cache(self) -> GlossaryPatternCacheProtocol: ...
    @property
    def is_debug_mode(self) -> bool: ...

# ============================================================================
# PROCESSOR CONFIGURATION PROTOCOL
# ============================================================================
@runtime_checkable
class ProcessorConfigProtocol(Protocol):
    """
    Everything RecursiveProcessor, the l10n helpers and the API call read.

    The processor passes this object on to call_model_api() and
    unmask_tags(), so their needs (metrics, max_error_display) are part of it.
    """

    @property
    def logger(self) -> Logger: ...
    @property
    def is_debug_mode(self) -> bool: ...
    @property
    def log_level(self) -> str: ...
    @property
    def cache_enabled(self) -> bool: ...
    @property
    def cache_manager(self) -> CacheManagerProtocol: ...
    @property
    def cache_auto_save_interval(self) -> int: ...
    @property
    def file_ops(self) -> FileOperationsProtocol: ...
    @property
    def glossary_patterns_cache(self) -> GlossaryPatternCacheProtocol: ...
    @property
    def value_deduplication_enabled(self) -> bool: ...
    @property
    def metrics(self) -> MetricsProtocol: ...
    @property
    def leaf_outcomes(self) -> LeafOutcomeRecorderProtocol: ...
    @property
    def usage_ledger(self) -> UsageLedgerProtocol: ...
    @property
    def circuit_breaker(self) -> CircuitBreakerProtocol: ...
    @property
    def validation(self) -> ValidationConfigProtocol: ...
    @property
    def log_dir(self) -> str: ...
    @property
    def max_recursion_depth(self) -> int: ...
    @property
    def dedup_log_interval(self) -> int: ...
    @property
    def max_error_display(self) -> int: ...

    def print_glossary_cache_stats(self) -> None: ...

# ============================================================================
# RUNTIME CONFIGURATION PROTOCOL
# ============================================================================
@runtime_checkable
class RuntimeConfigFactoryProtocol(Protocol):
    """Builds the flat runtime dictionaries passed to the API call."""

    def get_translation_config_for_mode(self, mode: str) -> dict: ...
    def get_verification_config(self) -> dict: ...

# ============================================================================
# RUN-LEVEL SETTINGS PROTOCOLS
# ============================================================================
@runtime_checkable
class PreflightProtocol(Protocol):
    """Endpoint check performed before a run ([preflight] of llm.toml)."""

    @property
    def check_availability(self) -> bool: ...
    @property
    def timeout(self) -> int: ...

@runtime_checkable
class ResilienceProtocol(Protocol):
    """Failure thresholds of a run ([resilience] of conductor.toml)."""

    @property
    def max_consecutive_failures(self) -> int: ...
    @property
    def max_failure_ratio(self) -> float: ...
    @property
    def fail_on_any_error(self) -> bool: ...


@runtime_checkable
class HeartbeatProgressProtocol(Protocol):
    """
    Leaf/stage progress ticks for the run heartbeat.

    Owned by the CLI writer; RecursiveProcessor only calls these two methods
    and never learns about the target or the exit code.
    """

    def begin_stage(self, stage: str, leaves_total: int) -> None: ...
    def note_leaf(self) -> None: ...

# ============================================================================
# ROOT CONFIGURATION PROTOCOL
# ============================================================================
@runtime_checkable
class ConfigProtocol(ProcessorConfigProtocol, CacheConfigProtocol, Protocol):
    """
    Root protocol of the configuration facade.

    Meant ONLY for the entry point and llm_conductor/cli; internal components depend
    on the narrow protocols above.
    """

    @property
    def profile_name(self) -> str: ...
    @property
    def heartbeat_interval(self) -> int: ...
    @property
    def translation_api(self) -> APIConfigProtocol: ...
    @property
    def verification_api(self) -> APIConfigProtocol: ...
    @property
    def preflight(self) -> PreflightProtocol: ...
    @property
    def resilience(self) -> ResilienceProtocol: ...

    def get_translation_config_for_mode(self, mode: str) -> dict: ...
    def get_verification_config(self) -> dict: ...
    def print_api_metrics(self) -> None: ...
    def print_token_metrics(self) -> None: ...
    def print_leaf_outcome_banner(self) -> None: ...
    def failure_ratio_violations(self) -> List[Tuple[str, float]]: ...
