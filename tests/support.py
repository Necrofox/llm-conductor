# -*- coding: utf-8 -*-
"""
Test doubles and helpers shared by all tiers.

Kept out of conftest.py as a plain module so they can be imported
explicitly (`from tests.support import RecordingLogger`).
"""
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from llm_conductor.cli.init import format_toml_value
from llm_conductor.config import ValidationConfig
from llm_conductor.constants import (
    CONDUCTOR_CONFIG_FILENAME,
    DEFAULT_DEDUP_LOG_INTERVAL,
    DEFAULT_MAX_ERROR_DISPLAY,
    DEFAULT_MAX_RECURSION_DEPTH,
    DEFAULT_VALIDATION_LATIN_POLICY,
    DEFAULT_VALIDATION_MAX_LENGTH_RATIO,
    DEFAULT_VALIDATION_MAX_LENGTH_SLACK,
    EXAMPLE_EN_RU_ALLOWED_SCRIPTS,
    LLM_CONFIG_FILENAME,
)
from llm_conductor.core.cache import GlossaryPatternCache
from llm_conductor.core.circuit_breaker import ConsecutiveFailureTracker
from llm_conductor.core.leaf_report import LeafOutcomeRecorder
from llm_conductor.core.metrics import MetricsCollector
from llm_conductor.core.usage import UsageLedger
from llm_conductor.interfaces import Logger
from llm_conductor.l10n.content import DEFAULT_CONTENT_RULES
from llm_conductor.schema import CONDUCTOR_SCHEMA, LLM_SCHEMA, FileSchema, SectionPath

# Fake endpoints: the .test domain is reserved (RFC 2606) and never resolves,
# so even a missed interception cannot reach the real network.
OLLAMA_URL = "http://llm.test/api/chat"
OPENAI_URL = "http://llm.test/v1/chat/completions"


class DeadlockGuardLock:
    """
    Wrapper around threading.Lock that raises on timeout instead of hanging
    forever — turns a potential deadlock into a test failure.
    """

    def __init__(self, timeout: float = 1.0):
        self._lock = threading.Lock()
        self._timeout = timeout

    def __enter__(self):
        if not self._lock.acquire(timeout=self._timeout):
            raise RuntimeError(
                "deadlock: non-reentrant lock re-acquired by the same thread"
            )
        return self

    def __exit__(self, *exc_info):
        self._lock.release()


def ollama_reply(text: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> dict:
    """Response body of Ollama /api/chat."""
    return {"message": {"role": "assistant", "content": text},
            "prompt_eval_count": prompt_tokens, "eval_count": completion_tokens}


def openai_reply(text: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> dict:
    """Response body of an OpenAI-compatible /v1/chat/completions."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


# ============================================================================
# TOML configuration helpers
# ============================================================================

def _schema_setting_paths(schema: FileSchema) -> set:
    """Full setting paths (section segments + key) declared by a schema."""
    paths = set()
    for section in schema.sections:
        for setting in section.settings:
            paths.add(section.path + (setting.key,))
    return paths


_CONDUCTOR_SETTING_PATHS = _schema_setting_paths(CONDUCTOR_SCHEMA)
_LLM_SETTING_PATHS = _schema_setting_paths(LLM_SCHEMA)
_CONDUCTOR_TABLES = {section.path[0] for section in CONDUCTOR_SCHEMA.sections}


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return base with overrides merged in, recursively for nested tables."""
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def split_overrides(**overrides: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Turn double-underscore keyword names into (llm_doc, conductor_doc) fragments.

    `translation__api__max_retries=3` sets [translation.api].max_retries of
    llm.toml. A name whose full path is declared in conductor.toml goes
    there; a path declared only in llm.toml (e.g. validation__allowed_scripts)
    goes to llm even when the top-level table name also exists in conductor.

    Returns:
        (llm overrides, conductor overrides) as nested dicts.
    """
    llm: Dict[str, Any] = {}
    conductor: Dict[str, Any] = {}
    for name, value in overrides.items():
        path: SectionPath = tuple(name.split("__"))
        if path in _CONDUCTOR_SETTING_PATHS:
            node = conductor
        elif path in _LLM_SETTING_PATHS:
            node = llm
        else:
            # Unknown / typo keys under a conductor top-level table stay there
            # so schema rejection still names the conductor file.
            node = conductor if path[0] in _CONDUCTOR_TABLES else llm
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return llm, conductor


def dump_toml(document: Mapping[str, Any]) -> str:
    """Serialize a nested dict of scalars, lists and tables to TOML text."""
    lines: List[str] = []

    def emit(table: Mapping[str, Any], path: Tuple[str, ...]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if path and scalars:
            lines.append(f"[{'.'.join(path)}]")
        lines.extend(f"{key} = {format_toml_value(value)}" for key, value in scalars.items())
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, path + (key,))

    emit(document, ())
    return "\n".join(lines) + "\n"


def base_llm_doc() -> Dict[str, Any]:
    """llm.toml of the test tiers: Ollama on a fake host, short prompts, no stop sequences."""
    return {
        "translation": {
            "api": {"url": OLLAMA_URL, "type": "ollama", "model": "test-model"},
            "prompts": {"system": "", "strict": "Translate the text: ", "style": "Translate the text: "},
            "sampling": {"stop": []},
        },
        "verification": {
            "api": {"url": OLLAMA_URL, "type": "ollama", "model": "test-model"},
            "prompts": {"system": "", "edit": "Verify the text: "},
            "sampling": {"stop": []},
        },
    }




def write_configs(directory: Path, llm_doc: Mapping[str, Any], conductor_doc: Mapping[str, Any]) -> Path:
    """
    Write conductor.toml and llm.toml into directory.

    Returns:
        Path of conductor.toml (the value for LLMC_CONFIG).
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / LLM_CONFIG_FILENAME).write_text(dump_toml(llm_doc), encoding="utf-8")
    conductor_path = directory / CONDUCTOR_CONFIG_FILENAME
    conductor_path.write_text(dump_toml(conductor_doc), encoding="utf-8")
    return conductor_path


# ============================================================================
# Doubles
# ============================================================================

class RecordingLogger(Logger):
    """
    Logger that keeps messages in memory instead of printing them.

    is_debug_mode is an attribute as well, because safe_read_file and
    safe_write_file look it up on the logger itself.
    """

    def __init__(self, is_debug_mode: bool = False):
        self.is_debug_mode = is_debug_mode
        self.records: List[Tuple[str, str]] = []

    def trace(self, message: str) -> None:
        self.records.append(('TRACE', message))

    def debug(self, message: str) -> None:
        self.records.append(('DEBUG', message))

    def info(self, message: str) -> None:
        self.records.append(('INFO', message))

    def warning(self, message: str) -> None:
        self.records.append(('WARNING', message))

    def error(self, message: str, include_traceback: bool = False) -> None:
        self.records.append(('ERROR', message))

    def messages(self, level: Optional[str] = None) -> List[str]:
        """
        Message texts, optionally of one level only.

        Args:
            level: 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR', or None for all.

        Returns:
            Messages in the order they arrived.
        """
        return [msg for lvl, msg in self.records if level is None or lvl == level]

    def contains(self, fragment: str, level: Optional[str] = None) -> bool:
        """True if at least one message contains fragment."""
        return any(fragment in msg for msg in self.messages(level))


class StubConfig:
    """
    Plain-attribute implementation of ProcessorConfigProtocol for unit tests.

    Meant for LinguisticContext, build_glossary_string, unmask_tags and the
    helpers of RecursiveProcessor that do not touch the cache. Scenarios with
    a real CacheManager use a full Config (make_config in the integration tier).
    """

    def __init__(self, is_debug_mode: bool = False, glossary_cache: Any = 'default',
                 cache_manager: Any = None, cache_enabled: bool = False,
                 value_deduplication_enabled: bool = False, log_level: str = 'INFO',
                 circuit_breaker: Any = 'default', validation: Any = 'default'):
        self.logger = RecordingLogger(is_debug_mode=is_debug_mode)
        self.is_debug_mode = is_debug_mode
        self.log_level = log_level
        self.glossary_patterns_cache = (
            GlossaryPatternCache(max_size=100) if glossary_cache == 'default' else glossary_cache
        )
        self.cache_manager = cache_manager
        self.cache_enabled = cache_enabled
        self.processing = DEFAULT_CONTENT_RULES
        self.value_deduplication_enabled = value_deduplication_enabled
        self.cache_auto_save_interval = 0
        self.file_ops = None
        self.metrics = MetricsCollector()
        self.leaf_outcomes = LeafOutcomeRecorder()
        self.usage_ledger = UsageLedger()
        # Disabled by default (threshold 0): existing tests never trip it.
        # Tests of the breaker itself pass their own low-threshold tracker.
        self.circuit_breaker = (
            ConsecutiveFailureTracker(max_consecutive_failures=0)
            if circuit_breaker == 'default' else circuit_breaker
        )
        self.validation = (
            ValidationConfig(
                enabled=True,
                max_length_ratio=DEFAULT_VALIDATION_MAX_LENGTH_RATIO,
                max_length_slack=DEFAULT_VALIDATION_MAX_LENGTH_SLACK,
                latin_text_policy=DEFAULT_VALIDATION_LATIN_POLICY,
                allowed_scripts=EXAMPLE_EN_RU_ALLOWED_SCRIPTS,
            )
            if validation == 'default' else validation
        )
        self.log_dir = ''
        self.max_recursion_depth = DEFAULT_MAX_RECURSION_DEPTH
        self.dedup_log_interval = DEFAULT_DEDUP_LOG_INTERVAL
        self.max_error_display = DEFAULT_MAX_ERROR_DISPLAY

    def print_glossary_cache_stats(self) -> None:
        """Protocol method; unused by unit tests."""


class FakeCacheManager:
    """
    CacheManager fake that records save() calls - for ShutdownManager.

    Args:
        fail_on_save: Raise from save() (checks that callers survive a
            failing cache write).
    """

    def __init__(self, fail_on_save: bool = False):
        self.saved: List[str] = []
        self.fail_on_save = fail_on_save

    def save(self, cache_type: str) -> bool:
        if self.fail_on_save:
            raise OSError(f"disk full while saving {cache_type}")
        self.saved.append(cache_type)
        return True
