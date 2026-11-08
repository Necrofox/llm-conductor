# -*- coding: utf-8 -*-
"""
Declarative schema of the two configuration files, llm.toml and conductor.toml.

The schema is the single source of truth for every section and key: its value
type, its default and its one-line description. Strict validation of a loaded
document (unknown keys, wrong types) and the commented templates written by
the ``init`` command are both derived from it, so a key cannot exist in one
place and be missing from the other.

Value invariants (ranges, non-empty strings) are not repeated here - they
belong to the ``validate()`` methods of the dataclasses in llm_conductor/config.py.
"""
import difflib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from llm_conductor.constants import (
    CONDUCTOR_CONFIG_FILENAME,
    CONFIG_KEY_SUGGESTION_CUTOFF,
    DEFAULT_API_KEEP_ALIVE,
    DEFAULT_CACHE_AUTO_SAVE_INTERVAL,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_ENABLED,
    DEFAULT_CACHE_EXPIRY_DAYS,
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_DEDUP_ENABLED,
    DEFAULT_DEDUP_LOG_INTERVAL,
    DEFAULT_FILE_OPERATION_TIMEOUT,
    DEFAULT_GLOSSARY_CACHE_MAX_SIZE,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_LARGE_FILE_THRESHOLD_MB,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_EXTENDED_DIAGNOSTIC,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_RETENTION_DAYS,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    DEFAULT_MAX_ERROR_DISPLAY,
    DEFAULT_MAX_FAILURE_RATIO,
    DEFAULT_MAX_RECURSION_DEPTH,
    DEFAULT_MAX_TOKENS_PARAM,
    DEFAULT_PREFLIGHT_CHECK_AVAILABILITY,
    DEFAULT_PREFLIGHT_TIMEOUT,
    DEFAULT_PROFILE_DESCRIPTION,
    DEFAULT_PROFILE_NAME,
    DEFAULT_STOP_SEQUENCES,
    DEFAULT_TRANSLATION_API_TIMEOUT,
    DEFAULT_TRANSLATION_API_TYPE,
    DEFAULT_TRANSLATION_API_URL,
    DEFAULT_TRANSLATION_FREQUENCY_PENALTY,
    DEFAULT_TRANSLATION_MAX_RETRIES,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_PRESENCE_PENALTY,
    DEFAULT_TRANSLATION_RETRY_BACKOFF_FACTOR,
    DEFAULT_TRANSLATION_STRICT_NUM_CTX,
    DEFAULT_TRANSLATION_STRICT_NUM_PREDICT,
    DEFAULT_TRANSLATION_STRICT_PROMPT,
    DEFAULT_TRANSLATION_STRICT_REPEAT_PENALTY,
    DEFAULT_TRANSLATION_STRICT_TEMPERATURE,
    DEFAULT_TRANSLATION_STRICT_TOP_K,
    DEFAULT_TRANSLATION_STRICT_TOP_P,
    DEFAULT_TRANSLATION_STYLE_NUM_CTX,
    DEFAULT_TRANSLATION_STYLE_NUM_PREDICT,
    DEFAULT_TRANSLATION_STYLE_PROMPT,
    DEFAULT_TRANSLATION_STYLE_REPEAT_PENALTY,
    DEFAULT_TRANSLATION_STYLE_TEMPERATURE,
    DEFAULT_TRANSLATION_STYLE_TOP_K,
    DEFAULT_TRANSLATION_STYLE_TOP_P,
    DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_VALIDATION_ALLOWED_SCRIPTS,
    DEFAULT_VALIDATION_ENABLED,
    DEFAULT_VALIDATION_LATIN_POLICY,
    DEFAULT_VALIDATION_MAX_LENGTH_RATIO,
    DEFAULT_VALIDATION_MAX_LENGTH_SLACK,
    DEFAULT_VERIFICATION_API_TIMEOUT,
    DEFAULT_VERIFICATION_API_TYPE,
    DEFAULT_VERIFICATION_API_URL,
    DEFAULT_VERIFICATION_EDIT_PROMPT,
    DEFAULT_VERIFICATION_FREQUENCY_PENALTY,
    DEFAULT_VERIFICATION_MAX_RETRIES,
    DEFAULT_VERIFICATION_MODEL,
    DEFAULT_VERIFICATION_NUM_CTX,
    DEFAULT_VERIFICATION_NUM_PREDICT,
    DEFAULT_VERIFICATION_PRESENCE_PENALTY,
    DEFAULT_VERIFICATION_REPEAT_PENALTY,
    DEFAULT_VERIFICATION_RETRY_BACKOFF_FACTOR,
    DEFAULT_VERIFICATION_SYSTEM_PROMPT,
    DEFAULT_VERIFICATION_TEMPERATURE,
    DEFAULT_VERIFICATION_TOP_K,
    DEFAULT_VERIFICATION_TOP_P,
    EXAMPLE_EN_RU_ALLOWED_SCRIPTS,
    LLM_CONFIG_FILENAME,
    MAX_OPENAI_STOP_SEQUENCES,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    VALID_ALLOWED_SCRIPTS,
    VALID_API_TYPES,
    VALID_LATIN_POLICIES,
    VALID_LOG_LEVELS,
    VALID_MAX_TOKENS_PARAMS,
)
from llm_conductor.exceptions import ConfigValidationError

SectionPath = Tuple[str, ...]


class ValueKind(Enum):
    """TOML value type a setting accepts; the enum value is its human-readable name."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    STRING_LIST = "list of strings"

    def accepts(self, value: Any) -> bool:
        """
        Check a parsed TOML value against this kind.

        bool is a subclass of int in Python, so booleans are rejected
        explicitly for numeric kinds; an integer is a valid FLOAT.

        Args:
            value: Value produced by tomllib.

        Returns:
            True if the value has this kind.
        """
        if self is ValueKind.BOOLEAN:
            return isinstance(value, bool)
        if isinstance(value, bool):
            return False
        if self is ValueKind.STRING:
            return isinstance(value, str)
        if self is ValueKind.INTEGER:
            return isinstance(value, int)
        if self is ValueKind.FLOAT:
            return isinstance(value, (int, float))
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    def normalize(self, value: Any) -> Any:
        """
        Convert an accepted value to its canonical immutable Python form.

        Args:
            value: Value for which accepts() returned True.

        Returns:
            float for FLOAT, tuple for STRING_LIST, the value itself otherwise.
        """
        if self is ValueKind.FLOAT:
            return float(value)
        if self is ValueKind.STRING_LIST:
            return tuple(value)
        return value


@dataclass(frozen=True)
class Setting:
    """
    One key of a configuration section.

    Attributes:
        key: Key name inside its section.
        kind: Accepted value type.
        default: Value used when the key is absent.
        description: One line for the template comment, including the valid
            range when there is one.
        multiline: Render the default as a multi-line literal string.
        template_default: Value written by ``init`` when it must differ from
            ``default`` (example profile vs language-agnostic absent-key
            resolve); None means ``init`` writes ``default``.
    """

    key: str
    kind: ValueKind
    default: Any
    description: str
    multiline: bool = False
    template_default: Any = None

    @property
    def init_value(self) -> Any:
        """Value ``init`` writes for this key."""
        return self.default if self.template_default is None else self.template_default


@dataclass(frozen=True)
class Section:
    """
    A TOML table with its settings.

    Attributes:
        path: Table path, e.g. ("translation", "sampling", "strict").
        description: One line rendered above the table header.
        settings: Keys of the table, in template order.
    """

    path: SectionPath
    description: str
    settings: Tuple[Setting, ...]

    @property
    def name(self) -> str:
        """Dotted table name as written in a TOML header."""
        return ".".join(self.path)

    def find(self, key: str) -> Optional[Setting]:
        """
        Look up a setting by key.

        Args:
            key: Key name inside this section.

        Returns:
            The setting, or None if the section has no such key.
        """
        for setting in self.settings:
            if setting.key == key:
                return setting
        return None


class FileSchema:
    """
    Schema of one configuration file.

    Validates a parsed TOML document strictly - every unknown table or key
    and every value of a wrong type is an error - and resolves it into a
    read-only mapping with defaults filled in. Instances are immutable after
    construction and safe to share between threads.
    """

    def __init__(self, file_name: str, header: str, sections: Tuple[Section, ...]):
        """
        Args:
            file_name: Default file name, used in templates.
            header: Comment block rendered at the top of the template.
            sections: Tables of the file, in template order.

        Raises:
            ValueError: If two sections share a path.
        """
        self.file_name = file_name
        self.header = header
        self.sections = sections
        self._by_path: Dict[SectionPath, Section] = {}
        for section in sections:
            if section.path in self._by_path:
                raise ValueError(f"Duplicate section [{section.name}] in schema of {file_name}")
            self._by_path[section.path] = section
        self._children: Dict[SectionPath, Tuple[str, ...]] = self._build_children()

    def _build_children(self) -> Dict[SectionPath, Tuple[str, ...]]:
        """Map every table path, including pure containers like [translation], to its sub-table names."""
        children: Dict[SectionPath, List[str]] = {}
        for section in self.sections:
            for depth in range(len(section.path)):
                parent, child = section.path[:depth], section.path[depth]
                names = children.setdefault(parent, [])
                if child not in names:
                    names.append(child)
        return {path: tuple(names) for path, names in children.items()}

    def resolve(self, document: Mapping[str, Any], source: Optional[str] = None
                ) -> Mapping[str, Mapping[str, Any]]:
        """
        Validate a parsed document and fill in defaults.

        Args:
            document: Result of tomllib.loads().
            source: File name for error messages; defaults to the schema's file name.

        Returns:
            Read-only mapping {section name: {key: value}} covering every
            section and key of the schema. Values are normalized by
            ValueKind.normalize().

        Raises:
            ConfigValidationError: Listing every problem found in the document.
        """
        errors: List[str] = []
        self._collect_errors(document, (), errors)
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ConfigValidationError(f"Invalid {source or self.file_name}:\n{details}")

        resolved: Dict[str, Mapping[str, Any]] = {}
        for section in self.sections:
            table = self._table_at(document, section.path)
            values = {
                setting.key: setting.kind.normalize(table.get(setting.key, setting.default))
                for setting in section.settings
            }
            resolved[section.name] = MappingProxyType(values)
        return MappingProxyType(resolved)

    @staticmethod
    def _table_at(document: Mapping[str, Any], path: SectionPath) -> Mapping[str, Any]:
        """Return the table at path, or an empty mapping if the document omits it."""
        table: Mapping[str, Any] = document
        for name in path:
            table = table.get(name, {})
        return table

    def _collect_errors(self, table: Mapping[str, Any], path: SectionPath, errors: List[str]) -> None:
        """Walk one table recursively and append a message for every violation."""
        section = self._by_path.get(path)
        children = self._children.get(path, ())
        location = f"[{'.'.join(path)}]" if path else "the top level"

        for key, value in table.items():
            if key in children:
                if isinstance(value, dict):
                    self._collect_errors(value, path + (key,), errors)
                else:
                    errors.append(f"'{key}' in {location} must be a table [{'.'.join(path + (key,))}]")
                continue

            setting = section.find(key) if section else None
            if setting is None:
                known = list(children) + ([s.key for s in section.settings] if section else [])
                errors.append(f"unknown key '{key}' in {location}{self._suggestion(key, known)}")
            elif not setting.kind.accepts(value):
                errors.append(
                    f"'{key}' in {location}: expected {setting.kind.value}, "
                    f"got {type(value).__name__} {value!r}"
                )

    @staticmethod
    def _suggestion(key: str, known: List[str]) -> str:
        """Format a 'did you mean' hint for a misspelled key, or an empty string."""
        matches = difflib.get_close_matches(key, known, n=1, cutoff=CONFIG_KEY_SUGGESTION_CUTOFF)
        return f" (did you mean '{matches[0]}'?)" if matches else ""

    def all_keys(self) -> List[str]:
        """
        List every key of the schema as 'section.key'.

        Returns:
            Fully qualified key names in template order.
        """
        return [f"{section.name}.{setting.key}" for section in self.sections for setting in section.settings]


# ============================================================================
# llm.toml - everything that changes with the model or the provider
# ============================================================================

def _api_section(stage: str, url: str, api_type: str, model: str, timeout: int,
                 max_retries: int, backoff: float) -> Section:
    """Build the [<stage>.api] table; both stages share its layout."""
    return Section((stage, "api"), f"Endpoint of the {stage} stage.", (
        Setting("url", ValueKind.STRING, url,
                "Chat endpoint: Ollama '/api/chat' or OpenAI-compatible '/v1/chat/completions'."),
        Setting("type", ValueKind.STRING, api_type,
                f"Request format, one of: {', '.join(VALID_API_TYPES)}."),
        Setting("key_env", ValueKind.STRING, "",
                "Name of the environment variable holding the API key (takes priority over 'key')."),
        Setting("key", ValueKind.STRING, "",
                "Literal API key; required for 'openai' unless key_env is set. Ignored by Ollama."),
        Setting("timeout", ValueKind.INTEGER, timeout, "Request timeout, seconds (> 0)."),
        Setting("max_retries", ValueKind.INTEGER, max_retries,
                "Attempts per request for 5xx, timeouts and 429; values below 1 mean one attempt."),
        Setting("retry_backoff_factor", ValueKind.FLOAT, backoff,
                "Exponential backoff base: wait = factor ** attempt seconds."),
        Setting("model", ValueKind.STRING, model, "Model name as the provider knows it."),
        Setting("keep_alive", ValueKind.STRING, DEFAULT_API_KEEP_ALIVE,
                "How long Ollama keeps the model in VRAM after a reply "
                "(\"5m\", \"0\" unload now, \"-1\" forever). Empty = server default. "
                "Ignored by OpenAI-compatible endpoints."),
        Setting("max_tokens_param", ValueKind.STRING, DEFAULT_MAX_TOKENS_PARAM,
                f"Name of the reply-length field for OpenAI-compatible APIs, "
                f"one of: {', '.join(VALID_MAX_TOKENS_PARAMS)}. "
                f"Use max_completion_tokens for o-series and other models that "
                f"reject max_tokens. Ignored by Ollama."),
    ))


def _mode_sampling_settings(temperature: float, top_p: float, top_k: int, repeat_penalty: float,
                            num_predict: int, num_ctx: int) -> Tuple[Setting, ...]:
    """Sampling keys that each translation mode and the verification stage set independently."""
    return (
        Setting("temperature", ValueKind.FLOAT, temperature,
                f"Randomness, {MIN_TEMPERATURE}..{MAX_TEMPERATURE}."),
        Setting("top_p", ValueKind.FLOAT, top_p, f"Nucleus sampling, {MIN_TOP_P}..{MAX_TOP_P}."),
        Setting("top_k", ValueKind.INTEGER, top_k, "Candidate tokens per step (Ollama only)."),
        Setting("repeat_penalty", ValueKind.FLOAT, repeat_penalty,
                "Penalty for repeated tokens, 1.0 = off (Ollama only)."),
        Setting("num_predict", ValueKind.INTEGER, num_predict,
                "Max tokens of the reply (> 0); sent to OpenAI-compatible APIs "
                "under the name chosen by [*.api].max_tokens_param."),
        Setting("num_ctx", ValueKind.INTEGER, num_ctx, "Context window, tokens (> 0, Ollama only)."),
    )


def _shared_sampling_settings(frequency_penalty: float, presence_penalty: float) -> Tuple[Setting, ...]:
    """Sampling keys shared by both translation modes; verification sets them in its own table."""
    return (
        Setting("frequency_penalty", ValueKind.FLOAT, frequency_penalty,
                "Penalty by token frequency (OpenAI-compatible only)."),
        Setting("presence_penalty", ValueKind.FLOAT, presence_penalty,
                "Penalty for any repeated token (OpenAI-compatible only)."),
        Setting("stop", ValueKind.STRING_LIST, DEFAULT_STOP_SEQUENCES,
                f"Stop sequences; at most {MAX_OPENAI_STOP_SEQUENCES} for 'openai'. "
                f"Ollama replaces the model's own stop tokens with this list."),
    )


LLM_SCHEMA = FileSchema(
    LLM_CONFIG_FILENAME,
    "Model profile: endpoints, keys, prompts and sampling.\n"
    "Switch profiles with --llm-config PATH. Prompt rules: doc/prompts.md.\n"
    "The model sees markers [T0], [T1] instead of {0}, <tags> and newlines.",
    (
        Section(("meta",), "Profile identity, printed in the run banner and the log.", (
            Setting("name", ValueKind.STRING, DEFAULT_PROFILE_NAME, "Short profile name."),
            Setting("description", ValueKind.STRING, DEFAULT_PROFILE_DESCRIPTION, "Free-form description."),
        )),
        Section(("preflight",), "Endpoint check before the run (sends a 1-token request).", (
            Setting("check_availability", ValueKind.BOOLEAN, DEFAULT_PREFLIGHT_CHECK_AVAILABILITY,
                    "Check the endpoints before processing; false skips the check."),
            Setting("timeout", ValueKind.INTEGER, DEFAULT_PREFLIGHT_TIMEOUT, "Check timeout, seconds (> 0)."),
        )),
        _api_section("translation", DEFAULT_TRANSLATION_API_URL, DEFAULT_TRANSLATION_API_TYPE,
                     DEFAULT_TRANSLATION_MODEL, DEFAULT_TRANSLATION_API_TIMEOUT,
                     DEFAULT_TRANSLATION_MAX_RETRIES, DEFAULT_TRANSLATION_RETRY_BACKOFF_FACTOR),
        Section(("translation", "prompts"),
                "Prompts. The leaf text is appended right after the user prompt, so end it with a label "
                "and a newline.", (
                    Setting("system", ValueKind.STRING, DEFAULT_TRANSLATION_SYSTEM_PROMPT,
                            "System prompt of both translation modes.", multiline=True),
                    Setting("strict", ValueKind.STRING, DEFAULT_TRANSLATION_STRICT_PROMPT,
                            "User prompt of --strict; {LINGUISTIC_CONTEXT} and {GLOSSARY} mark where "
                            "hints and glossary go.", multiline=True),
                    Setting("style", ValueKind.STRING, DEFAULT_TRANSLATION_STYLE_PROMPT,
                            "User prompt of the default (style) mode.", multiline=True),
                )),
        Section(("translation", "sampling"), "Sampling shared by both translation modes.",
                _shared_sampling_settings(DEFAULT_TRANSLATION_FREQUENCY_PENALTY,
                                          DEFAULT_TRANSLATION_PRESENCE_PENALTY)),
        Section(("translation", "sampling", "strict"), "Sampling of --strict (precise UI text).",
                _mode_sampling_settings(DEFAULT_TRANSLATION_STRICT_TEMPERATURE, DEFAULT_TRANSLATION_STRICT_TOP_P,
                                        DEFAULT_TRANSLATION_STRICT_TOP_K, DEFAULT_TRANSLATION_STRICT_REPEAT_PENALTY,
                                        DEFAULT_TRANSLATION_STRICT_NUM_PREDICT, DEFAULT_TRANSLATION_STRICT_NUM_CTX)),
        Section(("translation", "sampling", "style"), "Sampling of the style mode (dialogue, narrative).",
                _mode_sampling_settings(DEFAULT_TRANSLATION_STYLE_TEMPERATURE, DEFAULT_TRANSLATION_STYLE_TOP_P,
                                        DEFAULT_TRANSLATION_STYLE_TOP_K, DEFAULT_TRANSLATION_STYLE_REPEAT_PENALTY,
                                        DEFAULT_TRANSLATION_STYLE_NUM_PREDICT, DEFAULT_TRANSLATION_STYLE_NUM_CTX)),
        _api_section("verification", DEFAULT_VERIFICATION_API_URL, DEFAULT_VERIFICATION_API_TYPE,
                     DEFAULT_VERIFICATION_MODEL, DEFAULT_VERIFICATION_API_TIMEOUT,
                     DEFAULT_VERIFICATION_MAX_RETRIES, DEFAULT_VERIFICATION_RETRY_BACKOFF_FACTOR),
        Section(("verification", "prompts"),
                "Prompts of the proofreading stage.", (
                    Setting("system", ValueKind.STRING, DEFAULT_VERIFICATION_SYSTEM_PROMPT,
                            "System prompt of the verification stage.", multiline=True),
                    Setting("edit", ValueKind.STRING, DEFAULT_VERIFICATION_EDIT_PROMPT,
                            "User prompt; {LINGUISTIC_CONTEXT} and {GLOSSARY} as in the translation prompts, "
                            "plus {ORIGINAL_TEXT} for the masked English source (alignment reference only).",
                            multiline=True),
                )),
        Section(("verification", "sampling"), "Sampling of the verification stage (a single mode).",
                _mode_sampling_settings(DEFAULT_VERIFICATION_TEMPERATURE, DEFAULT_VERIFICATION_TOP_P,
                                        DEFAULT_VERIFICATION_TOP_K, DEFAULT_VERIFICATION_REPEAT_PENALTY,
                                        DEFAULT_VERIFICATION_NUM_PREDICT, DEFAULT_VERIFICATION_NUM_CTX)
                + _shared_sampling_settings(DEFAULT_VERIFICATION_FREQUENCY_PENALTY,
                                            DEFAULT_VERIFICATION_PRESENCE_PENALTY)),
        # Language-dependent half of reply validation: travels with --llm-config.
        # Thresholds/policy (enabled, length, latin_text_policy) stay in conductor.toml.
        Section(("validation",),
                "Target writing systems for the reply sanity check (both stages). "
                "Other [validation] keys live in conductor.toml.", (
                    Setting(
                        "allowed_scripts", ValueKind.STRING_LIST, DEFAULT_VALIDATION_ALLOWED_SCRIPTS,
                        "Writing systems always allowed in model replies (besides characters "
                        "already in the source or glossary). Empty = none. Known names: "
                        + ", ".join(VALID_ALLOWED_SCRIPTS)
                        + ". Absent key resolves empty; the EN→RU example profile sets "
                        "[\"cyrillic\"].",
                        template_default=EXAMPLE_EN_RU_ALLOWED_SCRIPTS,
                    ),
                )),
    ),
)

# ============================================================================
# conductor.toml - behaviour of the tool, independent of the model
# ============================================================================

CONDUCTOR_SCHEMA = FileSchema(
    CONDUCTOR_CONFIG_FILENAME,
    "Behaviour of LLM-Conductor, independent of the model profile.\n"
    "Relative paths are resolved from the directory of this file.",
    (
        Section(("logging",), "Run log.", (
            Setting("level", ValueKind.STRING, DEFAULT_LOG_LEVEL,
                    f"One of: {', '.join(VALID_LOG_LEVELS)}. DEBUG/VERBOSE add diagnostics."),
            Setting("dir", ValueKind.STRING, DEFAULT_LOG_DIR, "Directory of log files; created if missing."),
            Setting("retention_days", ValueKind.INTEGER, DEFAULT_LOG_RETENTION_DAYS,
                    "Rotated daily log files kept in [logging].dir (>= 1); the oldest is "
                    "deleted at midnight rotation."),
            Setting("extended_diagnostic", ValueKind.BOOLEAN, DEFAULT_LOG_EXTENDED_DIAGNOSTIC,
                    "Also write full prompts and cache-key hashes to a separate TRACE-level "
                    "log file."),
        )),
        Section(("cache",), "Disk cache of model replies.", (
            Setting("enabled", ValueKind.BOOLEAN, DEFAULT_CACHE_ENABLED, "Read and write the reply cache."),
            Setting("dir", ValueKind.STRING, DEFAULT_CACHE_DIR, "Directory of cache files; created if missing."),
            Setting("max_size", ValueKind.INTEGER, DEFAULT_CACHE_MAX_SIZE,
                    "Entries per cache file (> 0); the oldest are evicted first."),
            Setting("expiry_days", ValueKind.INTEGER, DEFAULT_CACHE_EXPIRY_DAYS,
                    "Entries older than this are dropped (> 0)."),
            Setting("auto_save_interval", ValueKind.INTEGER, DEFAULT_CACHE_AUTO_SAVE_INTERVAL,
                    "Flush to disk after this many new entries (>= 0, 0 = only at the end)."),
            Setting("glossary_patterns_max_size", ValueKind.INTEGER, DEFAULT_GLOSSARY_CACHE_MAX_SIZE,
                    "In-memory cache of compiled glossary term patterns (> 0)."),
        )),
        Section(("dedup",), "Reuse of replies for identical strings within one run.", (
            Setting("enabled", ValueKind.BOOLEAN, DEFAULT_DEDUP_ENABLED,
                    "Translate identical requests once per run."),
        )),
        Section(("heartbeat",), "Progress snapshot for external monitors (progress_state.json).", (
            Setting("interval", ValueKind.INTEGER, DEFAULT_HEARTBEAT_INTERVAL,
                    "Rewrite progress_state.json every N string leaves "
                    "(>= 0; 0 = only on file/stage/run boundaries)."),
        )),
        Section(("limits",), "Safety limits.", (
            Setting("max_recursion_depth", ValueKind.INTEGER, DEFAULT_MAX_RECURSION_DEPTH,
                    "Deeper JSON subtrees are kept untranslated (> 0)."),
            Setting("large_file_threshold_mb", ValueKind.INTEGER, DEFAULT_LARGE_FILE_THRESHOLD_MB,
                    "Warn when writing a file larger than this, MB (> 0)."),
            Setting("file_operation_timeout", ValueKind.INTEGER, DEFAULT_FILE_OPERATION_TIMEOUT,
                    "Timeout of one file read or write, seconds (> 0)."),
            Setting("max_error_display", ValueKind.INTEGER, DEFAULT_MAX_ERROR_DISPLAY,
                    "Items listed in one diagnostic message (> 0)."),
            Setting("dedup_log_interval", ValueKind.INTEGER, DEFAULT_DEDUP_LOG_INTERVAL,
                    "Log progress every N reused duplicates (> 0)."),
        )),
        Section(("resilience",), "Circuit breaker for runs of consecutive model-call failures.", (
            Setting("max_consecutive_failures", ValueKind.INTEGER, DEFAULT_MAX_CONSECUTIVE_FAILURES,
                    "Abort a stage after this many consecutive failed calls (>= 0, 0 disables the breaker)."),
            Setting("max_failure_ratio", ValueKind.FLOAT, DEFAULT_MAX_FAILURE_RATIO,
                    "Exit with a non-zero code if a stage's failure share exceeds this, 0.0..1.0."),
        )),
        Section(("validation",),
                "Sanity checks of a model reply before it is cached (both stages). "
                "allowed_scripts lives in llm.toml [validation] (travels with the profile).", (
            Setting("enabled", ValueKind.BOOLEAN, DEFAULT_VALIDATION_ENABLED,
                    "Reject an obviously broken reply instead of caching it."),
            Setting("max_length_ratio", ValueKind.FLOAT, DEFAULT_VALIDATION_MAX_LENGTH_RATIO,
                    "Reject a reply longer than original_length * ratio + slack (>= 0)."),
            Setting("max_length_slack", ValueKind.INTEGER, DEFAULT_VALIDATION_MAX_LENGTH_SLACK,
                    "Flat allowance added to the length check, characters (>= 0)."),
            Setting("latin_text_policy", ValueKind.STRING, DEFAULT_VALIDATION_LATIN_POLICY,
                    f"One of {', '.join(VALID_LATIN_POLICIES)}, for Latin text absent from "
                    f"the source and the glossary when latin is not in allowed_scripts."),
        )),
    ),
)
