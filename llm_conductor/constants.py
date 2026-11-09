# -*- coding: utf-8 -*-
import errno
import re

# ============================================================================
# CONFIGURATION FILES
# ============================================================================
# conductor.toml lives next to the executable; LLMC_CONFIG overrides its path
# (meant for tests, not documented as a user-facing switch). llm.toml defaults
# to the directory of conductor.toml and is replaced by --llm-config.
CONDUCTOR_CONFIG_FILENAME = 'conductor.toml'
LLM_CONFIG_FILENAME = 'llm.toml'
CONFIG_PATH_ENV_VAR = 'LLMC_CONFIG'

# ============================================================================
# conductor.toml DEFAULTS
# ============================================================================
# [logging]
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_DIR = './logs'
DEFAULT_LOG_RETENTION_DAYS = 7
DEFAULT_LOG_EXTENDED_DIAGNOSTIC = False
# Levels accepted in [logging].level; VERBOSE is an alias that turns on the
# debug-only diagnostics while the console handler still prints INFO.
VALID_LOG_LEVELS = ('DEBUG', 'VERBOSE', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
DEBUG_LOG_LEVELS = ('DEBUG', 'VERBOSE')
# Levels at which the per-stage processing statistics are printed.
STATISTICS_LOG_LEVELS = ('INFO', 'DEBUG', 'VERBOSE')

# TRACE is finer than DEBUG and never selectable through [logging].level: it
# is switched on only by [logging].extended_diagnostic and always goes to
# its own file (full prompts, cache-key hashes), never to the console.
TRACE_LOG_LEVEL = 5
TRACE_LOG_LEVEL_NAME = 'TRACE'
TRACE_LOG_FILE_PREFIX = 'trace'

# [cache]
DEFAULT_CACHE_ENABLED = True
DEFAULT_CACHE_DIR = './cache'
DEFAULT_CACHE_MAX_SIZE = 10000
DEFAULT_CACHE_EXPIRY_DAYS = 30
DEFAULT_CACHE_AUTO_SAVE_INTERVAL = 100
DEFAULT_GLOSSARY_CACHE_MAX_SIZE = 1000

# [dedup]
DEFAULT_DEDUP_ENABLED = True

# [heartbeat]
DEFAULT_HEARTBEAT_INTERVAL = 100
PROGRESS_STATE_FILENAME = 'progress_state.json'
HEARTBEAT_SCHEMA_VERSION = 1
HEARTBEAT_STATUS_RUNNING = 'running'
HEARTBEAT_STATUS_COMPLETED = 'completed'
HEARTBEAT_STATUS_INTERRUPTED = 'interrupted'

# [limits]
DEFAULT_MAX_RECURSION_DEPTH = 30
DEFAULT_LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_FILE_OPERATION_TIMEOUT = 30
DEFAULT_MAX_ERROR_DISPLAY = 10
DEFAULT_DEDUP_LOG_INTERVAL = 100

# errno values that FileOperations treats as transient (retry with backoff).
# Permanent path errors (FileNotFoundError, NotADirectoryError, …) are never
# retried even though they subclass OSError; PermissionError / TimeoutError
# are retried by type, not by this set.
RETRYABLE_IO_ERRNOS = frozenset({errno.EAGAIN, errno.EBUSY, errno.EIO})

# [resilience]
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_MAX_FAILURE_RATIO = 0.5
DEFAULT_FAIL_ON_ANY_ERROR = False

# [validation]
DEFAULT_VALIDATION_ENABLED = True
DEFAULT_VALIDATION_MAX_LENGTH_RATIO = 3.0
DEFAULT_VALIDATION_MAX_LENGTH_SLACK = 20
DEFAULT_VALIDATION_LATIN_POLICY = 'warn'
DEFAULT_VALIDATION_SAVE_REJECTED_REPLIES = False
VALID_LATIN_POLICIES = ('warn', 'reject')
# Writing systems always allowed in replies ([validation].allowed_scripts of
# llm.toml). Empty = none; the engine does not assume a target alphabet.
# The init template writes EXAMPLE_EN_RU_ALLOWED_SCRIPTS to match the EN→RU
# example prompts (template_default), while an absent key still resolves empty.
DEFAULT_VALIDATION_ALLOWED_SCRIPTS = ()
EXAMPLE_EN_RU_ALLOWED_SCRIPTS = ('cyrillic',)
VALID_ALLOWED_SCRIPTS = (
    'latin', 'cyrillic', 'greek', 'arabic', 'hebrew', 'cjk', 'thai',
)
# Under [logging].dir when [validation].save_rejected_replies is on.
REJECTED_REPLIES_SUBDIR = 'rejected'

# ============================================================================
# llm.toml DEFAULTS - a fully local profile: Ollama on both stages, no key
# ============================================================================
DEFAULT_PROFILE_NAME = 'Local Ollama (qwen2.5:14b)'
DEFAULT_PROFILE_DESCRIPTION = 'Translation and verification on a local Ollama server'

# [preflight]
DEFAULT_PREFLIGHT_CHECK_AVAILABILITY = True
DEFAULT_PREFLIGHT_TIMEOUT = 30

# [translation.api] / [verification.api]
VALID_API_TYPES = ['openai', 'ollama']
DEFAULT_TRANSLATION_API_URL = 'http://localhost:11434/api/chat'
DEFAULT_TRANSLATION_API_TYPE = 'ollama'
DEFAULT_TRANSLATION_MODEL = 'qwen2.5:14b'
DEFAULT_TRANSLATION_API_TIMEOUT = 90
DEFAULT_TRANSLATION_MAX_RETRIES = 3
DEFAULT_TRANSLATION_RETRY_BACKOFF_FACTOR = 2.0

DEFAULT_VERIFICATION_API_URL = 'http://localhost:11434/api/chat'
DEFAULT_VERIFICATION_API_TYPE = 'ollama'
DEFAULT_VERIFICATION_MODEL = 'qwen2.5:14b'
DEFAULT_VERIFICATION_API_TIMEOUT = 60
DEFAULT_VERIFICATION_MAX_RETRIES = 3
DEFAULT_VERIFICATION_RETRY_BACKOFF_FACTOR = 2.0

# Ollama how long to keep the model in VRAM after a reply. Empty string =
# omit the field and let the server use its own default (usually 5m).
# OpenAI-compatible endpoints ignore this key. Values follow Ollama's
# duration syntax: "5m", "1h", "0" (unload immediately), "-1" (forever).
DEFAULT_API_KEEP_ALIVE = ''
DEFAULT_MAX_TOKENS_PARAM = 'max_tokens'
VALID_MAX_TOKENS_PARAMS = ('max_tokens', 'max_completion_tokens')
# OpenAI preflight must request more than 1 token: reasoning models (GPT-5 /
# o-series) return HTTP 400 when max_completion_tokens is too small to finish
# even a hidden reasoning step. Classic chat models accept the higher cap.
DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS = 16

# [translation.prompts] / [verification.prompts] - written by the rules of
# doc/prompts.md: the model sees [Tn] markers instead of the original tags,
# the leaf value is glued to the prompt without a separator, so every user
# prompt ends with a label and a newline.
DEFAULT_TRANSLATION_SYSTEM_PROMPT = (
    "You are an automated localization tool. Translate the English text to Russian.\n"
    "CRITICAL RULES: Output ONLY the final translated text. NEVER include explanations,\n"
    "comments, alternatives, or apologies. NEVER mix English words into the translation.\n"
    "Even a single word or a short label is text to translate: output only its translation.\n"
    "Preserve every formatting marker exactly as it appears and in its correct position.\n"
)
DEFAULT_TRANSLATION_STRICT_PROMPT = (
    "[CONTEXT: USER INTERFACE TEXT - PRECISE, CONCISE TRANSLATION]{LINGUISTIC_CONTEXT}{GLOSSARY}\n"
    "Text to translate:\n"
)
DEFAULT_TRANSLATION_STYLE_PROMPT = (
    "[CONTEXT: DIALOGUE AND NARRATIVE TEXT - NATURAL, LIVELY STYLE]{LINGUISTIC_CONTEXT}{GLOSSARY}\n"
    "Text to translate:\n"
)
DEFAULT_VERIFICATION_SYSTEM_PROMPT = (
    "You are an automated proofreader. You receive text that is ALREADY translated into Russian.\n"
    "Fix grammar, agreement (gender, number, case) and awkward phrasing; keep the meaning and\n"
    "the length close to the input. CRITICAL RULES: Output ONLY the final Russian text.\n"
    "NEVER translate it into English or any other language. NEVER include explanations or lists\n"
    "of changes. Preserve every formatting marker exactly as it appears and in its correct\n"
    "position. If the text is already correct, output it unchanged. If an [ORIGINAL] block is\n"
    "present, it is the English source given only for reference - never copy it or its markers\n"
    "into your output.\n"
)
DEFAULT_VERIFICATION_EDIT_PROMPT = (
    "[TASK: PROOFREAD TRANSLATED TEXT]{LINGUISTIC_CONTEXT}{GLOSSARY}{ORIGINAL_TEXT}\n"
    "Text to proofread:\n"
)

# Sampling. Real newlines of the source are masked into [Tn] markers, so a
# blank line in a reply is always model chatter. Ollama replaces the model's
# own stop tokens with this list - the Qwen tokens of the default model are
# therefore repeated here.
DEFAULT_STOP_SEQUENCES = ("\n\n", "<|im_end|>", "<|im_start|>")

# [translation.sampling] - shared by both translation modes
DEFAULT_TRANSLATION_FREQUENCY_PENALTY = 0.0
DEFAULT_TRANSLATION_PRESENCE_PENALTY = 0.0

# [translation.sampling.strict]
DEFAULT_TRANSLATION_STRICT_TEMPERATURE = 0.3
DEFAULT_TRANSLATION_STRICT_TOP_P = 0.15
DEFAULT_TRANSLATION_STRICT_TOP_K = 5
DEFAULT_TRANSLATION_STRICT_REPEAT_PENALTY = 1.1
DEFAULT_TRANSLATION_STRICT_NUM_PREDICT = 150
DEFAULT_TRANSLATION_STRICT_NUM_CTX = 2048

# [translation.sampling.style]
DEFAULT_TRANSLATION_STYLE_TEMPERATURE = 0.8
DEFAULT_TRANSLATION_STYLE_TOP_P = 0.95
DEFAULT_TRANSLATION_STYLE_TOP_K = 60
DEFAULT_TRANSLATION_STYLE_REPEAT_PENALTY = 1.25
DEFAULT_TRANSLATION_STYLE_NUM_PREDICT = 200
DEFAULT_TRANSLATION_STYLE_NUM_CTX = 3072

# [verification.sampling]
DEFAULT_VERIFICATION_TEMPERATURE = 0.2
DEFAULT_VERIFICATION_TOP_P = 0.9
DEFAULT_VERIFICATION_TOP_K = 40
DEFAULT_VERIFICATION_REPEAT_PENALTY = 1.1
DEFAULT_VERIFICATION_FREQUENCY_PENALTY = 0.0
DEFAULT_VERIFICATION_PRESENCE_PENALTY = 0.0
DEFAULT_VERIFICATION_NUM_PREDICT = 200
DEFAULT_VERIFICATION_NUM_CTX = 2048

# Valid ranges of sampling parameters, checked by TranslationConfig.validate().
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0

# Unknown keys in a TOML file are reported with the closest known name when
# difflib similarity reaches this ratio.
CONFIG_KEY_SUGGESTION_CUTOFF = 0.6

# ============================================================================
# TRANSPORT
# ============================================================================
# Regex of the formatting tags masked before a model call.
TAG_MASK_PATTERN = re.compile(r'(?:\{\d+\})|(?:\n)|(?:<[^>]+>)')

# Default letter prefix for mask_tags() markers ("[T0]", "[T1]", ...). Tried
# first; a different prefix is chosen only if the source text already
# contains a literal "[T<digits>]"-shaped substring.
DEFAULT_MARKER_PREFIX = "T"

# Fallback transport defaults for make_request_with_retry() when runtime_config
# omits a key - actual values normally come from [translation.api] /
# [verification.api]; these are a last-resort safety net.
DEFAULT_API_REQUEST_TIMEOUT = 90
DEFAULT_API_MAX_RETRIES = 3
DEFAULT_API_BACKOFF_FACTOR = 2

# HTTP status codes used by the retry policy and preflight diagnostics.
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_PAYMENT_REQUIRED = 402
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_PAYLOAD_TOO_LARGE = 413
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_TOO_MANY_REQUESTS = 429

# 4xx codes that make_request_with_retry() treats as fatal on the first
# attempt - retrying reproduces the same client error (402 is an empty
# balance, 413 an oversized body; neither fixes itself between attempts).
NON_RETRYABLE_HTTP_STATUS_CODES = frozenset({
    HTTP_BAD_REQUEST, HTTP_UNAUTHORIZED, HTTP_PAYMENT_REQUIRED, HTTP_FORBIDDEN,
    HTTP_NOT_FOUND, HTTP_PAYLOAD_TOO_LARGE, HTTP_UNPROCESSABLE_ENTITY,
})

# Upper bound on a server-requested Retry-After wait, in seconds; a larger
# value (or a misbehaving proxy) must not stall the pipeline indefinitely.
MAX_RETRY_AFTER_SECONDS = 60

# The OpenAI Chat Completions API accepts at most this many stop sequences.
MAX_OPENAI_STOP_SEQUENCES = 4

# Max characters of a non-JSON error body kept by describe_http_error().
HTTP_ERROR_BODY_PREVIEW_LENGTH = 200

# MetricsCollector failure classification - each kind is a different
# problem, fixed in a different place (transport, parsing, provider-signaled
# truncation/refusal, or the local reply validator).
FAILURE_KIND_NETWORK = 'network'
FAILURE_KIND_PARSE = 'parse'
FAILURE_KIND_TRUNCATED = 'truncated'
FAILURE_KIND_REFUSAL = 'refusal'
FAILURE_KIND_REJECTED = 'rejected'

# Provider-reported finish/done reason meaning "the reply was cut off by a
# token limit". Shared by Ollama's done_reason and OpenAI's finish_reason.
FINISH_REASON_LENGTH = 'length'

# Reasons validate_model_reply() (llm_conductor/l10n/validation.py) rejects a reply;
# short machine slugs, used in log messages and metrics.
VALIDATION_REASON_NEWLINE_IN_REPLY = 'newline_in_reply'
VALIDATION_REASON_TAG_MARKER_MISMATCH = 'tag_marker_mismatch'
VALIDATION_REASON_FABRICATED_TAG = 'fabricated_tag'
VALIDATION_REASON_RESPONSE_TOO_LONG = 'response_too_long'
VALIDATION_REASON_FOREIGN_SCRIPT = 'foreign_script'
VALIDATION_REASON_UNEXPECTED_LATIN = 'unexpected_latin'

# Max characters of a rejected model reply kept in the WARNING log line, so
# the actual bad text (not just its rejection reason) survives for diagnosis.
# Full body persistence is opt-in via [validation].save_rejected_replies.
REJECTED_REPLY_LOG_LENGTH = 200

# ============================================================================
# EXIT CODES
# ============================================================================
# A scheduler sees only the exit code, so every outcome of a run has its own.
EXIT_SUCCESS = 0
EXIT_INTERNAL_ERROR = 1         # unexpected failure, traceback in the log
EXIT_USAGE_ERROR = 2            # bad command line or unusable run inputs (argparse also exits with 2)
EXIT_CONFIG_ERROR = 3           # missing or invalid conductor.toml / llm.toml
EXIT_API_UNAVAILABLE = 4        # preflight check of an endpoint failed
EXIT_FAILURE_RATIO_EXCEEDED = 5 # share of failed requests or failed files above the threshold
EXIT_CONSECUTIVE_FAILURES = 6   # run aborted by the consecutive-failure circuit breaker
EXIT_TARGET_LOCKED = 7          # target, cache or shared result root locked by another run
EXIT_INTERRUPTED = 130          # stopped by a signal (128 + SIGINT); the results are incomplete

# When one run hits several problems, the first code of this tuple that
# occurred wins. EXIT_USAGE_ERROR never combines with others in practice -
# it is raised before any processing starts.
EXIT_CODE_PRIORITY = (
    EXIT_INTERNAL_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_USAGE_ERROR,
    EXIT_API_UNAVAILABLE,
    EXIT_TARGET_LOCKED,
    EXIT_CONSECUTIVE_FAILURES,
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
)

# ============================================================================
# PIPELINE
# ============================================================================
# Translation modes
VALID_MODES = ['strict', 'style', 'verification']
MODE_STRICT = 'strict'      # precise, technical translation
MODE_STYLE = 'style'        # translation with natural style
MODE_VERIFICATION = 'verification'

# Operation types - each has its own cache
OPERATION_TRANSLATION = 'translation'
OPERATION_VERIFICATION = 'verification'

# Pipeline stages, used as result/ subdirectory names
STAGE_TRANSLATED = 'translated'
STAGE_REDACTED = 'redacted'
STAGE_FINAL = 'final'

# ============================================================================
# RUN COMMAND
# ============================================================================
# Values of `run --stage`.
RUN_STAGE_ALL = 'all'
RUN_STAGE_TRANSLATE = 'translate'
RUN_STAGE_VERIFY = 'verify'
VALID_RUN_STAGES = (RUN_STAGE_ALL, RUN_STAGE_TRANSLATE, RUN_STAGE_VERIFY)

# A target with this extension is a TOML run manifest ([[targets]]).
MANIFEST_EXTENSION = '.toml'
# Former line-oriented batch format; rejected with a migration hint.
LEGACY_LIST_EXTENSION = '.list'
UTF8_BOM = '\ufeff'
# A drive prefix ("D:\x.json", "D:x.json") is rejected on every OS, so one
# manifest behaves the same on Windows and POSIX.
MANIFEST_DRIVE_PATTERN = re.compile(r'^[A-Za-z]:')

# Allowed top-level keys of a run manifest (unknown → exit 2).
RUN_MANIFEST_RUN_KEYS = frozenset({
    'verb', 'result2target', 'cache2target', 'stage', 'strict',
    'llm_config', 'glossary', 'targets',
})
RUN_MANIFEST_TARGET_KEYS = frozenset({
    'original', 'draft', 'stage', 'strict', 'llm_config', 'glossary',
})

# Results go to <root>/result/{translated,redacted,final}/<relative path>.
RESULT_DIR_NAME = 'result'
# Per-leaf stage status sidecars live under <result_root>/leaves/<relative>.json
LEAVES_DIR_NAME = 'leaves'
RUN_REPORT_FILENAME = 'run_report.json'
USAGE_REPORT_FILENAME = 'usage.json'
LEAF_REPORT_SCHEMA_VERSION = 1
USAGE_REPORT_SCHEMA_VERSION = 1

# Per-leaf outcome statuses (machine report / verify-skip contract).
LEAF_STATUS_OK = 'ok'
LEAF_STATUS_SKIPPED = 'skipped'
LEAF_STATUS_CACHE_HIT = 'cache_hit'
LEAF_STATUS_DEDUP = 'dedup'
LEAF_STATUS_DEGRADED = 'degraded'
LEAF_STATUS_VERIFY_SKIPPED = 'verify_skipped'

# Why verification was skipped for a leaf (never inferred from string equality).
LEAF_SKIP_REASON_TRANSLATION_DEGRADED = 'translation_degraded'

# UsageLedger attempt outcomes and the separate preflight bucket.
USAGE_OUTCOME_SUCCESS = 'success'
USAGE_OUTCOME_FAILED = 'failed'
# HTTP succeeded but content validation rejected the reply (tokens still count).
USAGE_OUTCOME_REJECTED = 'rejected'
USAGE_STAGE_PREFLIGHT = 'preflight'
USAGE_NON_API_CACHE_HIT = 'cache_hit'
USAGE_NON_API_DEDUP = 'dedup'
# Cache directory created next to the target by --cache2target.
TARGET_CACHE_DIR_NAME = 'cache'
# Glossary looked up next to the target when --glossary is not given; the
# origin labels name the level of that lookup chain in the log.
GLOSSARY_FILENAME = 'glossary.json'
GLOSSARY_ORIGIN_OPTION = '--glossary'
GLOSSARY_ORIGIN_TARGET = 'next to the target'

# Log file name: conductor_<target name>_<start time>.log
LOG_FILE_PREFIX = 'conductor'
LOG_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# OS-level locks. The cache directory and the shared result/<target name>/
# root are locked through this file inside them; a target is locked on the
# file itself. On Windows the locked byte lies far beyond the end of any
# realistic input file, so locking it never blocks reading the content
# (Windows byte-range locks are mandatory). The offset stays below 2 GiB to
# remain valid for 32-bit file offsets.
CACHE_LOCK_FILENAME = '.llmc.lock'
# Owner marker of a shared result/<target name>/ directory: one line with
# the absolute, case-folded path of the target that claimed it.
RESULT_OWNER_FILENAME = '.llmc.target'
LOCK_REGION_OFFSET = 0x7FFFFFF0
LOCK_REGION_LENGTH = 1

# Prompt injection markers (user places these in prompt templates)
LINGUISTIC_CONTEXT_MARKER = "{LINGUISTIC_CONTEXT}"
GLOSSARY_MARKER = "{GLOSSARY}"
# Verification-only marker: the masked English source, inserted for
# alignment reference. Empty for translation - never present.
ORIGINAL_TEXT_MARKER = "{ORIGINAL_TEXT}"
TRANSLATION_TEXT_MARKER = "Text to translate:\n"
VERIFICATION_TEXT_MARKER = "Text to proofread:\n"

# Prompt block labels the engine inserts at the markers above.
# Not user-configurable: changing a label requires matching system-prompt
# wording (e.g. "If an [ORIGINAL] block is present...").
LINGUISTIC_CONTEXT_BLOCK_TEMPLATE = "\n[LINGUISTIC CONTEXT]: {instructions}.\n"
GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE = "\n[GLOSSARY DEFINITIONS]: {entries}.\n"
ORIGINAL_TEXT_BLOCK_TEMPLATE = "\n[ORIGINAL]: {masked_original}\n"
# Appended after ORIGINAL_TEXT_BLOCK_TEMPLATE only when the masked original
# actually contains markers (non-empty tag_map). Without markers the note
# is noise and can provoke fabricated_tag / tag_marker_mismatch.
ORIGINAL_TEXT_BLOCK_NOTE = (
    "(reference only - marker numbers may not align with the text below)\n"
)

# Cache key schema version. Written into every new cache entry; load()
# discards entries without it or with another value, so a key format change
# never needs a manual cleanup of cache_dir.
# v2: generation-affecting sampling parameters join the key identity.
CACHE_KEY_VERSION = 2

# Generation-affecting sampling fields in the cache key / fingerprints:
# (canonical name in the identity JSON, runtime_config key). keep_alive and
# max_tokens_param are intentionally excluded — they do not change the reply.
CACHE_SAMPLING_IDENTITY_FIELDS = (
    ('temperature', 'temp'),
    ('top_p', 'top_p'),
    ('top_k', 'top_k'),
    ('repeat_penalty', 'repeat_penalty'),
    ('frequency_penalty', 'frequency_penalty'),
    ('presence_penalty', 'presence_penalty'),
    ('num_predict', 'num_predict'),
    ('num_ctx', 'num_ctx'),
    ('stop', 'stop'),
)

# Length (in hex characters) of the request component fingerprints stored in
# a cache entry's metadata. They only explain cache misses
# (CacheManager.diagnose_miss) and are not part of the cache key, so the
# truncated MD5 does not affect caching correctness.
CACHE_FINGERPRINT_LENGTH = 12
