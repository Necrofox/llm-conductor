# -*- coding: utf-8 -*-
"""
Sanity checks of a model reply, run once per leaf before it is cached.

validate_model_reply() is a pure function (no state, no I/O) - the same
style as mask_tags()/build_glossary_string() rather than a class, since
there is nothing to encapsulate between calls.
"""
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Set, Tuple

from llm_conductor.constants import (
    TAG_MASK_PATTERN,
    VALIDATION_REASON_FABRICATED_TAG,
    VALIDATION_REASON_FOREIGN_SCRIPT,
    VALIDATION_REASON_NEWLINE_IN_REPLY,
    VALIDATION_REASON_RESPONSE_TOO_LONG,
    VALIDATION_REASON_TAG_MARKER_MISMATCH,
    VALIDATION_REASON_UNEXPECTED_LATIN,
)

if TYPE_CHECKING:
    from llm_conductor.interfaces import ProcessorConfigProtocol

# Unicode general-category prefixes always allowed regardless of script:
# punctuation, symbol (incl. emoji), number, separator (whitespace), control.
_COMMON_CATEGORY_PREFIXES = ('P', 'S', 'N', 'Z', 'C')

# Map [validation].allowed_scripts names to unicodedata.name() prefixes.
# Matching is prefix + space/hyphen (or exact name), so LATIN does not
# match a hypothetical LATINIZED… token. No third-party regex / \p{}.
_SCRIPT_NAME_PREFIXES: Dict[str, Tuple[str, ...]] = {
    'latin': ('LATIN',),
    'cyrillic': ('CYRILLIC',),
    'greek': ('GREEK',),
    'arabic': ('ARABIC',),
    'hebrew': ('HEBREW',),
    'thai': ('THAI',),
    'cjk': ('CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL'),
}


class ReplyValidationResult(NamedTuple):
    """
    Outcome of validate_model_reply().

    reason is set only when accepted is False. warning is set only on a
    non-rejecting finding (currently: Latin text under the 'warn' policy)
    for the caller to log without discarding the reply.
    """

    accepted: bool
    reason: Optional[str]
    warning: Optional[str]


def _name_matches_prefix(name: str, prefix: str) -> bool:
    """True if a Unicode character name belongs to the given script prefix."""
    return name == prefix or name.startswith((prefix + ' ', prefix + '-'))


def _script_of(char: str) -> Optional[str]:
    """
    Return the allowed_scripts name for char, or None if unclassified.

    Args:
        char: A single character.

    Returns:
        One of the keys of _SCRIPT_NAME_PREFIXES, or None.
    """
    name = unicodedata.name(char, '')
    if not name:
        return None
    for script, prefixes in _SCRIPT_NAME_PREFIXES.items():
        if any(_name_matches_prefix(name, prefix) for prefix in prefixes):
            return script
    return None


def _classify_char(
    char: str,
    original_value: str,
    glossary_text: str,
    allowed_scripts: Set[str],
) -> str:
    """
    Classify one character of a reply for the foreign-script/Latin check.

    Args:
        char: Character to classify.
        original_value: Source of the leaf - a character already present
            there (a proper noun, a brand name) is never flagged.
        glossary_text: Concatenation of every glossary translation - a
            character fixed there (e.g. {"Arasaka": "Arasaka"}) is allowed.
        allowed_scripts: Writing systems listed in llm.toml
            [validation].allowed_scripts.

    Returns:
        'common' (always allowed), 'latin' (allowed, but reported
        separately for the soft policy check when latin is not in
        allowed_scripts), or 'foreign' (never allowed).
    """
    if char.isspace() or char in original_value or char in glossary_text:
        return 'common'
    if unicodedata.category(char)[0] in _COMMON_CATEGORY_PREFIXES:
        return 'common'
    script = _script_of(char)
    if script is not None and script in allowed_scripts:
        return 'common'
    if script == 'latin':
        return 'latin'
    return 'foreign'


def validate_model_reply(
    original_value: str,
    masked_reply: str,
    unmasked_reply: str,
    tag_map: Dict[str, str],
    missing_markers: List[str],
    orphaned_markers: List[str],
    glossary: dict,
    config: 'ProcessorConfigProtocol',
) -> ReplyValidationResult:
    """
    Run the sanity checks of [validation] on one accepted model reply.

    Order matters: the first violation found is the rejection reason. The
    caller (RecursiveProcessor._process_leaf) only calls this once the
    transport/parsing layer (call_model_api) already reports success - it
    decides whether a bad reply is cached or discarded.

    Args:
        original_value: Unmasked source of the leaf.
        masked_reply: Model's raw reply, still carrying [Tn] markers - a
            literal newline in it is always chatter, since the source's
            real newlines were masked before the call.
        unmasked_reply: masked_reply after unmask_tags().
        tag_map: Marker -> original tag, as returned by mask_tags().
        missing_markers: Markers unmask_tags() could not find in the reply.
        orphaned_markers: [Tn]-shaped text left after unmasking.
        glossary: Glossary {term: translation}; translations are allowed
            Latin text even when absent from original_value.
        config: Supplies [validation] thresholds, policy, and
            allowed_scripts.

    Returns:
        A ReplyValidationResult.
    """
    if '\n' in masked_reply:
        return ReplyValidationResult(False, VALIDATION_REASON_NEWLINE_IN_REPLY, None)

    if missing_markers or orphaned_markers:
        return ReplyValidationResult(False, VALIDATION_REASON_TAG_MARKER_MISMATCH, None)

    if Counter(TAG_MASK_PATTERN.findall(unmasked_reply)) != Counter(tag_map.values()):
        return ReplyValidationResult(False, VALIDATION_REASON_FABRICATED_TAG, None)

    max_allowed_length = config.validation.max_length_ratio * len(original_value) + config.validation.max_length_slack
    if len(unmasked_reply) > max_allowed_length:
        return ReplyValidationResult(False, VALIDATION_REASON_RESPONSE_TOO_LONG, None)

    glossary_text = ''.join(str(translation) for translation in glossary.values()) if glossary else ''
    allowed_scripts = set(config.validation.allowed_scripts)
    foreign_chars = set()
    latin_chars = set()
    for char in unmasked_reply:
        classification = _classify_char(char, original_value, glossary_text, allowed_scripts)
        if classification == 'foreign':
            foreign_chars.add(char)
        elif classification == 'latin':
            latin_chars.add(char)

    if foreign_chars:
        return ReplyValidationResult(False, VALIDATION_REASON_FOREIGN_SCRIPT, None)

    if latin_chars:
        if config.validation.latin_text_policy == 'reject':
            return ReplyValidationResult(False, VALIDATION_REASON_UNEXPECTED_LATIN, None)
        warning = (
            f"reply contains Latin text absent from the source and the glossary: "
            f"{''.join(sorted(latin_chars))!r}"
        )
        return ReplyValidationResult(True, None, warning)

    return ReplyValidationResult(True, None, None)
