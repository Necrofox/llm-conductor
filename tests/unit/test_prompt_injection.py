# -*- coding: utf-8 -*-
"""
Unit tests for hint and glossary injection into the prompt
(RecursiveProcessor._build_enhanced_prompt and helpers, doc/pipeline.md).

The methods are pure — they only touch strings; the processor is built on
a stub config with no cache.
"""
import pytest

from llm_conductor.constants import (
    GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE,
    LINGUISTIC_CONTEXT_BLOCK_TEMPLATE,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    ORIGINAL_TEXT_BLOCK_TEMPLATE,
)
from llm_conductor.l10n.context import HintType, LinguisticContext, LinguisticHint
from llm_conductor.l10n.processor import RecursiveProcessor
from tests.support import StubConfig

GLOSSARY = GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE.format(entries="Guild = Гильдия")
HINT = LinguisticHint(HintType.GENDER, "Use MASCULINE forms", priority=10)
CONTEXT_BLOCK = LINGUISTIC_CONTEXT_BLOCK_TEMPLATE.format(instructions="Use MASCULINE forms")
# Fixture for placement tests only - note text is not under test here.
ORIGINAL_BLOCK = (
    ORIGINAL_TEXT_BLOCK_TEMPLATE.format(masked_original="Hello")
    + "(reference only)\n"
)


@pytest.fixture
def processor():
    config = StubConfig()
    return RecursiveProcessor(config, LinguisticContext(config))


def build(processor, base_prompt, hints=(HINT,), glossary=GLOSSARY, operation_type=OPERATION_TRANSLATION,
          original_text_block=""):
    return processor._build_enhanced_prompt(
        base_prompt=base_prompt,
        hints=list(hints),
        glossary_string=glossary,
        operation_type=operation_type,
        original_text_block=original_text_block,
    )


# ============================================================================
# Four marker scenarios
# ============================================================================

def test_both_markers_replaced_independently(processor):
    result = build(processor, "Rules.{LINGUISTIC_CONTEXT}Terms:{GLOSSARY}Text:\n")

    assert result == f"Rules.{CONTEXT_BLOCK}Terms:{GLOSSARY}Text:\n"


def test_only_context_marker_puts_glossary_right_after_context(processor):
    result = build(processor, "Rules.{LINGUISTIC_CONTEXT}Text:\n")

    assert result == f"Rules.{CONTEXT_BLOCK}{GLOSSARY}Text:\n"


def test_only_glossary_marker_puts_context_before_glossary(processor):
    result = build(processor, "Rules.{GLOSSARY}Text:\n")

    assert result == f"Rules.{CONTEXT_BLOCK}{GLOSSARY}Text:\n"


def test_no_markers_inserts_before_sentence_with_translate_keyword(processor):
    result = build(processor, "You are careful. Translate the text: ")

    assert result == f"You are careful. {CONTEXT_BLOCK}{GLOSSARY}Translate the text: "


def test_no_markers_verification_searches_verify_keyword(processor):
    result = build(processor, "Be strict.\nVerify this: ", operation_type=OPERATION_VERIFICATION)

    assert result == f"Be strict.\n{CONTEXT_BLOCK}{GLOSSARY}Verify this: "


def test_no_markers_and_no_keyword_appends_with_explicit_text_marker(processor):
    result = build(processor, "Do your best.")

    assert result == f"Do your best.\n\n{CONTEXT_BLOCK}{GLOSSARY}Text to translate:\n"


def test_no_markers_no_keyword_and_nothing_to_inject_leaves_prompt_untouched(processor):
    assert build(processor, "Do your best.", hints=(), glossary="") == "Do your best."


# ============================================================================
# {ORIGINAL_TEXT} (verification reference block)
# ============================================================================

def test_original_text_marker_is_replaced(processor):
    result = build(processor, "Rules.{ORIGINAL_TEXT}Verify the text: ", hints=(), glossary="",
                    operation_type=OPERATION_VERIFICATION, original_text_block=ORIGINAL_BLOCK)

    assert result == f"Rules.{ORIGINAL_BLOCK}Verify the text: "


def test_original_text_marker_with_empty_block_is_stripped(processor):
    result = build(processor, "Rules.{ORIGINAL_TEXT}Verify the text: ", hints=(), glossary="",
                    operation_type=OPERATION_VERIFICATION, original_text_block="")

    assert result == "Rules.Verify the text: "


def test_original_text_block_is_never_injected_for_translation(processor):
    # original_text_block is always "" for translation in production code;
    # even if a caller somehow passed one, there is no marker to replace it
    # at, and no "verify"/"translate" keyword mismatch should leak it in.
    result = build(processor, "Translate the text: ", hints=(), glossary="",
                    operation_type=OPERATION_TRANSLATION, original_text_block=ORIGINAL_BLOCK)

    assert result == f"{ORIGINAL_BLOCK}Translate the text: "


def test_original_text_without_marker_is_inserted_before_verify_keyword(processor):
    """
    Regression guard: the block must land *before* the trailing label, not
    after it - EchoModel (tests/system/harness.py) and real providers alike
    rely on "...the text: " being immediately followed by the leaf value,
    with nothing in between.
    """
    result = build(processor, "Verify the text: ", hints=(), glossary="",
                    operation_type=OPERATION_VERIFICATION, original_text_block=ORIGINAL_BLOCK)

    assert result == f"{ORIGINAL_BLOCK}Verify the text: "
    assert result.endswith("Verify the text: ")


def test_original_text_combines_with_context_and_glossary_in_no_marker_prompt(processor):
    result = build(processor, "Verify the text: ", operation_type=OPERATION_VERIFICATION,
                    original_text_block=ORIGINAL_BLOCK)

    assert result == f"{ORIGINAL_BLOCK}{CONTEXT_BLOCK}{GLOSSARY}Verify the text: "


def test_original_text_without_marker_or_keyword_appends_at_end(processor):
    result = build(processor, "Do your best.", hints=(), glossary="",
                    operation_type=OPERATION_VERIFICATION, original_text_block=ORIGINAL_BLOCK)

    assert result == f"Do your best.{ORIGINAL_BLOCK}"


# ============================================================================
# Empty blocks and repeated markers
# ============================================================================

def test_empty_blocks_remove_their_markers(processor):
    result = build(processor, "A{LINGUISTIC_CONTEXT}B{GLOSSARY}C", hints=(), glossary="")

    assert result == "ABC"


def test_only_last_occurrence_of_marker_is_replaced_others_removed(processor):
    result = build(processor, "{GLOSSARY}x{GLOSSARY}y", hints=(), glossary="[G]")

    assert result == "x[G]y"


def test_multiple_hints_joined_in_given_order(processor):
    second = LinguisticHint(HintType.PLURAL, "Use PLURAL", priority=5)

    result = build(processor, "{LINGUISTIC_CONTEXT}", hints=(HINT, second), glossary="")

    assert result == LINGUISTIC_CONTEXT_BLOCK_TEMPLATE.format(
        instructions="Use MASCULINE forms; Use PLURAL"
    )


# ============================================================================
# _replace_last_marker / _find_sentence_start
# ============================================================================

@pytest.mark.parametrize("text, expected", [
    ("a{M}b", "aXb"),
    ("{M}a{M}b{M}", "abX"),
    ("no marker", "no marker"),
])
def test_replace_last_marker(processor, text, expected):
    assert processor._replace_last_marker(text, "{M}", "X") == expected


@pytest.mark.parametrize("text, expected_start", [
    ("Intro.\nTranslate now", len("Intro.\n")),
    ("Intro. Translate now", len("Intro. ")),
    ("Translate now", 0),
])
def test_find_sentence_start(processor, text, expected_start):
    position = text.lower().rfind("translate")

    assert processor._find_sentence_start(text, position) == expected_start


@pytest.mark.characterization(
    reason="\\n\\n wins over a nearer \\n: insertion goes to paragraph start, not sentence start")
def test_find_sentence_start_prefers_paragraph_break(processor):
    text = "Intro.\n\nSecond.\nTranslate now"

    assert processor._find_sentence_start(text, text.rfind("Translate")) == len("Intro.\n\n")
