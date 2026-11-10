# -*- coding: utf-8 -*-
"""
Unit tests for llm_conductor/l10n/context.py: KeyPath, LinguisticHint, LinguisticContext
detectors, hint inheritance, unknown-pattern audit, build_glossary_string.
"""
import re

import pytest

from llm_conductor.constants import GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE
from llm_conductor.l10n.context import (
    HintType,
    KeyPath,
    LinguisticContext,
    LinguisticHint,
    build_glossary_string,
)
from tests.support import StubConfig

# ============================================================================
# KeyPath
# ============================================================================

@pytest.mark.parametrize("segments, expected", [
    ((), "root"),
    (('items',), "items"),
    (('items', 'sword', 'name'), "items.sword.name"),
    (('dialogues', 0, 'text'), "dialogues[0].text"),
    (('matrix', 1, 2), "matrix[1][2]"),
    ((0, 'title'), "[0].title"),
])
def test_keypath_str(segments, expected):
    assert str(KeyPath(segments)) == expected


def test_keypath_report_key_distinguishes_literal_dot_from_nested():
    literal = KeyPath(("a.b",)).report_key()
    nested = KeyPath(("a", "b")).report_key()
    assert literal == '["a.b"]'
    assert nested == '["a","b"]'
    assert literal != nested
    assert str(KeyPath(("a.b",))) == str(KeyPath(("a", "b"))) == "a.b"


def test_keypath_append_returns_new_path_without_mutating_original():
    base = KeyPath(('a',))

    child = base.append('b')

    assert child.segments == ('a', 'b')
    assert base.segments == ('a',)


def test_keypath_is_hashable_and_compared_by_value():
    assert KeyPath(('a', 0)) == KeyPath(('a', 0))
    assert len({KeyPath(('a', 0)), KeyPath(('a', 0))}) == 1


def test_keypath_get_parent():
    assert KeyPath(('a', 'b')).get_parent() == KeyPath(('a',))
    assert KeyPath(()).get_parent() is None


def test_keypath_suffix_and_prefix_ignore_int_last_segment():
    path = KeyPath(('greeting_male', 0))

    assert not path.matches_suffix('_male')
    assert not path.matches_prefix('greeting')


def test_keypath_suffix_and_prefix_on_string_segment():
    path = KeyPath(('dialogues', 'speaker_male'))

    assert path.matches_suffix('_male')
    assert path.matches_prefix('speaker_')
    assert path.contains_segment('dialogues')


def test_keypath_iter_ancestors_from_root():
    path = KeyPath(('dialogues', 'greeting_male', 'text'))

    assert path.iter_ancestors() == [
        KeyPath(()), KeyPath(('dialogues',)), KeyPath(('dialogues', 'greeting_male')),
    ]
    assert path.iter_ancestors(include_self=True)[-1] == path


def test_keypath_get_string_segments():
    path = KeyPath(('a', 0, 'b'))

    assert path.get_string_segments() == ['a', 'b']
    assert path.get_string_segments(include_indices=True) == ['a', '0', 'b']


def test_keypath_find_segment_with_pattern_nearest_to_leaf_case_insensitive():
    path = KeyPath(('npc_FEMALE', 'greeting_female', 'text'))

    assert path.find_segment_with_pattern('_female') == 'greeting_female'
    assert path.find_segment_with_pattern('npc_', check_suffix=False) == 'npc_FEMALE'
    assert path.find_segment_with_pattern('_plural') is None


# ============================================================================
# LinguisticHint
# ============================================================================

@pytest.mark.parametrize("confidence", [-0.1, 1.01])
def test_linguistic_hint_rejects_confidence_out_of_range(confidence):
    with pytest.raises(ValueError, match="confidence"):
        LinguisticHint(HintType.CUSTOM, "instruction", confidence=confidence)


def test_linguistic_hint_rejects_blank_instruction():
    with pytest.raises(ValueError, match="instruction"):
        LinguisticHint(HintType.CUSTOM, "   ")


def test_linguistic_hints_sort_by_priority():
    low = LinguisticHint(HintType.PLURAL, "p", priority=5)
    high = LinguisticHint(HintType.GENDER, "g", priority=10)

    assert sorted([high, low]) == [low, high]


# ============================================================================
# LinguisticContext — built-in detectors
# ============================================================================

@pytest.fixture
def context():
    return LinguisticContext(StubConfig())


def _hint_types(hints):
    return [hint.hint_type for hint in hints]


@pytest.mark.parametrize("segments, expected_type, instruction_fragment", [
    (('greeting_male',), HintType.GENDER, "MASCULINE"),
    (('line_female',), HintType.GENDER, "FEMININE"),
    (('greeting_m',), HintType.GENDER, "MASCULINE"),
    (('greeting_f',), HintType.GENDER, "FEMININE"),
    (('items_plural',), HintType.PLURAL, "PLURAL"),
    (('item_singular',), HintType.PLURAL, "SINGULAR"),
    (('count_arrows',), HintType.PLURAL, "COUNTABLE"),
    (('dialog_informal',), HintType.FORMALITY, "INFORMAL"),
], ids=lambda value: str(KeyPath(value)) if isinstance(value, tuple) else None)
def test_detectors_positive_cases(context, segments, expected_type, instruction_fragment):
    hints = context.analyze(KeyPath(segments), "text")

    assert _hint_types(hints) == [expected_type]
    assert instruction_fragment in hints[0].instruction


# ============================================================================
# Detector instructions are language-neutral (no hardcoded
# target language or embedded Cyrillic grammar terms).
# ============================================================================

_CYRILLIC = re.compile(r'[Ѐ-ӿ]')


@pytest.mark.parametrize("segments", [
    ('greeting_male',), ('line_female',), ('greeting_m',), ('greeting_f',),
    ('greeting_neutral',), ('speaker_male',), ('speaker_female',),
    ('items_plural',), ('item_singular',), ('item_few',), ('item_zero',),
    ('count_arrows',), ('dialog_formal',), ('dialog_informal',),
], ids=lambda value: str(KeyPath(value)))
def test_task40_instructions_are_language_neutral(context, segments):
    hints = context.analyze(KeyPath(segments), "text")

    assert hints, f"expected at least one hint for {segments}"
    for hint in hints:
        assert not _CYRILLIC.search(hint.instruction), hint.instruction
        assert "russian" not in hint.instruction.lower(), hint.instruction


@pytest.mark.parametrize("segments, keyword", [
    (('greeting_male',), "MASCULINE"),
    (('line_female',), "FEMININE"),
    (('greeting_neutral',), "NEUTER"),
    (('speaker_male',), "Speaker is MALE"),
    (('speaker_female',), "Speaker is FEMALE"),
    (('items_plural',), "PLURAL"),
    (('item_singular',), "SINGULAR"),
    (('item_few',), "FEW"),
    (('item_zero',), "ZERO"),
    (('count_arrows',), "COUNTABLE"),
    (('dialog_formal',), "FORMAL"),
    (('dialog_informal',), "INFORMAL"),
])
def test_task40_keywords_survive_neutral_wording(context, segments, keyword):
    hints = context.analyze(KeyPath(segments), "text")

    assert keyword in hints[0].instruction


def test_hints_are_sorted_by_priority_descending(context):
    """gender (10) before plural (5) — order drives injection order into the prompt."""
    context.register_pattern("plural_stub", lambda path: LinguisticHint(HintType.PLURAL, "p", priority=5))
    context.clear_cache()

    hints = context.analyze(KeyPath(('greeting_male',)), "text")

    assert [h.priority for h in hints] == sorted((h.priority for h in hints), reverse=True)
    assert hints[0].hint_type == HintType.GENDER


def test_plain_key_without_patterns_gets_no_hints(context):
    assert context.analyze(KeyPath(('ui', 'title')), "text") == []


# ============================================================================
# Inheritance from ancestors (_scan_ancestors)
# ============================================================================

def test_hint_inherited_from_parent_with_reduced_priority_and_confidence(context):
    hints = context.analyze(KeyPath(('npc_female', 'greeting')), "text")

    assert len(hints) == 1
    inherited = hints[0]
    assert inherited.hint_type == HintType.GENDER
    assert inherited.instruction.startswith("(inherited from parent) ")
    assert inherited.priority == 10 - 5
    assert inherited.confidence == pytest.approx(1.0 * 0.8)


def test_hint_inherited_through_list_index(context):
    """An array leaf (last segment is int) inherits the ancestor hint."""
    hints = context.analyze(KeyPath(('npc_female', 'lines', 0)), "text")

    assert _hint_types(hints) == [HintType.GENDER]


def test_local_hint_overrides_ancestor_hint_of_same_type(context):
    hints = context.analyze(KeyPath(('npc_female', 'greeting_male')), "text")

    assert len(hints) == 1
    assert "MASCULINE" in hints[0].instruction
    assert "inherited" not in hints[0].instruction


def test_nearest_ancestor_wins(context):
    hints = context.analyze(KeyPath(('group_male', 'npc_female', 'greeting')), "text")

    assert "FEMININE" in hints[0].instruction


# ============================================================================
# Custom detectors and analysis LRU cache (doc/concepts.md)
# ============================================================================

def _detect_diminutive(path):
    if path.matches_suffix("_dim"):
        return LinguisticHint(HintType.CUSTOM, "Use DIMINUTIVE form", confidence=0.9, priority=6)
    return None


def test_register_pattern_adds_custom_detector(context):
    context.register_pattern("diminutive", _detect_diminutive)

    hints = context.analyze(KeyPath(('cat_dim',)), "text")

    assert _hint_types(hints) == [HintType.CUSTOM]


def test_register_pattern_rejects_non_callable(context):
    with pytest.raises(TypeError):
        context.register_pattern("broken", "not a function")


@pytest.mark.characterization(
    reason="§10.2 contract: register_pattern() could clear the analysis LRU itself")
def test_analysis_cache_requires_clear_after_registering_detector(context):
    """
    Documents the §10.2 contract: the analysis LRU is unaware of detector-set
    changes — clear_cache() is required after register_pattern().
    """
    path = KeyPath(('cat_dim',))
    assert context.analyze(path, "text") == []

    context.register_pattern("diminutive", _detect_diminutive)
    assert context.analyze(path, "text") == []  # stale result from the cache

    context.clear_cache()
    assert _hint_types(context.analyze(path, "text")) == [HintType.CUSTOM]


def test_failing_detector_is_logged_and_does_not_break_analysis():
    config = StubConfig()
    context = LinguisticContext(config)

    def broken(path):
        raise RuntimeError("detector bug")

    context.register_pattern("broken", broken)

    hints = context.analyze(KeyPath(('greeting_male',)), "text")

    assert _hint_types(hints) == [HintType.GENDER]
    assert config.logger.contains("Pattern detector 'broken' failed", level='WARNING')


def test_failing_detector_on_ancestor_is_logged_and_does_not_break_inheritance():
    """_scan_ancestors() must warn like analyze(), not swallow the failure silently."""
    config = StubConfig()
    context = LinguisticContext(config)
    calls = {'n': 0}

    def flaky_gender(path):
        # Blow up only on the ancestor walk (parent key), not on the leaf itself.
        if path.segments == ('npc_female',):
            calls['n'] += 1
            raise RuntimeError("ancestor detector bug")

    context.register_pattern("flaky_gender", flaky_gender)

    hints = context.analyze(KeyPath(('npc_female', 'greeting')), "text")

    assert calls['n'] >= 1
    assert _hint_types(hints) == [HintType.GENDER]
    assert "inherited" in hints[0].instruction
    assert config.logger.contains("failed while scanning ancestor", level='WARNING')


# ============================================================================
# Unknown-pattern audit
# ============================================================================

@pytest.mark.parametrize("segment", ["tooltip_hover", "sys_value", "debug_text", "key-with-dash"])
def test_suspicious_key_without_hints_is_collected(segment):
    config = StubConfig()
    context = LinguisticContext(config)

    context.analyze(KeyPath(('ui', segment)), "text")
    context.clear_cache()
    context.analyze(KeyPath(('ui', segment)), "text")  # repeat — no duplicate

    assert context.get_unknown_patterns() == frozenset({f"ui.{segment}"})
    assert len(config.logger.messages('WARNING')) == 1


@pytest.mark.parametrize("segment", ["title", "description", "item_name", "farewell"])
def test_ordinary_key_is_not_reported_as_unknown(context, segment):
    context.analyze(KeyPath(('ui', segment)), "text")

    assert context.get_unknown_patterns() == frozenset()


# ============================================================================
# Detector suffixes are matched as a whole segment, not a
# substring anywhere in the key (regression tests, fixed).
# ============================================================================

@pytest.mark.parametrize("segments", [
    ('button_name',),            # '_n' from '_name' used to -> NEUTER
    ('phone_number',),           # '_n' from '_number' used to -> NEUTER
    ('error_message',),          # '_m' from '_message' used to -> MASCULINE
    ('meta_flag',),              # '_f' from '_flag' used to -> FEMININE
    ('ui_main', 'title'),        # a false MASCULINE used to leak to the whole subtree
    ('settings_menu', 'label'),
], ids=lambda segments: str(KeyPath(segments)))
def test_task39_no_gender_hint_for_ordinary_words(context, segments):
    hints = context.analyze(KeyPath(segments), "text")

    assert HintType.GENDER not in _hint_types(hints)


def test_task39_formal_key_gets_only_formality(context):
    """'_f' from '_formal' used to add a spurious FEMININE next to FORMALITY."""
    hints = context.analyze(KeyPath(('dialog_formal',)), "text")

    assert _hint_types(hints) == [HintType.FORMALITY]


def test_task39_few_key_gets_only_plural(context):
    """'_f' from '_few' used to add a spurious FEMININE next to PLURAL."""
    hints = context.analyze(KeyPath(('item_few',)), "text")

    assert _hint_types(hints) == [HintType.PLURAL]


def test_task39_nonexistent_gets_no_gender_or_plural(context):
    """'_n' and '_none' from '_nonexistent' used to -> NEUTER + ZERO items."""
    hints = context.analyze(KeyPath(('item_nonexistent',)), "text")

    assert hints == []


def test_task39_meta_prefixed_key_reaches_unknown_pattern_audit(context):
    """
    A spurious FEMININE from '_flag' used to make the hint list non-empty,
    which made the unknown-pattern audit (system prefix meta_) never fire.
    """
    context.analyze(KeyPath(('meta_flag',)), "text")

    assert context.get_unknown_patterns() == frozenset({"meta_flag"})


def test_task39_speaker_prefix_branch_is_reachable(context):
    """
    The speaker_* branch in _detect_gender used to be unreachable: the
    generic '_male'/'_female' suffix check ran first and both matched
    "speaker_female" (a real suffix match, not a substring bug), so its
    specific wording never reached the prompt.
    """
    hints = context.analyze(KeyPath(('speaker_female',)), "text")

    assert "Speaker is FEMALE" in hints[0].instruction


# ============================================================================
# build_glossary_string
# ============================================================================

_GLOSSARY = {"Guild": "Гильдия", "gold": "золото", "HP": "ОЗ"}


def test_glossary_matches_whole_words_case_insensitive():
    result = build_glossary_string("Welcome to the guild, take your GOLD.", _GLOSSARY, StubConfig())

    assert result == GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE.format(
        entries="Guild = Гильдия; gold = золото"
    )


def test_glossary_respects_word_boundaries():
    """'goldsmith' does not contain the term 'gold' as a whole word."""
    assert build_glossary_string("The goldsmith", _GLOSSARY, StubConfig()) == ""


def test_glossary_without_word_boundaries_matches_substring():
    result = build_glossary_string("The goldsmith", _GLOSSARY, StubConfig(), use_word_boundaries=False)

    assert "gold = золото" in result


@pytest.mark.parametrize("glossary", [{}, None])
def test_empty_glossary_gives_empty_string(glossary):
    assert build_glossary_string("Guild", glossary, StubConfig()) == ""


def test_glossary_falls_back_to_direct_regex_without_pattern_cache():
    config = StubConfig(glossary_cache=None)

    result = build_glossary_string("HP restored", _GLOSSARY, config)

    assert result == GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE.format(entries="HP = ОЗ")
    assert config.logger.contains("Glossary pattern cache not initialized", level='WARNING')


@pytest.mark.parametrize("term, text", [
    ("C++", "Learn C++ now"),
    ("Dr.", "Ask Dr. Who"),
    (".NET", "Built on .NET today"),
    ("(A)", "Press (A) to jump"),
])
def test_task50_term_with_punctuation_edge_matches_with_word_boundaries(term, text):
    """
    use_word_boundaries=True used to build \\bterm\\b; \\b never matches
    between '+' and a space, so the term silently dropped out of the prompt.
    """
    result = build_glossary_string(text, {term: "перевод"}, StubConfig())

    assert f"{term} = перевод" in result


def test_glossary_term_with_regex_metacharacters_is_escaped():
    result = build_glossary_string("Press (A) to jump", {"(A)": "(А)"}, StubConfig(),
                                   use_word_boundaries=False)

    assert "(A) = (А)" in result
