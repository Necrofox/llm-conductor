# -*- coding: utf-8 -*-
"""
Property-based tests (hypothesis) for pure functions with fragile invariants.

Unlike example tests, inputs are generated — the test checks a property
that must hold for ANY input, not only the ones the author thought of.
That is how a bug was found: an example test for [T1]/[T10] passed,
while a literal [Tn] in the text broke unmasking.

Regression properties (former xfail) carry an explicit @example with a
concrete counterexample from the backlog — hypothesis runs explicit
examples first, so a random breakage fails deterministically.
"""
import itertools
import re

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from llm_conductor.constants import TAG_MASK_PATTERN
from llm_conductor.l10n.context import KeyPath, LinguisticContext, build_glossary_string
from llm_conductor.l10n.processor import RecursiveProcessor
from llm_conductor.utils import mask_tags, unmask_tags
from tests.support import StubConfig

# Default hypothesis deadline (200 ms) false-fires on Windows cold start;
# the functions themselves are fast.
SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

# Text built from "risky" fragments: tags, placeholders, newlines, square
# brackets and literal markers — hits edges more often than random Unicode.
TOKENS = ["{0}", "{1}", "{12}", "<b>", "</b>", '<a href="x">', "\n", "[", "]", "T", "[T0]", "[T1]",
          "[T10]", "word", " ", "Привет", "{", "}", "<", ">"]
risky_text = st.lists(st.sampled_from(TOKENS), max_size=25).map("".join)
any_text = st.one_of(risky_text, st.text(max_size=60))

LITERAL_MARKER = re.compile(r'\[T\d+\]')


def _config():
    return StubConfig()


# ============================================================================
# mask_tags / unmask_tags
# ============================================================================

@SETTINGS
@given(any_text)
def test_mask_leaves_no_tags_and_numbers_markers_sequentially(text):
    masked, tag_map = mask_tags(text)

    assert TAG_MASK_PATTERN.search(masked) is None
    # The prefix defaults to "T" but mask_tags() switches to a collision-free
    # one when the source text already contains a literal "[T<digits>]"
    # Markers are still sequential, just under whatever single prefix was
    # actually chosen.
    if tag_map:
        prefix_match = re.match(r'\[([A-Za-z]+)\d+\]', next(iter(tag_map)))
        assert prefix_match is not None
        prefix = prefix_match.group(1)
    else:
        prefix = "T"
    assert list(tag_map) == [f"[{prefix}{i}]" for i in range(len(tag_map))]
    assert all(TAG_MASK_PATTERN.fullmatch(tag) for tag in tag_map.values())


@SETTINGS
@given(any_text.filter(lambda t: not LITERAL_MARKER.search(t)))
def test_roundtrip_without_literal_markers(text):
    masked, tag_map = mask_tags(text)

    assert unmask_tags(masked, tag_map, _config()).text == text


@SETTINGS
@given(any_text)
@example("<b>[T1]</b>")
@example("Press [T0] to {0}")
def test_task54_roundtrip_for_any_text(text):
    masked, tag_map = mask_tags(text)

    assert unmask_tags(masked, tag_map, _config()).text == text


@SETTINGS
@given(st.lists(st.sampled_from(["{0}", "<i>", "</i>", "\n", "{7}"]), min_size=1, max_size=12),
       st.randoms(use_true_random=False))
def test_model_reordering_markers_keeps_every_tag(tags, rnd):
    """The model may reorder markers — the tag multiset in the result is the same."""
    text = " w ".join(tags)
    _masked, tag_map = mask_tags(text)
    markers = list(tag_map)
    rnd.shuffle(markers)

    result = unmask_tags(" ".join(markers), tag_map, _config())

    assert sorted(TAG_MASK_PATTERN.findall(result.text)) == sorted(tags)


# ============================================================================
# _replace_last_marker: check against a reference implementation
# ============================================================================

MARKER = "{GLOSSARY}"
marker_text = st.lists(st.sampled_from([MARKER, "a", " ", "{", "GLOSSARY", "}", "\n"]), max_size=15).map("".join)


@SETTINGS
@given(marker_text, st.text(alphabet="xyz[]\n", max_size=10))
def test_replace_last_marker_matches_reference(text, replacement):
    """Reference: drop every occurrence, insert replacement at the last one's place."""
    processor = RecursiveProcessor(_config(), LinguisticContext(_config()))
    parts = text.split(MARKER)
    if len(parts) == 1:
        expected = text
    elif not replacement:
        expected = "".join(parts)
    else:
        expected = "".join(parts[:-1]) + replacement + parts[-1]

    assert processor._replace_last_marker(text, MARKER, replacement) == expected


# ============================================================================
# Glossary
# ============================================================================

word = st.text(alphabet="abcXYZ", min_size=1, max_size=8)


@SETTINGS
@given(word, word, word)
def test_glossary_finds_word_term_surrounded_by_spaces(prefix, term, suffix):
    text = f"{prefix} {term} {suffix}"

    assert f"{term} = T" in build_glossary_string(text, {term: "T"}, _config())


@SETTINGS
@given(word, word)
def test_glossary_never_matches_term_glued_inside_word(term, glue):
    assert build_glossary_string(f"{glue}{term}{glue}", {term: "T"}, _config()) == ""


@SETTINGS
@given(word, st.sampled_from(["+", "++", ".", "#", ")"]))
@example("C", "++")
def test_task50_term_with_punctuation_suffix_is_found(base, punctuation):
    term = base + punctuation

    assert f"{term} = T" in build_glossary_string(f"use {term} now", {term: "T"}, _config())


# ============================================================================
# KeyPath
# ============================================================================

segment = st.one_of(st.text(alphabet="abc_", min_size=1, max_size=6), st.integers(0, 50))


@SETTINGS
@given(st.lists(segment, max_size=8))
def test_keypath_ancestors_and_parent_are_consistent(segments):
    path = KeyPath(tuple(segments))
    ancestors = path.iter_ancestors(include_self=True)

    assert len(ancestors) == len(segments) + 1
    assert ancestors[-1] == path
    for parent, child in itertools.pairwise(ancestors):
        assert child.get_parent() == parent
