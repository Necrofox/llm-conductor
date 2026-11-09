# -*- coding: utf-8 -*-
"""
Unit tests for llm_conductor/utils.py: tag masking, timeouts.

mask_tags/unmask_tags are priority #1 per AGENTS.md: two non-obvious
invariants (doc/pipeline.md) that are easy to break by editing the regex or replacement
order.
"""
import builtins
import threading

import pytest

from llm_conductor.exceptions import ConductorError
from llm_conductor.exceptions import TimeoutError as ConductorTimeoutError
from llm_conductor.utils import (
    ThreadPoolTimeoutStrategy,
    is_format_only_leaf,
    mask_tags,
    run_with_timeout,
    split_boundary_whitespace,
    unmask_tags,
)
from tests.support import StubConfig

# ============================================================================
# boundary whitespace / format-only
# ============================================================================

@pytest.mark.parametrize("text,expected", [
    (" ({0})", (" ", "({0})", "")),
    ("  Open.\t", ("  ", "Open.", "\t")),
    ("Hello", ("", "Hello", "")),
    ("", ("", "", "")),
    ("   ", ("   ", "", "")),
    ("\nHi\n", ("\n", "Hi", "\n")),
])
def test_split_boundary_whitespace(text, expected):
    assert split_boundary_whitespace(text) == expected
    prefix, core, suffix = expected
    assert prefix + core + suffix == text


@pytest.mark.parametrize("text,expected", [
    (" ({0})", True),
    ("{0}", True),
    ("<b></b>", True),
    ("()", True),
    ("{0} {1}", True),
    ("  Open.\t", False),
    ("Hello {0}", False),
    ("", False),
    ("   ", False),
    ("100%", True),
    ("42", True),
])
def test_is_format_only_leaf(text, expected):
    assert is_format_only_leaf(text) is expected


# ============================================================================
# mask_tags
# ============================================================================

def test_mask_tags_replaces_placeholders_html_and_newlines_in_order():
    masked, tag_map = mask_tags("You have {0} <b>gold</b>.\nBye")

    assert masked == "You have [T0] [T1]gold[T2].[T3]Bye"
    assert tag_map == {'[T0]': '{0}', '[T1]': '<b>', '[T2]': '</b>', '[T3]': '\n'}


def test_mask_tags_without_tags_returns_text_unchanged():
    masked, tag_map = mask_tags("Plain text, no tags at all.")

    assert masked == "Plain text, no tags at all."
    assert tag_map == {}


def test_mask_tags_takes_full_match_of_tag_with_attributes():
    """Non-capturing groups + group(0): a tag with attributes is one full match."""
    masked, tag_map = mask_tags('<color="#ff0000">Alert</color>')

    assert masked == "[T0]Alert[T1]"
    assert tag_map['[T0]'] == '<color="#ff0000">'


@pytest.mark.parametrize("text", ["{a}", "{}", "{-1}", "{ 0 }"])
def test_mask_tags_ignores_non_numeric_braces(text):
    """Only {digits} counts as a placeholder — other braces are left alone."""
    masked, tag_map = mask_tags(text)

    assert masked == text
    assert tag_map == {}


# ============================================================================
# unmask_tags
# ============================================================================

@pytest.mark.parametrize("text", [
    "You have {0} gold coins.",
    "<b>Warning:</b> this action cannot be undone.",
    "First line.\nSecond line.",
    "{0} of {1} <i>{2}</i>\n\n<br/>",
    "",
])
def test_mask_unmask_roundtrip_restores_original(text):
    masked, tag_map = mask_tags(text)

    assert unmask_tags(masked, tag_map, StubConfig()).text == text


def test_unmask_tags_with_two_digit_markers():
    """
    Two-digit markers [T10]..[T11] restore the same way as [T0]..[T9].

    Mutation testing showed this test still passes with reverse replacement
    order: [T1] is NOT a prefix of [T10] — the closing bracket prevents it.
    Length sorting in unmask_tags() protects against nothing (doc/pipeline.md);
    the real marker collision is a literal [Tn] in the source text.
    """
    text = "".join(f"{{{i}}}" for i in range(12))  # {0}{1}...{11}
    masked, tag_map = mask_tags(text)

    assert "[T10]" in masked and "[T11]" in masked
    assert unmask_tags(masked, tag_map, StubConfig()).text == text


@pytest.mark.parametrize("text", [
    "<b>[T1]</b>",                   # literal used to collide with the </b> marker
    "Press [T0] to {0}",             # literal used to turn into an unrelated tag
    '<a title="[T1]">x</a>',         # marker-shaped literal inside the tag itself
])
def test_task54_literal_marker_in_source_text_survives_roundtrip(text):
    masked, tag_map = mask_tags(text)

    assert unmask_tags(masked, tag_map, StubConfig()).text == text


def test_unmask_tags_survives_reordered_markers():
    """The model may swap markers — tags follow the markers."""
    _, tag_map = mask_tags("{0} and {1}")

    assert unmask_tags("[T1] and [T0]", tag_map, StubConfig()).text == "{1} and {0}"


def test_unmask_tags_warns_about_lost_markers_in_debug():
    config = StubConfig(is_debug_mode=True)
    _, tag_map = mask_tags("<b>Bold</b>")

    result = unmask_tags("Жирный", tag_map, config)

    assert result.text == "Жирный"
    assert sorted(result.missing_markers) == ['[T0]', '[T1]']
    assert result.orphaned_markers == []
    assert config.logger.contains("2 marker(s) not found", level='WARNING')


def test_unmask_tags_warns_about_orphaned_markers_in_debug():
    config = StubConfig(is_debug_mode=True)

    result = unmask_tags("Текст [T7]", {}, config)

    assert result.text == "Текст [T7]"
    assert result.missing_markers == []
    assert result.orphaned_markers == ['[T7]']
    assert config.logger.contains("orphaned marker", level='WARNING')


@pytest.mark.characterization(
    reason="lost/swapped markers are silent outside debug — #48 requires the opposite")
def test_unmask_tags_is_silent_outside_debug():
    """Outside debug mode, lost/extra markers are not logged (§4.6)."""
    config = StubConfig(is_debug_mode=False)
    _, tag_map = mask_tags("<b>Bold</b>")

    unmask_tags("Жирный [T9]", tag_map, config)

    assert config.logger.records == []


# ============================================================================
# run_with_timeout / timeout strategies
# ============================================================================

def test_run_with_timeout_returns_function_result():
    assert run_with_timeout(lambda: 42, timeout_seconds=5) == 42


def test_run_with_timeout_propagates_function_exception():
    def boom():
        raise ValueError("inner failure")

    with pytest.raises(ValueError, match="inner failure"):
        run_with_timeout(boom, timeout_seconds=5)


def test_thread_pool_strategy_raises_conductor_timeout():
    """
    Exercises the Windows strategy directly (on POSIX run_with_timeout would
    pick SIGALRM). The worker is released via an event so it does not hang
    in the background.
    """
    release = threading.Event()
    try:
        with pytest.raises(ConductorTimeoutError, match="slow_op timed out"):
            ThreadPoolTimeoutStrategy().run(lambda: release.wait(5), 0.05, "slow_op")
    finally:
        release.set()


def test_conductor_timeout_error_is_also_builtin_timeout_and_oserror():
    """One object is caught by both ConductorError and the builtin excepts."""
    error = ConductorTimeoutError("x")

    assert isinstance(error, ConductorError)
    assert isinstance(error, builtins.TimeoutError)
    assert isinstance(error, OSError)
