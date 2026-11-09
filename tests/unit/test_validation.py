# -*- coding: utf-8 -*-
"""
Unit tests of validate_model_reply() (llm_conductor/l10n/validation.py).

Each case builds masked_reply/unmasked_reply/missing_markers/orphaned_markers
explicitly, the same values _process_leaf() would pass after a real
mask_tags()/unmask_tags() round trip - that keeps "the model echoed the
marker correctly" and "the model's own output" unambiguous, which a
mask_tags()-on-the-final-text shortcut would blur.
"""
from llm_conductor.config import ValidationConfig
from llm_conductor.constants import EXAMPLE_EN_RU_ALLOWED_SCRIPTS
from llm_conductor.l10n.validation import validate_model_reply
from tests.support import StubConfig


def _config(max_length_ratio=3.0, max_length_slack=20, latin_text_policy='warn',
            allowed_scripts=EXAMPLE_EN_RU_ALLOWED_SCRIPTS):
    return StubConfig(validation=ValidationConfig(
        enabled=True, max_length_ratio=max_length_ratio,
        max_length_slack=max_length_slack, latin_text_policy=latin_text_policy,
        allowed_scripts=tuple(allowed_scripts),
    ))


def _validate(original_value, masked_reply, unmasked_reply=None, tag_map=None,
             missing_markers=None, orphaned_markers=None, glossary=None, config=None):
    return validate_model_reply(
        original_value, masked_reply,
        unmasked_reply if unmasked_reply is not None else masked_reply,
        tag_map or {}, missing_markers or [], orphaned_markers or [],
        glossary or {}, config or _config(),
    )


def test_accepts_a_faithful_reply_with_matching_tags():
    result = _validate(
        "You have {0} gold coins.",
        masked_reply="У вас [T0] золотых монет.",
        unmasked_reply="У вас {0} золотых монет.",
        tag_map={'[T0]': '{0}'},
    )

    assert result == (True, None, None, None)


def test_rejects_literal_newline_in_reply():
    """A real '\\n' in the reply is always chatter - the source's own
    newlines are masked into [Tn] before the call."""
    result = _validate("Hello", masked_reply="Hello\nAnything else?")

    assert result.accepted is False
    assert result.reason == 'newline_in_reply'
    assert result.detail is None


def test_rejects_lost_tag_marker():
    result = _validate(
        "Press {0} to jump",
        masked_reply="Нажмите для прыжка",     # [T0] never reproduced
        tag_map={'[T0]': '{0}'},
        missing_markers=['[T0]'],
    )

    assert result.accepted is False
    assert result.reason == 'tag_marker_mismatch'
    assert result.detail == 'missing=[T0]'


def test_rejects_orphaned_literal_marker_left_in_reply():
    result = _validate(
        "Plain text",
        masked_reply="Обычный текст [T3]",
        unmasked_reply="Обычный текст [T3]",   # nothing in tag_map to replace it with
        orphaned_markers=['[T3]'],
    )

    assert result.accepted is False
    assert result.reason == 'tag_marker_mismatch'
    assert result.detail == 'extra=[T3]'


def test_rejects_duplicated_tag_marker_before_fabricated_tag():
    """A doubled expected marker is tag_marker_mismatch, not fabricated_tag."""
    result = _validate(
        "Press {0} to jump",
        masked_reply="Нажмите [T0] для [T0] прыжка",
        unmasked_reply="Нажмите {0} для {0} прыжка",
        tag_map={'[T0]': '{0}'},
    )

    assert result.accepted is False
    assert result.reason == 'tag_marker_mismatch'
    assert result.detail == 'duplicated=[T0]'


def test_rejects_combined_missing_and_extra_markers():
    result = _validate(
        "Press {0} to jump",
        masked_reply="Нажмите [T9] для прыжка",
        unmasked_reply="Нажмите [T9] для прыжка",
        tag_map={'[T0]': '{0}'},
        missing_markers=['[T0]'],
        orphaned_markers=['[T9]'],
    )

    assert result.accepted is False
    assert result.reason == 'tag_marker_mismatch'
    assert result.detail == 'missing=[T0]; extra=[T9]'


def test_rejects_fabricated_tag_not_present_in_original():
    """The model invents a {0}-shaped placeholder that was never a real marker."""
    result = _validate(
        "No placeholders here",
        masked_reply="({0}вход 82{0}) здесь нет плейсхолдеров",
    )

    assert result.accepted is False
    assert result.reason == 'fabricated_tag'
    assert result.detail is None


def test_rejects_response_far_longer_than_the_source():
    essay = "Сложная задача или задание было не указано в предоставленном тексте. " * 3

    result = _validate("Hard", masked_reply=essay)

    assert result.accepted is False
    assert result.reason == 'response_too_long'


def test_accepts_within_length_slack_for_short_strings():
    # len("Hard") * 3 + 20 == 32; this reply is under that.
    result = _validate("Hard", masked_reply="Очень сложно")

    assert result.accepted is True


def test_rejects_foreign_script_text():
    result = _validate(
        "<b>Warning:</b> this action cannot be undone.",
        masked_reply="[T0]Предупреждение:[T1] эту动作不能被撤销",
        unmasked_reply="<b>Предупреждение:</b> эту动作不能被撤销",
        tag_map={'[T0]': '<b>', '[T1]': '</b>'},
    )

    assert result.accepted is False
    assert result.reason == 'foreign_script'


def test_empty_allowed_scripts_rejects_cyrillic():
    """Absent/empty allowed_scripts assumes no target alphabet."""
    result = _validate(
        "Hello", masked_reply="Привет",
        config=_config(allowed_scripts=()),
    )

    assert result.accepted is False
    assert result.reason == 'foreign_script'


def test_cyrillic_allowed_accepts_russian_rejects_cjk():
    result_ok = _validate("Hello", masked_reply="Привет",
                          config=_config(allowed_scripts=('cyrillic',)))
    result_cjk = _validate("Hello", masked_reply="你好",
                           config=_config(allowed_scripts=('cyrillic',)))

    assert result_ok.accepted is True
    assert result_cjk.accepted is False
    assert result_cjk.reason == 'foreign_script'


def test_latin_in_allowed_scripts_skips_latin_policy():
    result = _validate(
        "Hallo", masked_reply="Willkommen",
        config=_config(allowed_scripts=('latin',), latin_text_policy='reject'),
    )

    assert result.accepted is True
    assert result.warning is None


def test_cjk_allowed_accepts_han_and_kana():
    result = _validate(
        "Hello", masked_reply="こんにちは世界",
        config=_config(allowed_scripts=('cjk',)),
    )

    assert result.accepted is True


def test_latin_absent_from_source_warns_under_warn_policy():
    result = _validate(
        "Привет, как дела?", masked_reply="Привет Hello",
        config=_config(latin_text_policy='warn'),
    )

    assert result.accepted is True
    assert result.reason is None
    assert result.warning is not None


def test_latin_absent_from_source_rejected_under_reject_policy():
    result = _validate(
        "Привет, как дела?", masked_reply="Привет Hello",
        config=_config(latin_text_policy='reject'),
    )

    assert result.accepted is False
    assert result.reason == 'unexpected_latin'


def test_latin_present_in_source_is_never_flagged():
    """A preserved proper noun (brand name) kept verbatim is not chatter."""
    result = _validate("Welcome to Arasaka Tower", masked_reply="Добро пожаловать в башню Arasaka")

    assert result.accepted is True
    assert result.warning is None


def test_latin_fixed_by_glossary_is_never_flagged():
    result = _validate(
        "Добро пожаловать", masked_reply="Добро пожаловать в Arasaka",
        glossary={"Arasaka": "Arasaka"},
    )

    assert result.accepted is True
    assert result.warning is None
