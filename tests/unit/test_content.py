"""Content contract and executable TOML recipes, without model calls."""

import re
import tomllib
from pathlib import Path

import pytest

from llm_conductor.cli.draft_validate import compare_draft_markers
from llm_conductor.config import ConfigData
from llm_conductor.constants import (
    DEFAULT_MASK_PATTERN,
    DEFAULT_REQUIRED_CONTENT_PATTERN,
)
from llm_conductor.exceptions import ConfigValidationError
from llm_conductor.l10n.content import ContentRules
from llm_conductor.l10n.validation import validate_model_reply
from llm_conductor.utils import mask_tags, unmask_tags
from tests.support import StubConfig


@pytest.mark.parametrize('text, accepted', [
    ('Hello', True), ('Привет', True), ('你好', True), ('é', True),
    (' +0042 ', False), ('-007', False), ('2026-11-09', False),
    ('Ⅳ', False), ('²', False), ('...', False), ('😀', False),
    ('<ns:item id="abc">42</ns:item>', False), ('<ns:item/>', False),
    ('{0}', False), ('<item>Hello {0}</item>', True),
])
def test_default_eligibility_preserves_letter_contract(text, accepted):
    assert ContentRules().accepts(text) is accepted


@pytest.mark.parametrize('required', ['', r'\p{L}', r'[\p{L}\p{Nd}]', r'\s'])
@pytest.mark.parametrize('text', ['', ' ', '\t\r\n', '\u2003'])
def test_blank_leaves_never_admitted(required, text):
    assert not ContentRules(required, '').accepts(text)


def test_spaces_hide_protected_text_without_joining_neighbors():
    rules = ContentRules(r'Open.*door', r'\{[^}]+\}')
    assert rules.eligibility_text('  Open {player_name} door\t') == 'Open   door'
    assert rules.accepts('  Open {player_name} door\t')
    assert not ContentRules('Open', rules.mask_pattern).accepts('Op{0}en')
    # A single protected occurrence contributes exactly one space.
    assert rules.eligibility_text('a{very_long_name}{0}b') == 'a  b'
    assert ContentRules(r'^a  b$', rules.mask_pattern).accepts('a{foo}{bar}b')


def test_custom_mask_replaces_defaults_and_capture_groups_use_whole_match():
    rules = ContentRules(mask_pattern=r'\{([a-z]+)\}')
    source = '<b>Hello {name}, {name}\n[T0]</b>'
    masked, mapping = mask_tags(source, rules)
    assert masked == '<b>Hello [A0], [A1]\n[T0]</b>'
    assert mapping == {'[A0]': '{name}', '[A1]': '{name}'}
    assert unmask_tags(masked, mapping, StubConfig()).text == source
    assert rules.protected_fragments(source) == ['{name}', '{name}']


def test_zero_width_matches_protect_nothing():
    rules = ContentRules('', r'(?=x)|\{\d+\}')
    assert rules.protected_fragments('x{0}') == ['{0}']
    assert mask_tags('x{0}', rules)[0] == 'x[T0]'


def test_disabled_mask_does_not_reserve_literal_marker_syntax():
    config = StubConfig()
    config.processing = ContentRules('', '')
    source = 'Literal [T0]'
    masked, mapping = mask_tags(source, config.processing)
    restored = unmask_tags(masked, mapping, config)
    assert restored.text == source
    assert restored.orphaned_markers == []
    assert validate_model_reply(source, masked, restored.text, mapping, [], [], {}, config).accepted


@pytest.mark.parametrize('key', ['required_content_pattern', 'mask_pattern'])
@pytest.mark.parametrize('pattern', ['[', '(?V1)x'])
def test_bad_regex_is_a_named_config_error(key, pattern):
    with pytest.raises(ConfigValidationError, match=rf'\[processing\].{key}'):
        ConfigData.from_toml({'processing': {key: pattern}}, {}, environ={})


def test_missing_rules_keep_defaults_and_empty_rules_disable_them():
    defaults = ConfigData.from_toml({}, {}, environ={}).processing
    assert defaults.identity == (DEFAULT_REQUIRED_CONTENT_PATTERN, DEFAULT_MASK_PATTERN)
    custom = ConfigData.from_toml({'processing': {
        'required_content_pattern': '', 'mask_pattern': '',
    }}, {}, environ={}).processing
    assert custom.accepts('42')
    assert custom.protected_fragments('<b>{0}</b>\n') == []


def test_processing_section_is_rejected_in_conductor_config():
    with pytest.raises(ConfigValidationError, match='processing'):
        ConfigData.from_toml({}, {'processing': {'mask_pattern': ''}}, environ={})


@pytest.mark.parametrize('mask, accepted', [('', True), (r'\{\d+\}', True), (r'\n', False)])
def test_raw_newlines_follow_active_protection(mask, accepted):
    config = StubConfig()
    config.processing = ContentRules('', mask)
    result = validate_model_reply('Hello', 'Hello\nworld', 'Hello\nworld', {}, [], [], {}, config)
    assert result.accepted is accepted
    if not accepted:
        assert result.reason == 'newline_in_reply'


def test_custom_protection_applies_to_drafts_and_reply_inventions():
    rules = ContentRules(mask_pattern=r'\{([a-z]+)\}')
    original = {'x': 'Hello {name}'}
    assert compare_draft_markers(original, {'x': 'Hi {name}'}, rules) == []
    assert compare_draft_markers(original, {'x': 'Hi {other}'}, rules)
    assert compare_draft_markers(original, {'x': 'Hi {other}'}, ContentRules(mask_pattern='')) == []
    config = StubConfig()
    config.processing = rules
    result = validate_model_reply('Hello', 'Hello {name}', 'Hello {name}', {}, [], [], {}, config)
    assert result.reason == 'fabricated_tag'


def test_draft_anchor_rules_see_peeled_core_boundaries():
    rules = ContentRules(mask_pattern=r'^ID:\d+')
    source = {'x': ' ID:42 Hello\t'}
    assert compare_draft_markers(source, {'x': ' ID:43 Hi\t'}, rules)
    assert compare_draft_markers(source, {'x': ' ID:42 Hi\t'}, rules) == []


def _recipes():
    source = (Path(__file__).resolve().parents[2] / 'doc' / 'content.md').read_text(encoding='utf-8')
    return {
        heading: tomllib.loads(block)
        for heading, block in re.findall(r'^### ([^\n]+)\n\n```toml\n(.*?)\n```', source, re.MULTILINE | re.DOTALL)
    }


@pytest.mark.parametrize('heading', [
    'Letters only (default)', 'Letters or decimal digits', 'Every nonblank string',
    'No masking', 'HTML and XML tags', 'Positional and named brace placeholders',
    'Double-brace placeholders', 'Dollar-brace placeholders', 'Printf placeholders',
    'Combined localization recipe',
])
def test_documented_recipe_is_valid_toml_and_config(heading):
    rules = ConfigData.from_toml(_recipes()[heading], {}, environ={}).processing
    assert not rules.accepts('\t\n ')


@pytest.mark.parametrize('heading,text,fragments', [
    ('HTML and XML tags', '<item id="42">Hi</item><ns:item/><br />',
     ['<item id="42">', '</item>', '<ns:item/>', '<br />']),
    ('HTML and XML tags', '<item test="a > b" other=\'x > y\'>Hi</item>',
     ['<item test="a > b" other=\'x > y\'>', '</item>']),
    ('HTML and XML tags', '<элемент атрибут="42">Hi</элемент>',
     ['<элемент атрибут="42">', '</элемент>']),
    ('Positional and named brace placeholders', 'Hi {0} {42} {name} {player_name}',
     ['{0}', '{42}', '{name}', '{player_name}']),
    ('Double-brace placeholders', '{{name}} {{ user.name }}', ['{{name}}', '{{ user.name }}']),
    ('Dollar-brace placeholders', '${name} ${user.name}', ['${name}', '${user.name}']),
    ('Printf placeholders', '%s %d %02d %.2f %1$s %(name)s %%',
     ['%s', '%d', '%02d', '%.2f', '%1$s', '%(name)s', '%%']),
    ('Combined localization recipe', '<ns:item a=">">Hi {{ user.name }} ${name} {0} %02d\r\n</ns:item>',
     ['<ns:item a=">">', '{{ user.name }}', '${name}', '{0}', '%02d', '\r\n', '</ns:item>']),
    ('Combined localization recipe', '<элемент>Hi {0}</элемент>',
     ['<элемент>', '{0}', '</элемент>']),
])
def test_documented_protection_recipes_match_complete_fragments(heading, text, fragments):
    rules = ConfigData.from_toml(_recipes()[heading], {}, environ={}).processing
    assert rules.protected_fragments(text) == fragments
    masked, mapping = mask_tags(text, rules)
    config = StubConfig()
    config.processing = rules
    restored = unmask_tags(masked, mapping, config)
    assert restored.text == text
    assert validate_model_reply(
        text, masked, restored.text, mapping,
        restored.missing_markers, restored.orphaned_markers, {}, config,
    ).accepted


@pytest.mark.parametrize('heading,accepted', [
    ('Letters only (default)', False), ('Letters or decimal digits', True),
    ('Every nonblank string', True),
])
def test_documented_eligibility_recipes(heading, accepted):
    rules = ConfigData.from_toml(_recipes()[heading], {}, environ={}).processing
    for value in (' +0042 ', '-007', '2026-11-09'):
        assert rules.accepts(value) is accepted
