"""Exercise content policies through CLI configuration and intercepted HTTP."""

import json

import pytest
import responses

from llm_conductor.constants import (
    EXIT_CONFIG_ERROR,
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
)
from llm_conductor.l10n.context import KeyPath
from tests.support import OLLAMA_URL, dump_toml, ollama_reply
from tests.system.harness import base_docs


@pytest.fixture
def http():
    """Intercept HTTP; tests assert request counts and results directly."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


def test_numeric_policy_across_translate_and_standalone_verify(workspace, http, run_main):
    workspace.write_configs(preflight__check_availability=False,
                            processing__required_content_pattern='',
                            validation__allowed_scripts=['latin'])
    workspace.write_json('sample.json', {'x': ' +0042 ', 'empty': '', 'blank': '\t\r\n '})
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('42'))
    assert run_main('run', '--target', 'sample.json', '--stage', 'translate') == EXIT_SUCCESS
    assert workspace.read_json(workspace.result('translated')) == {
        'x': ' 42 ', 'empty': '', 'blank': '\t\r\n ',
    }
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('43'))
    assert run_main('run', '--target', 'sample.json', '--stage', 'verify') == EXIT_SUCCESS
    assert workspace.read_json(workspace.result('final'))['x'] == ' 43 '
    assert len(http.calls) == 2
    requests = [json.loads(call.request.body)['messages'][-1]['content'] for call in http.calls]
    assert requests[0].endswith('+0042')
    assert requests[1].endswith('42')
    assert '[ORIGINAL]:  +0042 ' in requests[1]
    sidecar = workspace.read_json(workspace.result('leaves'))
    leaf = sidecar['leaves'][KeyPath(('x',)).report_key()]
    assert leaf['translation']['status'] == 'ok'
    assert leaf['verification']['status'] == 'ok'
    for stage in ('translation', 'verification'):
        assert sidecar['leaves'][KeyPath(('blank',)).report_key()][stage]['status'] == 'skipped'


def test_enable_numeric_verification_after_default_translation_skip(workspace, http, run_main):
    workspace.write_configs(preflight__check_availability=False)
    workspace.write_json('sample.json', {'x': '-007'})
    assert run_main('run', '--target', 'sample.json', '--stage', 'translate') == EXIT_SUCCESS
    assert len(http.calls) == 0
    workspace.write_configs(preflight__check_availability=False,
                            processing__required_content_pattern='')
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('-7'))
    assert run_main('run', '--target', 'sample.json', '--stage', 'verify') == EXIT_SUCCESS
    assert workspace.read_json(workspace.result('final')) == {'x': '-7'}
    assert len(http.calls) == 1


def test_numeric_policy_does_not_bypass_stale_source_gate(workspace, http, run_main):
    workspace.write_configs(preflight__check_availability=False,
                            processing__required_content_pattern='',
                            resilience__fail_on_any_error=True)
    workspace.write_json('sample.json', {'x': '+0042'})
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('42'))
    assert run_main('run', '--target', 'sample.json', '--stage', 'translate') == EXIT_SUCCESS
    workspace.write_json('sample.json', {'x': '+0043'})
    assert run_main('run', '--target', 'sample.json', '--stage', 'verify') == EXIT_FAILURE_RATIO_EXCEEDED
    assert workspace.read_json(workspace.result('final')) == {'x': '+0043'}
    assert len(http.calls) == 1
    sidecar = workspace.read_json(workspace.result('leaves'))
    assert sidecar['leaves'][KeyPath(('x',)).report_key()]['verification']['error_kind'] == 'source_stale'


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('mask,expected', [(r'\{([a-z]+)\}', EXIT_USAGE_ERROR), ('', EXIT_SUCCESS)])
def test_draft_preflight_uses_active_protection(workspace, model, run_main, dry_run, mask, expected):
    workspace.write_configs(preflight__check_availability=False, processing__mask_pattern=mask)
    workspace.write_json('original.json', {'x': 'Hello {name}'})
    workspace.write_json('draft.json', {'x': 'Hi {other}'})
    workspace.write_text('job.toml', '[[targets]]\noriginal = "original.json"\ndraft = "draft.json"\n')
    args = ['run', '--target', 'job.toml']
    if dry_run:
        args.append('--dry-run')
    assert run_main(*args) == expected
    assert len(model.model_calls) == int(not dry_run and expected == EXIT_SUCCESS)
    if not dry_run and expected == EXIT_SUCCESS:
        assert workspace.read_json(workspace.result('final', 'original.json', 'job.toml')) == {
            'x': 'OK(Hi {other})',
        }


def test_custom_placeholder_protection_in_full_pipeline(workspace, model, run_main):
    workspace.write_configs(preflight__check_availability=False,
                            processing__mask_pattern=r'\{([a-z]+)\}')
    workspace.write_json('sample.json', {'x': 'Hello {name} {name}'})
    assert run_main('run', '--target', 'sample.json') == EXIT_SUCCESS
    assert len(model.model_calls) == 2
    assert workspace.read_json(workspace.result('final')) == {'x': 'OK(RU(Hello {name} {name}))'}


def test_invalid_content_regex_fails_before_http(workspace, model, run_main):
    workspace.write_configs(processing__mask_pattern='[')
    assert run_main('run', '--target', 'sample.json') == EXIT_CONFIG_ERROR
    assert model.model_calls == []


def _manifest_profiles(workspace, policies, *, reverse=False, drafts=False):
    """Write independent profiles and a manifest in either bootstrap order."""
    entries = []
    for name, policy in policies.items():
        llm, _ = base_docs(preflight__check_availability=False,
                           validation__allowed_scripts=['latin'])
        llm['processing'] = policy
        workspace.write_text(f'{name}.toml', dump_toml(llm))
        draft = f'draft = "{name}-draft.json"\n' if drafts else ''
        entries.append(f'[[targets]]\noriginal = "{name}.json"\n'
                       f'llm_config = "{name}.toml"\n{draft}')
    workspace.write_text('job.toml', '\n'.join(reversed(entries) if reverse else entries))


@pytest.mark.parametrize('reverse', [False, True])
def test_manifest_numeric_eligibility_uses_each_profile(workspace, http, run_main, reverse):
    _manifest_profiles(workspace, {'letters': {}, 'numbers': {'required_content_pattern': ''}},
                       reverse=reverse)
    for name in ('letters', 'numbers'):
        workspace.write_json(f'{name}.json', {'x': '42', 'blank': '\t '})
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('43'))
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('44'))
    assert run_main('run', '--target', 'job.toml') == EXIT_SUCCESS
    assert len(http.calls) == 2
    assert workspace.read_json(workspace.result('final', 'letters.json', 'job.toml')) == {
        'x': '42', 'blank': '\t ',
    }
    assert workspace.read_json(workspace.result('final', 'numbers.json', 'job.toml')) == {
        'x': '44', 'blank': '\t ',
    }


@pytest.mark.parametrize('reverse', [False, True])
def test_manifest_masks_original_and_disabled_markers_use_each_profile(workspace, model, run_main, reverse):
    _manifest_profiles(workspace, {'protected': {'mask_pattern': r'\{[a-z]+\}'},
                                   'literal': {'mask_pattern': ''}}, reverse=reverse)
    workspace.write_json('protected.json', {'x': 'Hello {name}'})
    workspace.write_json('literal.json', {'x': 'Hello [T0] <b>'})
    assert run_main('run', '--target', 'job.toml') == EXIT_SUCCESS
    assert len(model.model_calls) == 4
    for name, source in [('protected', 'Hello {name}'), ('literal', 'Hello [T0] <b>')]:
        assert workspace.read_json(workspace.result('final', f'{name}.json', 'job.toml')) == {
            'x': f'OK(RU({source}))',
        }
    prompts = [p['messages'][-1]['content'] for p in model.model_calls]
    assert any('[ORIGINAL]: Hello [T0]' in p and 'RU(Hello [T0])' in p for p in prompts)
    assert any('[ORIGINAL]: Hello [T0] <b>' in p for p in prompts)


def test_manifest_content_identity_separates_dedup_and_warm_cache(workspace, model, run_main):
    _manifest_profiles(workspace, {'one': {'required_content_pattern': 'Hello'},
                                   'two': {'required_content_pattern': r'\p{L}'}})
    for name in ('one', 'two'):
        workspace.write_json(f'{name}.json', {'x': 'Hello'})
    assert run_main('run', '--target', 'job.toml') == EXIT_SUCCESS
    assert len(model.model_calls) == 4
    assert run_main('run', '--target', 'job.toml') == EXIT_SUCCESS
    assert len(model.model_calls) == 4


@pytest.mark.parametrize('reverse', [False, True])
def test_manifest_rejects_fabricated_fragments_using_each_profile(workspace, http, run_main, reverse):
    workspace.write_configs(resilience__fail_on_any_error=True)
    _manifest_profiles(workspace, {'protected': {'mask_pattern': r'@@'},
                                   'literal': {'mask_pattern': ''}}, reverse=reverse)
    for name in ('protected', 'literal'):
        workspace.write_json(f'{name}.json', {'x': 'Hello'})
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply('Hi @@'))
    assert run_main('run', '--target', 'job.toml') == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(http.calls) == 3
    assert workspace.read_json(workspace.result('final', 'protected.json', 'job.toml')) == {'x': 'Hello'}
    assert workspace.read_json(workspace.result('final', 'literal.json', 'job.toml')) == {'x': 'Hi @@'}


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('damage_protected', [False, True])
def test_manifest_draft_preflight_uses_each_profile(
        workspace, model, run_main, reverse, dry_run, damage_protected):
    _manifest_profiles(workspace, {'protected': {'mask_pattern': r'\{[a-z]+\}'},
                                   'literal': {'mask_pattern': ''}}, reverse=reverse, drafts=True)
    for name in ('protected', 'literal'):
        workspace.write_json(f'{name}.json', {'x': 'Hello {name}'})
        fragment = '{other}' if name == 'literal' or damage_protected else '{name}'
        workspace.write_json(f'{name}-draft.json', {'x': f'Hi {fragment}'})
    args = ['run', '--target', 'job.toml'] + (['--dry-run'] if dry_run else [])
    expected = EXIT_USAGE_ERROR if damage_protected else EXIT_SUCCESS
    assert run_main(*args) == expected
    assert len(model.model_calls) == (2 if not dry_run and not damage_protected else 0)
