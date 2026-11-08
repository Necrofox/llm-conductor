# -*- coding: utf-8 -*-
"""
Live contract checks of the OpenAI transport against a real
OpenAI-compatible provider (proxyapi.ru, mistralai/mistral-nemo).

Run only on demand, with the key in the environment (it is never stored in
the repository):

    $env:LLMC_OPENAI_API_KEY = "<key>"; pytest -m live tests/live/test_live_openai.py

Without the variable every test is skipped. Requests are tiny (a few tokens
each) but billed, so the module is excluded from the default run by the
`live` marker like the rest of this tier.

What this adds over tests/integration/test_api_call.py: those pin our
handling of *simulated* provider answers; these confirm that the simulations
match what a real provider sends (response shape, usage counters, error
bodies, Authorization handling, parameter tolerance).
"""
import os

import pytest

from llm_conductor.cli.validators import check_api_availability
from llm_conductor.constants import (
    FAILURE_KIND_NETWORK,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.core.api_client import call_model_api
from llm_conductor.core.metrics import MetricsCollector
from tests.live.runner import CYRILLIC
from tests.support import RecordingLogger

KEY_ENV = "LLMC_OPENAI_API_KEY"
API_URL = os.environ.get("LLMC_OPENAI_API_URL", "https://api.proxyapi.ru/v1/chat/completions")
MODEL = os.environ.get("LLMC_OPENAI_MODEL", "mistralai/mistral-nemo")
# Model that rejects max_tokens with 400 (OpenAI reasoning / GPT-5 family via proxyapi).
REASONING_MODEL = os.environ.get("LLMC_OPENAI_REASONING_MODEL", "openai/gpt-5-nano")

SYSTEM_PROMPT = "You are a professional EN->RU game translator. Output ONLY the translation."
PROMPT = "Translate to Russian.\nText to translate:\n"
SOURCE = "You found [T0] gold coins!"


class _Config:
    """Minimal ProcessorConfigProtocol: logger, metrics and debug flag."""

    is_debug_mode = False

    def __init__(self):
        self.logger = RecordingLogger()
        self.metrics = MetricsCollector()


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get(KEY_ENV)
    if not key:
        pytest.skip(f"{KEY_ENV} is not set: live OpenAI checks need a real key")
    return key


@pytest.fixture
def runtime(api_key):
    return {
        'api_type': 'openai', 'api_key': api_key, 'api_url': API_URL, 'model': MODEL,
        'system_prompt': SYSTEM_PROMPT, 'prompt': PROMPT,
        'temp': 0.2, 'top_p': 0.9, 'top_k': 40, 'num_predict': 200, 'num_ctx': 2048,
        'repeat_penalty': 1.1, 'frequency_penalty': 0.0, 'presence_penalty': 0.0,
        'stop': ["\n\n", "<|im_end|>", "<|im_start|>"],
        'api_timeout': 60, 'max_retries': 2, 'retry_backoff_factor': 2,
        'max_tokens_param': 'max_tokens',
    }


@pytest.fixture
def reasoning_runtime(api_key):
    """
    Minimal runtime for models that reject max_tokens and most sampling knobs.

    num_predict must leave room for hidden reasoning tokens; a tight cap
    yields finish_reason=length with an empty visible reply.
    """
    return {
        'api_type': 'openai', 'api_key': api_key, 'api_url': API_URL,
        'model': REASONING_MODEL,
        'system_prompt': '', 'prompt': '',
        'temp': None, 'top_p': None, 'top_k': None, 'num_predict': 1024, 'num_ctx': None,
        'repeat_penalty': None, 'frequency_penalty': None, 'presence_penalty': None,
        'stop': [],
        'api_timeout': 120, 'max_retries': 1, 'retry_backoff_factor': 2,
        'max_tokens_param': 'max_completion_tokens',
    }


def _metrics(config, operation=OPERATION_TRANSLATION):
    return config.metrics.get_snapshot()[operation]


def test_preflight_accepts_valid_key(api_key):
    config = _Config()

    assert check_api_availability(API_URL, MODEL, 'openai', config, api_key=api_key) is True


def test_translation_roundtrip_counts_tokens_and_keeps_tag_marker(runtime):
    config = _Config()

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config, runtime)

    assert outcome.success, outcome
    assert CYRILLIC.search(outcome.text), outcome.text
    assert "[T0]" in outcome.text, outcome.text
    metrics = _metrics(config)
    assert (metrics['total_requests'], metrics['failed_requests']) == (1, 0)
    assert metrics['prompt_tokens'] > 0 and metrics['completion_tokens'] > 0


def test_verification_returns_a_successful_outcome(runtime):
    config = _Config()

    outcome = call_model_api("Вы нашли [T0] золотых монет!", OPERATION_VERIFICATION, config, runtime)

    assert outcome.success is True and outcome.failure_kind is None
    assert CYRILLIC.search(outcome.text), outcome.text
    assert _metrics(config, OPERATION_VERIFICATION)['failed_requests'] == 0


def test_key_with_surrounding_whitespace_is_accepted(runtime):
    config = _Config()

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config,
                             {**runtime, 'api_key': f"  {runtime['api_key']}\n"})

    assert outcome.success and outcome.text != SOURCE
    assert _metrics(config)['failed_requests'] == 0


def test_unset_sampling_parameters_are_accepted(runtime):
    config = _Config()
    unset = {**runtime, 'temp': None, 'top_p': None, 'num_predict': None,
             'frequency_penalty': None, 'presence_penalty': None, 'stop': None}

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config, unset)

    assert outcome.success and outcome.text != SOURCE
    assert _metrics(config)['failed_requests'] == 0


def test_num_predict_limits_completion_tokens(runtime):
    config = _Config()

    call_model_api("Write a long story about a dragon.", OPERATION_TRANSLATION, config,
                   {**runtime, 'num_predict': 5, 'stop': []})

    assert _metrics(config)['completion_tokens'] <= 5


def test_reasoning_model_rejects_default_max_tokens_param(reasoning_runtime):
    """gpt-5-nano / o-series: max_tokens is a fatal 400; max_retries cannot help."""
    config = _Config()
    runtime = {**reasoning_runtime, 'max_tokens_param': 'max_tokens', 'num_predict': 16}

    outcome = call_model_api("Hi", OPERATION_TRANSLATION, config, runtime)

    assert outcome == ("Hi", False, FAILURE_KIND_NETWORK)
    assert _metrics(config)['network_failures'] == 1
    assert config.logger.contains("Unsupported parameter", level='WARNING')
    assert config.logger.contains("max_tokens", level='WARNING')
    assert config.logger.contains("Failed after 1 attempt(s)", level='WARNING')


def test_reasoning_model_accepts_max_completion_tokens(reasoning_runtime):
    config = _Config()

    outcome = call_model_api(
        "Reply with exactly the word OK.", OPERATION_TRANSLATION, config, reasoning_runtime)

    assert outcome.success, outcome
    assert "OK" in outcome.text
    assert _metrics(config)['failed_requests'] == 0
    assert _metrics(config)['completion_tokens'] > 0


def test_preflight_reasoning_model_needs_max_completion_tokens(api_key):
    config = _Config()

    assert check_api_availability(
        API_URL, REASONING_MODEL, 'openai', config, api_key=api_key,
        max_tokens_param='max_tokens') is False
    assert config.logger.contains("Unsupported parameter", level='ERROR')

    config_ok = _Config()
    assert check_api_availability(
        API_URL, REASONING_MODEL, 'openai', config_ok, api_key=api_key,
        max_tokens_param='max_completion_tokens') is True


def test_classic_model_accepts_both_token_limit_field_names(runtime):
    """Control: mistral / gpt-4o-mini still work with either field name."""
    for field_name in ('max_tokens', 'max_completion_tokens'):
        config = _Config()
        outcome = call_model_api(
            SOURCE, OPERATION_TRANSLATION, config,
            {**runtime, 'max_tokens_param': field_name, 'num_predict': 64})
        assert outcome.success, (field_name, outcome)
        assert _metrics(config)['failed_requests'] == 0


def test_rejected_key_is_reported_with_provider_text_and_not_retried(runtime):
    config = _Config()
    secret = "sk-definitely-wrong-key"

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config, {**runtime, 'api_key': secret})

    assert outcome == (SOURCE, False, FAILURE_KIND_NETWORK)
    metrics = _metrics(config)
    assert (metrics['failed_requests'], metrics['network_failures']) == (1, 1)
    assert config.logger.contains("non-retryable status 401", level='WARNING')
    assert config.logger.contains("Failed after 1 attempt(s)", level='WARNING')
    assert not config.logger.contains("non-JSON")
    assert not config.logger.contains(secret)


def test_unknown_model_is_a_fatal_client_error(runtime):
    config = _Config()

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config,
                             {**runtime, 'model': 'mistralai/no-such-model'})

    assert outcome == (SOURCE, False, FAILURE_KIND_NETWORK)
    assert config.logger.contains("Failed after 1 attempt(s)", level='WARNING')


def test_key_with_embedded_newline_is_rejected_locally_without_leaking(runtime):
    config = _Config()
    key = runtime['api_key']
    broken = key[:8] + "\n" + key[8:]

    outcome = call_model_api(SOURCE, OPERATION_TRANSLATION, config, {**runtime, 'api_key': broken})

    assert outcome == (SOURCE, False, FAILURE_KIND_NETWORK)
    assert config.logger.contains("invalid HTTP header", level='WARNING')
    assert not config.logger.contains(key[8:])
