# -*- coding: utf-8 -*-
"""
Integration tests for the HTTP layer: call_model_api / make_request_with_retry /
check_api_availability on top of responses (no real network).

Failure policy §4.3: call_model_api() never raises; it returns a CallOutcome
with success=False and text = the original value — so every degradation
branch is checked both on the result and on the metrics.
"""
import json

import pytest
import requests

from llm_conductor.cli.validators import check_api_availability
from llm_conductor.constants import (
    DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS,
    MAX_RETRY_AFTER_SECONDS,
    MODE_STYLE,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.core.api_client import (
    CallOutcome,
    call_model_api,
    translate_key,
    verify_key,
)
from tests.support import OLLAMA_URL, OPENAI_URL, ollama_reply, openai_reply


def _runtime(config, **overrides):
    runtime = config.get_translation_config_for_mode(MODE_STYLE)
    runtime.update(overrides)
    return runtime


def _metrics(config, operation=OPERATION_TRANSLATION):
    return config.metrics.get_snapshot()[operation]


def _sent_json(http, index=0):
    return json.loads(http.calls[index].request.body)


def _openai_config(make_config, **overrides):
    return make_config(translation__api__type="openai", translation__api__url=OPENAI_URL,
                       translation__api__key="sk-test", **overrides)


# ============================================================================
# Successful requests
# ============================================================================

def test_ollama_translation_roundtrip(make_config, http):
    config = make_config(translation__prompts__system="SYS")
    http.post(OLLAMA_URL, json=ollama_reply("  Привет, мир  ", 11, 4))

    result = translate_key("Hello, world", config, _runtime(config))

    assert result.text == "Привет, мир"                  # model reply is strip()'d
    assert result.success is True
    sent = _sent_json(http)
    assert sent["model"] == "test-model"
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    # prompt and value are concatenated without a separator (§9.1)
    assert sent["messages"][1]["content"] == "Translate the text: Hello, world"
    assert "Authorization" not in http.calls[0].request.headers
    metrics = _metrics(config)
    assert (metrics['total_requests'], metrics['failed_requests']) == (1, 0)
    assert (metrics['prompt_tokens'], metrics['completion_tokens']) == (11, 4)


def test_openai_request_carries_bearer_key(make_config, http):
    config = make_config(translation__api__type="openai", translation__api__url=OPENAI_URL,
                         translation__api__key="sk-test")
    http.post(OPENAI_URL, json=openai_reply("Привет"))

    result = translate_key("Hello", config, _runtime(config))

    assert result.text == "Привет"
    assert http.calls[0].request.headers["Authorization"] == "Bearer sk-test"
    assert "max_tokens" in _sent_json(http)


def test_openai_request_uses_max_completion_tokens_from_api_section(make_config, http):
    config = _openai_config(make_config, translation__api__max_tokens_param="max_completion_tokens")
    http.post(OPENAI_URL, json=openai_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config)).success is True
    body = _sent_json(http)
    assert body["max_completion_tokens"] == config.get_translation_config_for_mode(MODE_STYLE)["num_predict"]
    assert "max_tokens" not in body


def test_ollama_request_sends_keep_alive_at_top_level(make_config, http):
    config = make_config(translation__api__keep_alive="5m")
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config)).success is True
    body = _sent_json(http)
    assert body["keep_alive"] == "5m"
    assert "keep_alive" not in body["options"]


def test_ollama_request_omits_blank_keep_alive(make_config, http):
    config = make_config(translation__api__keep_alive="")
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config)).success is True
    body = _sent_json(http)
    assert "keep_alive" not in body
    assert "keep_alive" not in body["options"]


def test_verification_returns_call_outcome(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, json=ollama_reply("Проверено"))

    assert verify_key("Проверить", config, config.get_verification_config()) == CallOutcome(
        "Проверено", True, None
    )
    assert _metrics(config, OPERATION_VERIFICATION)['total_requests'] == 1


@pytest.mark.parametrize("operation", [OPERATION_TRANSLATION, OPERATION_VERIFICATION])
def test_blank_value_makes_no_request(make_config, http, operation):
    config = make_config()

    outcome = call_model_api("   ", operation, config, _runtime(config))

    assert outcome == CallOutcome("   ", True, None)
    assert len(http.calls) == 0


@pytest.mark.characterization(
    reason="unknown runtime api_type is swapped for ollama instead of raising")
def test_unknown_api_type_falls_back_to_ollama(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config, api_type="grpc")).text == "Привет"
    assert config.logger.contains("Falling back to 'ollama'", level='WARNING')


# ============================================================================
# Retries and exhausted attempts
# ============================================================================

def test_retry_with_exponential_backoff_then_success(make_config, http, no_sleep):
    config = make_config(translation__api__max_retries=3, translation__api__retry_backoff_factor=2)
    http.post(OLLAMA_URL, status=500)
    http.post(OLLAMA_URL, status=503)
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config)).text == "Привет"
    assert len(http.calls) == 3
    assert no_sleep == [1, 2]                         # backoff_factor ** attempt


def test_exhausted_retries_degrade_to_original_and_count_failure(make_config, http, no_sleep):
    config = make_config(translation__api__max_retries=3)
    http.post(OLLAMA_URL, status=500)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert len(http.calls) == 3
    assert len(no_sleep) == 2                         # no wait after the last attempt
    metrics = _metrics(config)
    assert (metrics['total_requests'], metrics['failed_requests']) == (1, 1)
    assert config.logger.contains("API request failed", level='WARNING')


def test_connection_error_is_retried_and_degrades(make_config, http, no_sleep):
    """URL not registered in responses — ConnectionError on every attempt."""
    config = make_config(translation__api__url="http://unreachable.test/api/chat",
                         translation__api__max_retries=2)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert _metrics(config)['failed_requests'] == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_task25_client_errors_are_not_retried(make_config, http, no_sleep, status):
    config = make_config(translation__api__max_retries=3)
    http.post(OLLAMA_URL, status=status)

    translate_key("Hello", config, _runtime(config))

    assert len(http.calls) == 1
    assert no_sleep == []


# ============================================================================
# Invalid replies
# ============================================================================

@pytest.mark.parametrize("reply", [
    ollama_reply(""),                    # empty text
    {"unexpected": "shape"},             # no message — empty text
])
def test_empty_model_answer_degrades_to_original(make_config, http, reply):
    config = make_config()
    http.post(OLLAMA_URL, json=reply)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert config.logger.contains("API returned invalid response", level='WARNING')


def test_non_json_body_degrades_to_original(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, body="<html>502 Bad Gateway</html>", content_type="text/html")

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert config.logger.contains("Failed to parse response", level='ERROR')


def test_openai_reply_without_choices_degrades_to_original(make_config, http):
    config = make_config(translation__api__type="openai", translation__api__url=OPENAI_URL,
                         translation__api__key="sk-test")
    http.post(OPENAI_URL, json={"error": {"message": "model not found"}})

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"


def test_task2_parse_failure_counts_as_failed_request(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, body="<html>502</html>", content_type="text/html")

    translate_key("Hello", config, _runtime(config))

    assert _metrics(config)['failed_requests'] == 1


def test_task2_empty_answer_counts_as_failed_request(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, json=ollama_reply(""))

    translate_key("Hello", config, _runtime(config))

    assert _metrics(config)['failed_requests'] == 1


def test_task2_incomplete_runtime_config_degrades_instead_of_raising(make_config, http):
    """A caller bug that drops a runtime_config key degrades like any other
    failure instead of raising KeyError - no route is registered here, so
    the request itself still fails and falls back to the original value."""
    config = make_config()
    runtime = _runtime(config)
    del runtime['temp']

    assert translate_key("Hello", config, runtime).text == "Hello"


# ============================================================================
# Provider reports the problem itself: truncation and model refusal
# ============================================================================

def test_ollama_length_done_reason_degrades_and_is_not_cached_as_success(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, json={**ollama_reply("Обрез"), "done_reason": "length"})

    outcome = translate_key("Hello", config, _runtime(config))

    assert outcome == CallOutcome("Hello", False, "truncated")
    assert _metrics(config)['truncated_responses'] == 1
    assert _metrics(config)['failed_requests'] == 1


def test_openai_length_finish_reason_degrades(make_config, http):
    config = _openai_config(make_config)
    http.post(OPENAI_URL, json={
        "choices": [{"message": {"content": "Обрез"}, "finish_reason": "length"}],
    })

    outcome = translate_key("Hello", config, _runtime(config))

    assert outcome == CallOutcome("Hello", False, "truncated")
    assert _metrics(config)['truncated_responses'] == 1


def test_openai_refusal_degrades_and_is_not_truncated(make_config, http):
    config = _openai_config(make_config)
    http.post(OPENAI_URL, json={
        "choices": [{"message": {"content": None, "refusal": "I can't help with that."}}],
    })

    outcome = translate_key("Hello", config, _runtime(config))

    assert outcome == CallOutcome("Hello", False, "refusal")
    assert _metrics(config)['refusal_responses'] == 1


# ============================================================================
# check_api_availability (preflight)
# ============================================================================

def test_preflight_ok_sends_minimal_test_payload(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, json=ollama_reply("ok"))

    assert check_api_availability(OLLAMA_URL, "test-model", "ollama", config) is True
    assert _sent_json(http)["options"] == {"num_predict": 1}


def test_preflight_openai_sends_bearer(make_config, http):
    config = make_config()
    http.post(OPENAI_URL, json=openai_reply("ok"))

    assert check_api_availability(OPENAI_URL, "gpt", "openai", config, api_key="sk-1") is True
    assert http.calls[0].request.headers["Authorization"] == "Bearer sk-1"


def test_preflight_openai_honours_max_tokens_param(make_config, http):
    config = make_config()
    http.post(OPENAI_URL, json=openai_reply("ok"))

    assert check_api_availability(
        OPENAI_URL, "gpt", "openai", config, api_key="sk-1",
        max_tokens_param="max_completion_tokens") is True
    body = _sent_json(http)
    assert body["max_completion_tokens"] == DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS
    assert "max_tokens" not in body


@pytest.mark.parametrize("status", [401, 404, 500])
def test_preflight_non_200_is_unavailable(make_config, http, status):
    config = make_config()
    http.post(OLLAMA_URL, status=status)

    assert check_api_availability(OLLAMA_URL, "test-model", "ollama", config) is False
    assert config.logger.contains(f"returned status code {status}", level='ERROR')


@pytest.mark.parametrize("error, log_fragment", [
    (requests.Timeout(), "timed out"),
    (requests.ConnectionError(), "Failed to connect"),
])
def test_preflight_network_errors_are_unavailable(make_config, http, error, log_fragment):
    config = make_config()
    http.post(OLLAMA_URL, body=error)

    assert check_api_availability(OLLAMA_URL, "test-model", "ollama", config) is False
    assert config.logger.contains(log_fragment, level='ERROR')


def test_preflight_unknown_api_type_is_unavailable(make_config, http):
    config = make_config()

    assert check_api_availability(OLLAMA_URL, "test-model", "grpc", config) is False
    assert len(http.calls) == 0


# ============================================================================
# Hardening of the transport against hostile or odd provider behaviour
# ============================================================================

def test_max_retries_zero_still_makes_one_attempt(make_config, http, no_sleep):
    config = make_config()
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config, max_retries=0)).text == "Привет"
    assert len(http.calls) == 1


def test_max_retries_zero_failure_degrades_instead_of_raising(make_config, http, no_sleep):
    config = make_config()
    http.post(OLLAMA_URL, status=500)

    assert translate_key("Hello", config, _runtime(config, max_retries=0)).text == "Hello"
    assert len(http.calls) == 1


def test_api_key_surrounding_whitespace_is_stripped_from_header(make_config, http):
    config = make_config()
    http.post(OPENAI_URL, json=openai_reply("Привет"))
    runtime = _runtime(config, api_type="openai", api_url=OPENAI_URL, api_key="  sk-test\n")

    assert translate_key("Hello", config, runtime).text == "Привет"
    assert http.calls[0].request.headers["Authorization"] == "Bearer sk-test"


def test_invalid_header_value_is_not_retried_and_never_logged(make_config, http, no_sleep):
    """A key with an embedded newline cannot be sent (stripping only fixes the
    edges); the failure is permanent and the secret must not reach the log
    through the exception text."""
    config = make_config()
    secret = "sk-SECRET\nTAIL"
    runtime = _runtime(config, api_type="openai", api_url=OPENAI_URL, api_key=secret)

    assert translate_key("Hello", config, runtime).text == "Hello"
    assert no_sleep == []
    assert len(http.calls) == 0
    assert not config.logger.contains("SECRET")
    assert config.logger.contains("invalid HTTP header", level='WARNING')


@pytest.mark.parametrize("status", [402, 413])
def test_non_recoverable_statuses_are_not_retried(make_config, http, no_sleep, status):
    config = make_config(translation__api__max_retries=3)
    http.post(OLLAMA_URL, status=status)

    translate_key("Hello", config, _runtime(config))

    assert len(http.calls) == 1
    assert no_sleep == []


@pytest.mark.parametrize("status", [408, 409, 425])
def test_transient_looking_statuses_are_still_retried(make_config, http, no_sleep, status):
    config = make_config(translation__api__max_retries=3)
    http.post(OLLAMA_URL, status=status)

    translate_key("Hello", config, _runtime(config))

    assert len(http.calls) == 3


def test_retry_after_is_capped(make_config, http, no_sleep):
    config = make_config(translation__api__max_retries=2)
    http.post(OLLAMA_URL, status=429, headers={"Retry-After": "999999"})
    http.post(OLLAMA_URL, json=ollama_reply("Привет"))

    assert translate_key("Hello", config, _runtime(config)).text == "Привет"
    assert no_sleep == [MAX_RETRY_AFTER_SECONDS]


@pytest.mark.parametrize("body", [
    {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": None}}]},
    [1, 2, 3],
    "just a string",
])
def test_malformed_openai_body_degrades_instead_of_raising(make_config, http, body):
    config = _openai_config(make_config)
    http.post(OPENAI_URL, json=body)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert _metrics(config)['failed_requests'] == 1


def test_openai_null_usage_is_a_successful_answer_with_unknown_tokens(make_config, http):
    config = _openai_config(make_config)
    http.post(OPENAI_URL, json={"choices": [{"message": {"content": "Привет"}}], "usage": None})

    assert translate_key("Hello", config, _runtime(config)).text == "Привет"
    assert _metrics(config)['failed_requests'] == 0


@pytest.mark.parametrize("body", [{"message": None}, {"message": "text"}, [1]])
def test_malformed_ollama_body_degrades_instead_of_raising(make_config, http, body):
    config = make_config()
    http.post(OLLAMA_URL, json=body)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert _metrics(config)['failed_requests'] == 1


@pytest.mark.parametrize("body", [[{"error": "x"}], "oops", 42])
def test_non_object_json_error_body_degrades_instead_of_raising(make_config, http, no_sleep, body):
    config = make_config(translation__api__max_retries=2)
    http.post(OLLAMA_URL, status=500, json=body)

    assert translate_key("Hello", config, _runtime(config)).text == "Hello"
    assert len(http.calls) == 2


def test_fastapi_style_error_detail_is_reported_without_non_json_label(make_config, http):
    config = _openai_config(make_config)
    http.post(OPENAI_URL, status=401, json={"detail": "Invalid API Key"})

    translate_key("Hello", config, _runtime(config))

    assert config.logger.contains("Invalid API Key", level='WARNING')
    assert not config.logger.contains("non-JSON", level='WARNING')


def test_unrecognised_json_error_body_is_not_labelled_non_json(make_config, http):
    config = make_config()
    http.post(OLLAMA_URL, status=400, json={"unexpected": "shape"})

    translate_key("Hello", config, _runtime(config))

    assert config.logger.contains("unexpected", level='WARNING')
    assert not config.logger.contains("non-JSON", level='WARNING')


@pytest.mark.parametrize("bad_key", [123, ["sk"], {"k": 1}])
def test_non_string_api_key_degrades_instead_of_raising(make_config, http, bad_key):
    config = make_config()
    http.post(OPENAI_URL, json=openai_reply("Привет"))
    runtime = _runtime(config, api_type="openai", api_url=OPENAI_URL, api_key=bad_key)

    assert translate_key("Hello", config, runtime).text in ("Hello", "Привет")
    for call in http.calls:
        assert "Authorization" not in call.request.headers
