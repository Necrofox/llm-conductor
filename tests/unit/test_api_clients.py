# -*- coding: utf-8 -*-
"""
Unit tests for API client payload building and response parsing
(llm_conductor/core/api_client.py, doc/pipeline.md). No network — only the pure
build_*/parse_response methods and the create_api_client factory.
"""
import pytest

from llm_conductor.constants import (
    DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS,
    MAX_RETRY_AFTER_SECONDS,
)
from llm_conductor.core.api_client import (
    OllamaAPIClient,
    OpenAIAPIClient,
    _parse_retry_after,
    create_api_client,
)

SAMPLING = {
    'temperature': 0.3,
    'top_p': 0.9,
    'top_k': 40,
    'num_predict': 150,
    'num_ctx': 2048,
    'repeat_penalty': 1.1,
    'frequency_penalty': 0.1,
    'presence_penalty': 0.2,
    'stop': ["\n\n"],
}


# ============================================================================
# Ollama
# ============================================================================

def test_ollama_payload_has_system_and_user_messages_and_options():
    payload = OllamaAPIClient().build_request_payload("qwen", "Translate: hi", "Be precise", SAMPLING)

    assert payload == {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": "Translate: hi"},
        ],
        "stream": False,
        "options": SAMPLING,
    }


@pytest.mark.parametrize("system_prompt", ["", "   "])
def test_ollama_payload_omits_blank_system_prompt(system_prompt):
    payload = OllamaAPIClient().build_request_payload("qwen", "hi", system_prompt, {})

    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_ollama_parse_response_reads_message_and_token_counters():
    response = {"message": {"content": "Привет"}, "prompt_eval_count": 12, "eval_count": 3}

    assert OllamaAPIClient().parse_response(response) == ("Привет", 12, 3, None, False, False)


def test_ollama_parse_response_defaults_missing_or_invalid_counters_to_zero():
    response = {"message": {"content": "Привет"}, "prompt_eval_count": "12"}

    # Invalid prompt_eval_count coerced to 0; eval_count absent → usage still
    # considered present because one usage key was sent.
    assert OllamaAPIClient().parse_response(response) == ("Привет", 0, 0, None, False, False)


def test_ollama_parse_response_without_message_returns_empty_text():
    assert OllamaAPIClient().parse_response({}) == ("", 0, 0, None, False, True)


def test_ollama_parse_response_reads_done_reason():
    response = {"message": {"content": "Привет"}, "done_reason": "length"}

    assert OllamaAPIClient().parse_response(response).finish_reason == "length"


def test_ollama_parse_response_has_no_refusal_concept():
    response = {"message": {"content": "Привет"}}

    assert OllamaAPIClient().parse_response(response).refusal is False


def test_ollama_parse_response_concatenates_text_content_parts():
    response = {"message": {"content": [
        {"type": "text", "text": "Hello "},
        {"type": "other", "text": "ignored"},
        {"type": "text", "text": "world"},
    ]}}

    assert OllamaAPIClient().parse_response(response).text == "Hello world"


def test_ollama_test_payload_requests_single_token():
    payload = OllamaAPIClient().build_test_payload("qwen")

    assert payload["model"] == "qwen"
    assert payload["options"] == {"num_predict": 1}
    assert payload["stream"] is False


# ============================================================================
# OpenAI-compatible
# ============================================================================

def test_openai_payload_maps_num_predict_and_drops_ollama_only_params():
    payload = OpenAIAPIClient().build_request_payload("gpt", "Translate: hi", "Be precise", SAMPLING)

    assert payload == {
        "model": "gpt",
        "messages": [
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": "Translate: hi"},
        ],
        "stream": False,
        "max_tokens": 150,
        "temperature": 0.3,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "stop": ["\n\n"],
    }
    for ollama_only in ("top_k", "repeat_penalty", "num_ctx", "num_predict", "options"):
        assert ollama_only not in payload


def test_openai_payload_omits_empty_stop_and_system_prompt():
    payload = OpenAIAPIClient().build_request_payload("gpt", "hi", "", {**SAMPLING, 'stop': []})

    assert "stop" not in payload
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_openai_parse_response_reads_first_choice_and_usage():
    response = {
        "choices": [{"message": {"content": "Привет"}}, {"message": {"content": "ignored"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }

    assert OpenAIAPIClient().parse_response(response) == ("Привет", 10, 2, None, False, False)


def test_openai_parse_response_without_usage_counts_zero_tokens():
    response = {"choices": [{"message": {"content": "Привет"}}]}

    assert OpenAIAPIClient().parse_response(response) == ("Привет", 0, 0, None, False, True)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "stop", "tool_calls"])
def test_openai_parse_response_reads_finish_reason(finish_reason):
    response = {"choices": [{"message": {"content": "x"}, "finish_reason": finish_reason}]}

    assert OpenAIAPIClient().parse_response(response).finish_reason == finish_reason


def test_openai_parse_response_reads_refusal():
    response = {"choices": [{"message": {"content": None, "refusal": "I can't help with that."}}]}

    parsed = OpenAIAPIClient().parse_response(response)
    assert parsed.refusal is True
    assert parsed.text == ""


@pytest.mark.parametrize("refusal", [None, "", False])
def test_openai_parse_response_blank_refusal_field_is_not_a_refusal(refusal):
    response = {"choices": [{"message": {"content": "ok", "refusal": refusal}}]}

    assert OpenAIAPIClient().parse_response(response).refusal is False


def test_openai_parse_response_concatenates_text_content_parts():
    response = {"choices": [{"message": {"content": [
        {"type": "text", "text": "Hello "},
        {"type": "refusal", "refusal": "nope"},
        {"type": "text", "text": "world"},
    ]}}]}

    assert OpenAIAPIClient().parse_response(response).text == "Hello world"


@pytest.mark.parametrize("response", [{}, {"choices": []}])
def test_openai_parse_response_rejects_missing_choices(response):
    with pytest.raises(ValueError, match="Invalid OpenAI response format"):
        OpenAIAPIClient().parse_response(response)


def test_openai_test_payload_requests_preflight_token_budget():
    payload = OpenAIAPIClient().build_test_payload("gpt")

    assert payload["max_tokens"] == DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS
    assert payload["messages"] == [{"role": "user", "content": "test"}]


def test_openai_payload_uses_max_completion_tokens_when_configured():
    payload = OpenAIAPIClient().build_request_payload(
        "gpt", "hi", "", {**SAMPLING, 'max_tokens_param': 'max_completion_tokens'})

    assert payload["max_completion_tokens"] == 150
    assert "max_tokens" not in payload


def test_openai_test_payload_honours_max_tokens_param():
    payload = OpenAIAPIClient().build_test_payload(
        "gpt", max_tokens_param="max_completion_tokens")

    assert payload["max_completion_tokens"] == DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS
    assert "max_tokens" not in payload


def test_ollama_test_payload_ignores_max_tokens_param():
    payload = OllamaAPIClient().build_test_payload(
        "qwen", max_tokens_param="max_completion_tokens")

    assert "max_completion_tokens" not in payload
    assert "max_tokens" not in payload
    assert payload["options"] == {"num_predict": 1}


# ============================================================================
# Factory
# ============================================================================

@pytest.mark.parametrize("api_type, expected_class", [
    ("ollama", OllamaAPIClient),
    ("openai", OpenAIAPIClient),
])
def test_create_api_client_returns_cached_instance_of_right_type(api_type, expected_class):
    client = create_api_client(api_type)

    assert isinstance(client, expected_class)
    assert create_api_client(api_type) is client


def test_create_api_client_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown API type"):
        create_api_client("anthropic-native")


# ============================================================================
# Hardening against malformed provider responses
# ============================================================================

@pytest.mark.parametrize("response", [
    {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": "text"}]},
    {"choices": "abc"},
    {"choices": {"0": 1}},
])
def test_openai_parse_response_rejects_malformed_choices_with_value_error(response):
    with pytest.raises(ValueError, match="Invalid OpenAI response format"):
        OpenAIAPIClient().parse_response(response)


def test_openai_parse_response_tolerates_null_usage():
    response = {"choices": [{"message": {"content": "ok"}}], "usage": None}

    assert OpenAIAPIClient().parse_response(response) == ("ok", 0, 0, None, False, True)


@pytest.mark.parametrize("content", [None, 42, [{"type": "refusal", "refusal": "no"}]])
def test_openai_parse_response_non_text_content_becomes_empty_text(content):
    response = {"choices": [{"message": {"content": content}}]}

    assert OpenAIAPIClient().parse_response(response).text == ""


@pytest.mark.parametrize("response", [{"message": None}, {"message": "text"}, [], "text"])
def test_ollama_parse_response_rejects_malformed_message_with_value_error(response):
    with pytest.raises(ValueError, match="Invalid Ollama response format"):
        OllamaAPIClient().parse_response(response)


def test_ollama_parse_response_null_content_becomes_empty_text():
    assert OllamaAPIClient().parse_response({"message": {"content": None}}) == (
        "", 0, 0, None, False, True,
    )


NONE_SAMPLING = {
    'temperature': None, 'top_p': None, 'top_k': None, 'num_predict': None,
    'num_ctx': None, 'repeat_penalty': None, 'frequency_penalty': None,
    'presence_penalty': None, 'stop': None,
}


def test_openai_payload_omits_params_that_are_none():
    payload = OpenAIAPIClient().build_request_payload("gpt", "hi", "", NONE_SAMPLING)

    assert payload == {"model": "gpt", "messages": [{"role": "user", "content": "hi"}], "stream": False}


def test_ollama_payload_options_omit_params_that_are_none():
    payload = OllamaAPIClient().build_request_payload("qwen", "hi", "", {**NONE_SAMPLING, 'temperature': 0.3})

    assert payload["options"] == {'temperature': 0.3}


def test_ollama_payload_includes_keep_alive_at_top_level_not_in_options():
    payload = OllamaAPIClient().build_request_payload(
        "qwen", "hi", "", {**SAMPLING, 'keep_alive': '5m'})

    assert payload['keep_alive'] == '5m'
    assert 'keep_alive' not in payload['options']
    assert payload['options'] == SAMPLING


def test_ollama_payload_omits_blank_keep_alive():
    for blank in ('', '   ', None):
        payload = OllamaAPIClient().build_request_payload(
            "qwen", "hi", "", {**SAMPLING, 'keep_alive': blank})

        assert 'keep_alive' not in payload
        assert 'keep_alive' not in payload['options']


def test_openai_payload_silently_drops_keep_alive():
    payload = OpenAIAPIClient().build_request_payload(
        "gpt", "hi", "", {**SAMPLING, 'keep_alive': '5m'})

    assert 'keep_alive' not in payload
    assert 'top_k' not in payload


def test_openai_payload_defaults_max_tokens_param_when_absent():
    """sampling_params without the key still emit max_tokens (compat default)."""
    payload = OpenAIAPIClient().build_request_payload("gpt", "hi", "", SAMPLING)

    assert payload["max_tokens"] == 150
    assert "max_completion_tokens" not in payload


@pytest.mark.parametrize("client", [OllamaAPIClient(), OpenAIAPIClient()])
@pytest.mark.parametrize("body", [[{"error": "x"}], "oops", 42, None])
def test_parse_error_detail_ignores_non_object_json(client, body):
    assert client.parse_error_detail(body) is None


def test_openai_parse_error_detail_reads_fastapi_detail_string():
    assert OpenAIAPIClient().parse_error_detail({"detail": "Invalid API Key"}) == "Invalid API Key"


def test_openai_parse_error_detail_joins_fastapi_validation_messages():
    body = {"detail": [{"loc": ["body", "model"], "msg": "field required"},
                       {"loc": ["body", "messages"], "msg": "too short"}]}

    assert OpenAIAPIClient().parse_error_detail(body) == "field required; too short"


@pytest.mark.parametrize("header, expected", [
    ("7", 7.0),
    ("0", 0.0),
    ("1e9", float(MAX_RETRY_AFTER_SECONDS)),
    ("-5", 0.0),
    ("inf", None),
    ("nan", None),
    ("Wed, 21 Oct 2026 07:28:00 GMT", None),
    ("", None),
    (None, None),
])
def test_parse_retry_after_is_finite_and_capped(header, expected):
    assert _parse_retry_after(header) == expected
