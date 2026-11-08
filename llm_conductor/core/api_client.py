# -*- coding: utf-8 -*-
import json
import math
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional

import requests

from llm_conductor.constants import (
    DEFAULT_API_BACKOFF_FACTOR,
    DEFAULT_API_MAX_RETRIES,
    DEFAULT_API_REQUEST_TIMEOUT,
    DEFAULT_MAX_TOKENS_PARAM,
    DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS,
    FAILURE_KIND_NETWORK,
    FAILURE_KIND_PARSE,
    FAILURE_KIND_REFUSAL,
    FAILURE_KIND_TRUNCATED,
    FINISH_REASON_LENGTH,
    HTTP_ERROR_BODY_PREVIEW_LENGTH,
    HTTP_TOO_MANY_REQUESTS,
    MAX_RETRY_AFTER_SECONDS,
    NON_RETRYABLE_HTTP_STATUS_CODES,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.exceptions import APIRequestError

if TYPE_CHECKING:
    from llm_conductor.interfaces import ProcessorConfigProtocol


class ParsedReply(NamedTuple):
    """
    A successful API response, decoded by one provider's parse_response().

    finish_reason and refusal carry signals the provider reports about
    itself - a cut-off reply or a declined request - that call_model_api()
    classifies as failures even though the HTTP call succeeded.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Optional[str]
    refusal: bool


class CallOutcome(NamedTuple):
    """
    Result of one call_model_api() call, for translation and verification alike.

    success is False on any failure (network, parse, truncation, refusal);
    text then carries the original, unmodified value. The caller
    (RecursiveProcessor._process_leaf) never sees a raw bool/str asymmetry
    between the two operation types.
    """

    text: str
    success: bool
    failure_kind: Optional[str]


def _join_text_parts(content: Any) -> str:
    """
    Normalize a provider's message content into plain text.

    Args:
        content: The raw "content" field of a reply message - a string, a
            list of content parts ({"type": "text", "text": ...} among
            others, e.g. multimodal providers), or anything else.

    Returns:
        The content itself if it is already a string; the concatenation of
        every part whose "type" is "text" if it is a list; '' otherwise.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            part.get('text', '') for part in content
            if isinstance(part, dict) and part.get('type') == 'text'
        )
    return ''


class APIClient(ABC):
    """Abstract base class for provider-specific API clients."""

    @abstractmethod
    def build_test_payload(self, model: str, *,
                           max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM) -> dict:
        """
        Build a minimal payload for a preflight availability check.

        Args:
            model: Model name.
            max_tokens_param: OpenAI-compatible reply-length field name;
                ignored by Ollama.

        Returns:
            A minimal request payload.
        """

    @abstractmethod
    def build_request_payload(self, model: str, prompt: str, system_prompt: str,
                             sampling_params: dict) -> dict:
        """Build the request payload for a translation/verification call."""

    @abstractmethod
    def parse_response(self, response_json: dict) -> ParsedReply:
        """Parse a successful API response into a ParsedReply."""

    @abstractmethod
    def parse_error_detail(self, response_json: dict) -> Optional[str]:
        """
        Extract a human-readable error message from a non-200 JSON body.

        Each provider shapes its error payload differently, so this is
        provider-specific rather than a generic dict lookup.

        Args:
            response_json: The parsed JSON body of a non-200 response.

        Returns:
            The provider's error message, or None if the shape is unrecognized.
        """

class OllamaAPIClient(APIClient):
    """Client for the Ollama API."""

    def build_request_payload(self, model: str, prompt: str, system_prompt: str,
                            sampling_params: dict) -> dict:
        """
        Build the payload for the Ollama /api/chat endpoint.

        Uses the messages[] structure for native isolation between the
        system and user context.
        """
        messages = []

        # Only attach a system message when a non-blank prompt is configured
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # A None option means "not configured"; Ollama would read an explicit
        # null as a value, so unset parameters are left out. keep_alive is a
        # top-level Ollama field, not an options entry.
        options = {
            k: v for k, v in sampling_params.items()
            if v is not None and k != 'keep_alive'
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        keep_alive = sampling_params.get('keep_alive')
        if isinstance(keep_alive, str) and keep_alive.strip():
            payload['keep_alive'] = keep_alive.strip()

        return payload

    def parse_response(self, response_json: dict) -> ParsedReply:
        """
        Parse a response from the Ollama /api/chat endpoint.

        Args:
            response_json: Decoded JSON body of a 200 response.

        Returns:
            A ParsedReply; text is '' when the reply carries no string
            content, finish_reason mirrors Ollama's own done_reason (e.g.
            "length" when the model hit num_predict), refusal is always
            False (Ollama has no such field).

        Raises:
            ValueError: If the body or its "message" field has the wrong type.
        """
        message = response_json.get("message", {}) if isinstance(response_json, dict) else None
        if not isinstance(message, dict):
            # ValueError, not TypeError: it is the contract call_model_api() degrades on.
            raise ValueError(f"Invalid Ollama response format: {response_json}")  # noqa: TRY004

        result_value = _join_text_parts(message.get("content", ''))

        prompt_tokens = response_json.get('prompt_eval_count', 0)
        completion_tokens = response_json.get('eval_count', 0)

        if not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        if not isinstance(completion_tokens, int):
            completion_tokens = 0

        finish_reason = response_json.get('done_reason')
        if not isinstance(finish_reason, str):
            finish_reason = None

        return ParsedReply(result_value, prompt_tokens, completion_tokens, finish_reason, False)

    def parse_error_detail(self, response_json: dict) -> Optional[str]:
        """Extract Ollama's error field, e.g. {"error": "model 'x' not found"}."""
        if not isinstance(response_json, dict):
            return None
        error = response_json.get('error')
        if isinstance(error, str) and error:
            return error
        return None

    def build_test_payload(self, model: str, *,
                           max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM) -> dict:
        """Build a minimal test request for the Ollama /api/chat endpoint."""
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "test"
                }
            ],
            "stream": False,
            "options": {
                "num_predict": 1
            }
        }

class OpenAIAPIClient(APIClient):
    """Client for OpenAI-compatible APIs (OpenAI, DeepSeek, OpenRouter, LM Studio, vLLM)."""

    def build_request_payload(self, model: str, prompt: str, system_prompt: str,
                             sampling_params: dict) -> dict:
        """
        Build the payload for an OpenAI-compatible /chat/completions endpoint.

        Parameters that are absent or None are omitted rather than sent as
        null, which strict servers reject. Ollama-only parameters (top_k,
        repeat_penalty, num_ctx, keep_alive) are dropped; num_predict is
        sent under sampling_params['max_tokens_param'] (default max_tokens).
        """
        openai_params = {}

        if sampling_params.get('num_predict') is not None:
            tokens_key = sampling_params.get('max_tokens_param') or DEFAULT_MAX_TOKENS_PARAM
            openai_params[tokens_key] = sampling_params['num_predict']
        if sampling_params.get('temperature') is not None:
            openai_params['temperature'] = sampling_params['temperature']
        if sampling_params.get('top_p') is not None:
            openai_params['top_p'] = sampling_params['top_p']
        if sampling_params.get('frequency_penalty') is not None:
            openai_params['frequency_penalty'] = sampling_params['frequency_penalty']
        if sampling_params.get('presence_penalty') is not None:
            openai_params['presence_penalty'] = sampling_params['presence_penalty']
        if sampling_params.get('stop'):
            openai_params['stop'] = sampling_params['stop']

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": model,
            "messages": messages,
            "stream": False,
            **openai_params
        }

    def parse_response(self, response_json: dict) -> ParsedReply:
        """
        Parse a response from an OpenAI-compatible chat completions endpoint.

        Args:
            response_json: Decoded JSON body of a 200 response.

        Returns:
            A ParsedReply; text is '' when the first choice carries no
            string content (null, or a list of content parts with no
            "text" part), missing or null usage counts as zero tokens,
            finish_reason mirrors the choice's own value (e.g. "length",
            "content_filter"), and refusal is True when message.refusal is
            a non-empty string.

        Raises:
            ValueError: If the body, "choices", the first choice or its
                "message" has the wrong type.
        """
        choices = response_json.get('choices') if isinstance(response_json, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError(f"Invalid OpenAI response format: {response_json}")

        message = choices[0].get('message', {})
        if not isinstance(message, dict):
            raise ValueError(f"Invalid OpenAI response format: {response_json}")  # noqa: TRY004

        result_value = _join_text_parts(message.get('content', ''))
        refusal = isinstance(message.get('refusal'), str) and bool(message['refusal'])

        usage = response_json.get('usage')
        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)

        if not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        if not isinstance(completion_tokens, int):
            completion_tokens = 0

        finish_reason = choices[0].get('finish_reason')
        if not isinstance(finish_reason, str):
            finish_reason = None

        return ParsedReply(result_value, prompt_tokens, completion_tokens, finish_reason, refusal)

    def parse_error_detail(self, response_json: dict) -> Optional[str]:
        """
        Extract the error text of an OpenAI-compatible provider.

        Understands the OpenAI shape {"error": {"message": ..., "code": ...}},
        a bare {"error": "..."} string, and the FastAPI shape used by some
        gateways: {"detail": "..."} or {"detail": [{"msg": ...}, ...]}.
        """
        if not isinstance(response_json, dict):
            return None

        error = response_json.get('error')
        if isinstance(error, dict):
            message = error.get('message') or error.get('code')
            if message:
                return str(message)
        elif isinstance(error, str) and error:
            return error

        detail = response_json.get('detail')
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, list):
            messages = [str(item['msg']) for item in detail
                        if isinstance(item, dict) and item.get('msg')]
            if messages:
                return '; '.join(messages)
        return None

    def build_test_payload(self, model: str, *,
                           max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM) -> dict:
        """Build a minimal test request for an OpenAI-compatible API."""
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": "test"}
            ],
            max_tokens_param: DEFAULT_OPENAI_PREFLIGHT_MAX_TOKENS,
            "stream": False
        }

_api_clients_cache: Dict[str, APIClient] = {}
_cache_lock = threading.Lock()

def create_api_client(api_type: str) -> APIClient:
    """
    Create (or reuse) an API client with thread-safe caching.

    Uses double-checked locking; clients are stateless and safely shared
    across threads.

    Args:
        api_type: 'ollama' or 'openai'.

    Returns:
        A cached APIClient instance.

    Raises:
        ValueError: If api_type is unknown.
    """
    if api_type in _api_clients_cache:
        return _api_clients_cache[api_type]

    with _cache_lock:
        if api_type not in _api_clients_cache:
            if api_type == 'ollama':
                _api_clients_cache[api_type] = OllamaAPIClient()
            elif api_type == 'openai':
                _api_clients_cache[api_type] = OpenAIAPIClient()
            else:
                raise ValueError(f"Unknown API type: {api_type}")

        return _api_clients_cache[api_type]

def describe_http_error(
    response: requests.Response,
    api_client: Optional[APIClient] = None,
    preview_length: int = HTTP_ERROR_BODY_PREVIEW_LENGTH,
) -> str:
    """
    Build a human-readable description of a non-200 HTTP response.

    Prefers the provider-specific error shape via api_client.parse_error_detail()
    (e.g. OpenAI/DeepSeek/OpenRouter's {"error": {"message": ...}} or
    Ollama's {"error": "..."}); falls back to a truncated raw body, labelled
    "unrecognized error format" for JSON and "non-JSON response" otherwise
    (an HTML page from a reverse proxy, for instance).

    Args:
        response: The non-200 response to describe.
        api_client: Provider client whose parse_error_detail() decodes the
            JSON error shape; omit to always use the raw body preview.
        preview_length: Max characters of the raw body kept in the fallback.

    Returns:
        A short, log-friendly description of the error body.
    """
    try:
        error_body = response.json()
    except (ValueError, json.JSONDecodeError):
        is_json = False
    else:
        is_json = True
        if api_client is not None:
            detail = api_client.parse_error_detail(error_body)
            if detail:
                return detail

    body_preview = response.text[:preview_length].strip()
    if not body_preview:
        return "(empty response body)"
    label = "unrecognized error format" if is_json else "non-JSON response"
    return f"{body_preview} ({label})"

def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """
    Parse a Retry-After header as seconds.

    Returns None for a missing header, one in the HTTP-date form (RFC
    7231 allows both) or a non-finite number; the caller falls back to
    exponential backoff in that case. The wait is clamped to
    [0, MAX_RETRY_AFTER_SECONDS].
    """
    if not header_value:
        return None
    try:
        seconds = float(header_value)
    except ValueError:
        return None
    if not math.isfinite(seconds):
        return None
    return min(max(0.0, seconds), float(MAX_RETRY_AFTER_SECONDS))

def make_request_with_retry(
    url: str,
    json_data: dict,
    config: 'ProcessorConfigProtocol',
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[int] = None,
    api_client: Optional[APIClient] = None,
) -> requests.Response:
    """
    Execute an HTTP POST with exponential backoff on transient failures.

    A 4xx client error (bad request, auth, not found, unprocessable) is
    treated as fatal and raised after the first attempt instead of being
    retried - it reproduces identically every time. A 429 respects a
    numeric Retry-After header when the provider sends one, falling back
    to the same exponential schedule otherwise. 5xx responses, timeouts
    and connection errors keep the exponential backoff.

    Args:
        url: Request URL.
        json_data: JSON-serializable request body.
        config: Logger/debug-flag facade used for diagnostics.
        headers: HTTP headers; defaults to a JSON content-type header.
        timeout: Per-request timeout in seconds; falls back to a constant.
        max_retries: Maximum attempts; falls back to a constant.
        backoff_factor: Exponential backoff base; falls back to a constant.
        api_client: Provider client used to decode a non-200 body into a
            human-readable error; omit to fall back to a raw body preview.

    Returns:
        The successful requests.Response.

    Raises:
        APIRequestError: After the last attempt is exhausted, or
            immediately on a non-retryable status code.
    """
    if timeout is None:
        timeout = DEFAULT_API_REQUEST_TIMEOUT
    if max_retries is None:
        max_retries = DEFAULT_API_MAX_RETRIES
    if backoff_factor is None:
        backoff_factor = DEFAULT_API_BACKOFF_FACTOR

    # At least one attempt, whatever the caller configured.
    max_retries = max(1, max_retries)

    if headers is None:
        headers = {"Content-Type": "application/json"}

    last_exception: Optional[requests.RequestException] = None
    last_detail: Optional[str] = None

    if config.is_debug_mode:
        config.logger.debug(
            f"HTTP POST to {url}\n"
            f"  Timeout: {timeout}s\n"
            f"  Max retries: {max_retries}"
        )

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, headers=headers, timeout=timeout)
            response.raise_for_status()

            if config.is_debug_mode:
                config.logger.debug(f"Request successful on attempt {attempt + 1}")

            return response

        except requests.exceptions.InvalidHeader:
            # Permanent, and the exception text embeds the offending header
            # value - for Authorization that is the API key, so it must be
            # neither logged nor chained.
            raise APIRequestError(
                "Request rejected locally: invalid HTTP header value "
                "(check api_key for control characters)"
            ) from None

        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema) as e:
            raise APIRequestError(f"Request rejected locally: {e}", original_exception=e) from e

        except requests.RequestException as e:
            last_exception = e
            status_code = e.response.status_code if e.response is not None else None
            last_detail = describe_http_error(e.response, api_client) if e.response is not None else None

            if status_code in NON_RETRYABLE_HTTP_STATUS_CODES:
                config.logger.warning(
                    f"Request failed with non-retryable status {status_code}: "
                    f"{last_detail}. Not retrying."
                )
                break

            is_last_attempt = attempt >= max_retries - 1
            if is_last_attempt:
                config.logger.warning(f"All {max_retries} attempts failed: {last_detail or last_exception}")
                break

            if status_code == HTTP_TOO_MANY_REQUESTS:
                wait_time = _parse_retry_after(e.response.headers.get('Retry-After')) if e.response is not None else None
                if wait_time is None:
                    wait_time = backoff_factor ** attempt
            else:
                wait_time = backoff_factor ** attempt

            if config.is_debug_mode:
                config.logger.debug(
                    f"Request failed: {last_detail or e}. Retry {attempt + 1}/{max_retries} in {wait_time}s"
                )
            time.sleep(wait_time)

    raise APIRequestError(
        f"Failed after {attempt + 1} attempt(s): {last_detail or last_exception}",
        original_exception=last_exception
    )

def call_model_api(value: str, operation_type: str, config: 'ProcessorConfigProtocol', runtime_config: dict,
                  glossary_string: str = "", mode: Optional[str] = None) -> CallOutcome:
    """
    Call the model API to process one text leaf (translate or verify).

    Never raises on a request/parsing failure: it logs and degrades to the
    original value instead (see call sites - the pipeline must not abort on
    a single bad line). Payload fields are read from runtime_config via
    .get() with safe fallbacks rather than direct indexing, so a caller bug
    that drops a key degrades the same way instead of raising KeyError. A
    reply the provider itself flags as cut off (finish/done reason "length")
    or declined (message.refusal) degrades the same way, even though the
    HTTP call succeeded - the content is still unusable.

    - This function is a pure API client with no side effects beyond
      config.metrics.
    - Tag masking/unmasking is the caller's responsibility.
    - The outcome's success flag reflects only the transport and parsing
      layer; it does not run the content sanity checks of
      llm_conductor/l10n/validation.py - that is the caller's job, after unmasking.

    Args:
        value: Text to process (may already be tag-masked by the caller).
        operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
        config: Processor-facing config (logger, metrics, debug flag).
        runtime_config: Flattened API request parameters.
        glossary_string: Glossary definitions to inject into the prompt.
        mode: Translation/verification mode, used only for metrics.

    Returns:
        A CallOutcome; on any failure its text is the unmodified value.
    """
    if not value.strip():
        return CallOutcome(value, True, None)

    start_time = time.time()

    api_type = runtime_config.get('api_type', 'ollama')
    try:
        api_client = create_api_client(api_type)
    except ValueError as e:
        config.logger.warning(f"{e}. Falling back to 'ollama'")
        api_client = create_api_client('ollama')

    # value may already be tag-masked by the caller (_process_leaf)
    user_prompt = runtime_config.get('prompt', '') + glossary_string + value
    system_prompt = runtime_config.get('system_prompt', '')
    model_name = runtime_config.get('model', '')

    keep_alive = runtime_config.get('keep_alive') or None
    sampling_params = {
        'temperature': runtime_config.get('temp'),
        'top_p': runtime_config.get('top_p'),
        'top_k': runtime_config.get('top_k'),
        'num_predict': runtime_config.get('num_predict'),
        'num_ctx': runtime_config.get('num_ctx'),
        'repeat_penalty': runtime_config.get('repeat_penalty'),
        'frequency_penalty': runtime_config.get('frequency_penalty'),
        'presence_penalty': runtime_config.get('presence_penalty'),
        'stop': runtime_config.get('stop', []),
        'keep_alive': keep_alive,
        'max_tokens_param': runtime_config.get(
            'max_tokens_param', DEFAULT_MAX_TOKENS_PARAM),
    }

    json_data = api_client.build_request_payload(model_name, user_prompt, system_prompt, sampling_params)

    headers = {"Content-Type": "application/json"}

    api_key = runtime_config.get('api_key')
    api_type = runtime_config.get('api_type', 'ollama')

    if api_type == 'openai' and isinstance(api_key, str) and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    if config.is_debug_mode:
        # Mask the API key before logging the headers
        safe_headers = headers.copy()
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = "Bearer ***MASKED***"

        config.logger.debug(
            f"API Request ({operation_type}):\n"
            f"  Model: {model_name}\n"
            f"  API Type: {api_type}\n"
            f"  Headers: {safe_headers}"
        )

        config.logger.debug(
            f"System Prompt:\n{system_prompt if system_prompt else '(empty)'}"
        )

        prompt_preview = user_prompt if len(user_prompt) <= 500 else user_prompt[:500] + "..."
        config.logger.debug(
            f"User Prompt:\n{prompt_preview}"
        )

        config.logger.debug(
            f"Sampling params: temp={sampling_params['temperature']}, "
            f"top_p={sampling_params['top_p']}, "
            f"top_k={sampling_params['top_k']}, "
            f"num_predict={sampling_params['num_predict']}"
        )

    try:
        response = make_request_with_retry(
            runtime_config.get('api_url', ''),
            json_data,
            config,
            headers=headers,
            timeout=runtime_config.get('api_timeout'),
            max_retries=runtime_config.get('max_retries'),
            backoff_factor=runtime_config.get('retry_backoff_factor'),
            api_client=api_client,
        )

        end_time = time.time()
        request_duration = end_time - start_time

        try:
            response_json = response.json()
            parsed = api_client.parse_response(response_json)

            if parsed.refusal:
                config.logger.warning("Model declined the request (refusal). Using original value.")
                result_value = value
                failure_kind: Optional[str] = FAILURE_KIND_REFUSAL
                config.metrics.update(
                    operation_type, request_duration, failed=True,
                    prompt_tokens=parsed.prompt_tokens, completion_tokens=parsed.completion_tokens,
                    failure_kind=FAILURE_KIND_REFUSAL,
                )
            elif parsed.finish_reason == FINISH_REASON_LENGTH:
                config.logger.warning(
                    "Reply was cut off by the token limit (finish/done reason "
                    f"'{FINISH_REASON_LENGTH}'). Using original value."
                )
                result_value = value
                failure_kind = FAILURE_KIND_TRUNCATED
                config.metrics.update(
                    operation_type, request_duration, failed=True,
                    prompt_tokens=parsed.prompt_tokens, completion_tokens=parsed.completion_tokens,
                    failure_kind=FAILURE_KIND_TRUNCATED,
                )
            elif not parsed.text or not isinstance(parsed.text, str):
                config.logger.warning("API returned invalid response. Using original value.")
                result_value = value
                failure_kind = FAILURE_KIND_PARSE
                config.metrics.update(
                    operation_type, request_duration, failed=True,
                    prompt_tokens=parsed.prompt_tokens, completion_tokens=parsed.completion_tokens,
                    failure_kind=FAILURE_KIND_PARSE,
                )
            else:
                result_value = parsed.text.strip()
                failure_kind = None
                config.metrics.update(
                    operation_type, request_duration, failed=False,
                    prompt_tokens=parsed.prompt_tokens, completion_tokens=parsed.completion_tokens,
                )

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError, KeyError) as e:
            config.logger.error(
                f"Failed to parse response ({type(e).__name__}): {e}. Using original value."
            )
            result_value = value
            failure_kind = FAILURE_KIND_PARSE
            config.metrics.update(operation_type, request_duration, failed=True, failure_kind=FAILURE_KIND_PARSE)

    except APIRequestError as e:
        end_time = time.time()
        request_duration = end_time - start_time
        config.metrics.update(operation_type, request_duration, failed=True, failure_kind=FAILURE_KIND_NETWORK)
        config.logger.warning(f"API request failed: {e}. Using original value.")
        result_value = value
        failure_kind = FAILURE_KIND_NETWORK

    return CallOutcome(result_value, failure_kind is None, failure_kind)

def translate_key(value: str, config: 'ProcessorConfigProtocol', runtime_config: dict, glossary_string: str = "",
                 mode: Optional[str] = None) -> CallOutcome:
    """
    Translate one text leaf via the model.

    value may already carry masked tags ([T0], [T1], ...) from
    _process_leaf. Caching happens one layer up, not here.

    Args:
        value: Text to translate (may be tag-masked).
        config: Processor-facing config.
        runtime_config: Flattened API request parameters.
        glossary_string: Glossary definitions to inject into the prompt.
        mode: Translation mode (strict/style), used only for metrics.

    Returns:
        A CallOutcome; its text still carries masked tags.
    """
    return call_model_api(
        value=value,
        operation_type=OPERATION_TRANSLATION,
        config=config,
        runtime_config=runtime_config,
        glossary_string=glossary_string,
        mode=mode
    )

def verify_key(value: str, config: 'ProcessorConfigProtocol', runtime_config: dict, glossary_string: str = "",
              mode: Optional[str] = None) -> CallOutcome:
    """
    Verify/restyle one text leaf via the model.

    value may already carry masked tags ([T0], [T1], ...) from
    _process_leaf. Caching happens one layer up, not here.

    Args:
        value: Text to verify (may be tag-masked).
        config: Processor-facing config.
        runtime_config: Flattened API request parameters.
        glossary_string: Glossary definitions to inject into the prompt.
        mode: Mode (verification), used only for metrics.

    Returns:
        A CallOutcome; its text still carries masked tags.
    """
    return call_model_api(
        value=value,
        operation_type=OPERATION_VERIFICATION,
        config=config,
        runtime_config=runtime_config,
        glossary_string=glossary_string,
        mode=mode
    )
