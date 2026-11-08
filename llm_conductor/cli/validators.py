# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

import requests

from llm_conductor.constants import (
    DEFAULT_MAX_TOKENS_PARAM,
    DEFAULT_PREFLIGHT_TIMEOUT,
    HTTP_FORBIDDEN,
    HTTP_NOT_FOUND,
    HTTP_PAYMENT_REQUIRED,
    HTTP_TOO_MANY_REQUESTS,
    HTTP_UNAUTHORIZED,
    VALID_API_TYPES,
    VALID_MAX_TOKENS_PARAMS,
    VALID_MODES,
)
from llm_conductor.core.api_client import create_api_client, describe_http_error
from llm_conductor.exceptions import ConfigValidationError

if TYPE_CHECKING:
    from llm_conductor.interfaces import ConfigProtocol

def validate_runtime_config(runtime_config: Optional[dict], config: 'ConfigProtocol', operation_type: str) -> dict:
    """
    Validate ``runtime_config`` before it is used in API requests.

    Checks:
    - recommended fields (``model``, ``prompt``)
    - types of all known parameters
    - value ranges (``temp``, ``top_p``, ``num_predict``, …)

    Args:
        runtime_config: Flat runtime config dict to validate.
        config: Config facade used for logging.
        operation_type: ``'translation'`` or ``'verification'``.

    Raises:
        ConfigValidationError: On critical validation failures.

    Returns:
        The same ``runtime_config`` after validation.
    """
    if runtime_config is None:
        raise ConfigValidationError("runtime_config cannot be None")

    if not runtime_config:
        raise ConfigValidationError("runtime_config cannot be empty")

    if not isinstance(runtime_config, dict):
        raise ConfigValidationError(
            f"runtime_config must be a dictionary, got {type(runtime_config).__name__}"
        )

    required_fields = ['model', 'prompt']
    for field in required_fields:
        if field not in runtime_config:
            config.logger.warning(f"runtime_config missing recommended field: {field}")

    validations: Dict[str, Union[Type, Tuple[Type, ...]]] = {
        'model': str,
        'system_prompt': str,
        'prompt': str,
        'temp': (int, float),
        'top_p': (int, float),
        'top_k': int,
        'repeat_penalty': (int, float),
        'frequency_penalty': (int, float),
        'presence_penalty': (int, float),
        'num_predict': int,
        'num_ctx': int,
        'stop': list,
        'api_url': str,
        'mode': str,
        'keep_alive': str,
        'max_tokens_param': str,
    }
    
    for key, expected_type in validations.items():
        if key in runtime_config:
            value = runtime_config[key]
            if not isinstance(value, expected_type):
                raise ConfigValidationError(
                    f"runtime_config['{key}'] must be {expected_type}, "
                    f"got {type(value).__name__}"
                )

    if 'api_type' in runtime_config and runtime_config['api_type'] not in VALID_API_TYPES:
        config.logger.warning(
            f"runtime_config['api_type'] should be one of {VALID_API_TYPES}, "
            f"got '{runtime_config['api_type']}'"
        )

    if ('max_tokens_param' in runtime_config
            and runtime_config['max_tokens_param'] not in VALID_MAX_TOKENS_PARAMS):
        raise ConfigValidationError(
            f"runtime_config['max_tokens_param'] must be one of "
            f"{VALID_MAX_TOKENS_PARAMS}, got '{runtime_config['max_tokens_param']}'"
        )

    if 'temp' in runtime_config and not 0.0 <= runtime_config['temp'] <= 2.0:
        config.logger.warning(
            f"runtime_config['temp'] should be between 0.0 and 2.0, "
            f"got {runtime_config['temp']}"
        )

    if 'top_p' in runtime_config and not 0.0 <= runtime_config['top_p'] <= 1.0:
        config.logger.warning(
            f"runtime_config['top_p'] should be between 0.0 and 1.0, "
            f"got {runtime_config['top_p']}"
        )

    if 'num_predict' in runtime_config and runtime_config['num_predict'] <= 0:
        raise ConfigValidationError(
            f"runtime_config['num_predict'] must be positive, "
            f"got {runtime_config['num_predict']}"
        )

    if 'num_ctx' in runtime_config and runtime_config['num_ctx'] <= 0:
        raise ConfigValidationError(
            f"runtime_config['num_ctx'] must be positive, "
            f"got {runtime_config['num_ctx']}"
        )

    if 'mode' in runtime_config and runtime_config['mode'] not in VALID_MODES:
        config.logger.warning(
            f"runtime_config['mode'] should be one of {VALID_MODES}, "
            f"got '{runtime_config['mode']}'"
        )
    
    if config.is_debug_mode:
        config.logger.debug(
            f"Validated {operation_type} runtime_config: "
            f"model={runtime_config.get('model')}, "
            f"temp={runtime_config.get('temp')}, "
            f"mode={runtime_config.get('mode')}"
        )
    
    return runtime_config

def validate_input_data(input_data: Any, file_name: str, config: 'ConfigProtocol') -> bool:
    """
    Validate input data for the recursive engine (lightweight check).

    Args:
        input_data: Parsed JSON value to validate.
        file_name: File name used in error messages.
        config: Config facade used for logging.

    Returns:
        True for any valid JSON value; False only on critical problems.
    """
    if input_data is None:
        config.logger.error(f"Input file '{file_name}' contains null data")
        return False

    if isinstance(input_data, (dict, list, str, int, float, bool)):
        if isinstance(input_data, dict) and len(input_data) == 0:
            config.logger.warning(f"Input file '{file_name}' contains empty object")
        elif isinstance(input_data, list) and len(input_data) == 0:
            config.logger.warning(f"Input file '{file_name}' contains empty array")

        return True

    # Should not happen for well-formed JSON.
    config.logger.error(
        f"Input file '{file_name}' contains unsupported data type: {type(input_data).__name__}"
    )
    return False

def _diagnose_status_code(status_code: int) -> str:
    """
    Map a non-200 preflight status code to an actionable hint.

    Args:
        status_code: HTTP status code returned by the preflight request.

    Returns:
        A short phrase explaining the likely cause.
    """
    if status_code in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
        return "invalid or revoked API key"
    if status_code == HTTP_PAYMENT_REQUIRED:
        return "payment required - check the account balance"
    if status_code == HTTP_NOT_FOUND:
        return "endpoint or model not found - check url and model in [translation.api] / [verification.api]"
    if status_code == HTTP_TOO_MANY_REQUESTS:
        return "rate limit exceeded"
    if 500 <= status_code < 600:
        return "provider-side failure"
    return "unexpected response"

def check_api_availability(
    api_url: str,
    model_name: str,
    api_type: str,
    config: 'ConfigProtocol',
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_PREFLIGHT_TIMEOUT,
    max_tokens_param: str = DEFAULT_MAX_TOKENS_PARAM,
) -> bool:
    """
    Check whether an API endpoint and model respond before the pipeline starts.

    A non-200 response is decoded through describe_http_error() (the
    provider's JSON error shape, or a truncated raw body for an HTML page
    from a reverse proxy) and annotated with a status-specific hint, so a
    misconfigured profile fails with an explanation instead of a bare
    status code.

    Args:
        api_url: API endpoint URL.
        model_name: Model name to probe.
        api_type: 'ollama' or 'openai'.
        config: Config facade used for logging.
        api_key: Bearer token for api_type='openai'.
        timeout: Request timeout in seconds.
        max_tokens_param: Reply-length field name for OpenAI-compatible
            preflight probes; ignored by Ollama.

    Returns:
        True if the endpoint responded with 200, False otherwise.
    """
    try:
        api_client = create_api_client(api_type)
        test_data = api_client.build_test_payload(
            model_name, max_tokens_param=max_tokens_param)

        headers = {"Content-Type": "application/json"}
        if api_type == 'openai' and isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        response = requests.post(api_url, json=test_data, headers=headers, timeout=timeout)

        if response.status_code == 200:
            config.logger.info(
                f"API endpoint '{api_url}' ({api_type.upper()}) is accessible, "
                f"model '{model_name}' is available"
            )
            return True

        detail = describe_http_error(response, api_client)
        hint = _diagnose_status_code(response.status_code)
        config.logger.error(
            f"API endpoint '{api_url}' ({api_type.upper()}) "
            f"returned status code {response.status_code} ({hint}): {detail}"
        )
        return False

    except requests.exceptions.InvalidHeader:
        # The exception text embeds the header value, i.e. the API key.
        config.logger.error(
            f"API endpoint '{api_url}' ({api_type.upper()}): invalid HTTP header value "
            f"(check api_key for control characters)"
        )
        return False

    except ValueError as e:
        # Client construction failed - unknown api_type
        config.logger.error(f"Invalid API type '{api_type}': {e}")
        return False

    except requests.Timeout:
        config.logger.error(
            f"API endpoint '{api_url}' ({api_type.upper()}) "
            f"timed out after {timeout} seconds"
        )
        return False

    except requests.ConnectionError:
        config.logger.error(
            f"Failed to connect to API endpoint '{api_url}' ({api_type.upper()}). "
            f"Check if the server is running."
        )
        return False

    except Exception as e:  # noqa: BLE001 - preflight must return False, not crash the run
        config.logger.error(
            f"Unexpected error checking API availability ({api_type.upper()}): {e}"
        )
        if config.is_debug_mode:
            config.logger.error("Full traceback:", include_traceback=True)
        return False
