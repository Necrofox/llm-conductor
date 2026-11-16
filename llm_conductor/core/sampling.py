"""Pure sampling serialization shared by HTTP payloads and cache identity."""
from typing import Any, Dict, Mapping

from llm_conductor.constants import DEFAULT_MAX_TOKENS_PARAM


def build_openai_sampling_payload(sampling_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Select and map the sampling fields sent to an OpenAI-compatible API.

    Args:
        sampling_params: Logical sampling names after explicit omissions,
            plus the optional ``max_tokens_param`` transport selector.

    Returns:
        The actual JSON field names and values, without messages or model.
        Unsupported fields, None values, and empty stop lists are absent.
    """
    openai_params: Dict[str, Any] = {}
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
    return openai_params
