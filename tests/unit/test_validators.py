# -*- coding: utf-8 -*-
"""
Unit tests for CLI-layer validators without network (llm_conductor/cli/validators.py):
validate_runtime_config and validate_input_data.

check_api_availability hits HTTP — covered in the integration tier (responses).
"""
import pytest

from llm_conductor.cli.validators import validate_input_data, validate_runtime_config
from llm_conductor.config import ConfigData, RuntimeConfigFactory
from llm_conductor.constants import MODE_STRICT
from llm_conductor.exceptions import ConfigValidationError
from tests.support import StubConfig


def _runtime(**overrides):
    runtime = RuntimeConfigFactory(ConfigData.from_toml({}, {}, environ={})).get_translation_config_for_mode(MODE_STRICT)
    runtime['mode'] = MODE_STRICT
    runtime.update(overrides)
    return runtime


# ============================================================================
# validate_runtime_config
# ============================================================================

def test_default_runtime_config_is_valid_and_returned_as_is():
    runtime = _runtime()

    assert validate_runtime_config(runtime, StubConfig(), "translation") is runtime


@pytest.mark.parametrize("runtime", [None, {}])
def test_missing_runtime_config_is_rejected(runtime):
    with pytest.raises(ConfigValidationError):
        validate_runtime_config(runtime, StubConfig(), "translation")


@pytest.mark.parametrize("key, bad_value", [
    ("model", 42),
    ("temp", "0.3"),
    ("top_k", 4.5),
    ("num_predict", "150"),
    ("stop", "\n\n"),
    ("api_url", None),
    ("keep_alive", 5),
    ("max_tokens_param", True),
])
def test_wrong_type_is_rejected(key, bad_value):
    with pytest.raises(ConfigValidationError, match=key):
        validate_runtime_config(_runtime(**{key: bad_value}), StubConfig(), "translation")


@pytest.mark.parametrize("key", ["num_predict", "num_ctx"])
def test_non_positive_token_limits_are_rejected(key):
    with pytest.raises(ConfigValidationError, match=key):
        validate_runtime_config(_runtime(**{key: 0}), StubConfig(), "translation")


def test_invalid_max_tokens_param_is_rejected():
    with pytest.raises(ConfigValidationError, match="max_tokens_param"):
        validate_runtime_config(
            _runtime(max_tokens_param="max_output_tokens"), StubConfig(), "translation")


@pytest.mark.characterization(
    reason="out-of-range temp/top_p/api_type only warn here — ConfigData.validate() rejects them")
@pytest.mark.parametrize("overrides, warning_fragment", [
    ({"temp": 2.5}, "temp"),
    ({"top_p": 1.5}, "top_p"),
    ({"mode": "creative"}, "mode"),
    ({"api_type": "grpc"}, "api_type"),
])
def test_soft_range_violations_only_warn(overrides, warning_fragment):
    config = StubConfig()

    validate_runtime_config(_runtime(**overrides), config, "translation")

    assert config.logger.contains(f"runtime_config['{warning_fragment}']", level='WARNING')


def test_missing_recommended_fields_only_warn():
    config = StubConfig()

    validate_runtime_config({"temp": 0.3}, config, "translation")

    assert config.logger.contains("missing recommended field: model", level='WARNING')
    assert config.logger.contains("missing recommended field: prompt", level='WARNING')


# ============================================================================
# validate_input_data
# ============================================================================

@pytest.mark.parametrize("data", [{"a": "b"}, ["a"], "text", 1, 1.5, True])
def test_any_json_value_is_accepted(data):
    assert validate_input_data(data, "in.json", StubConfig()) is True


@pytest.mark.parametrize("data, warning", [({}, "empty object"), ([], "empty array")])
def test_empty_containers_are_accepted_with_warning(data, warning):
    config = StubConfig()

    assert validate_input_data(data, "in.json", config) is True
    assert config.logger.contains(warning, level='WARNING')


def test_null_input_is_rejected():
    config = StubConfig()

    assert validate_input_data(None, "in.json", config) is False
    assert config.logger.contains("null data", level='ERROR')


def test_non_json_type_is_rejected():
    assert validate_input_data(object(), "in.json", StubConfig()) is False
