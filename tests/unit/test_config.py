# -*- coding: utf-8 -*-
"""
Unit tests of the configuration layer without files: the schema
(llm_conductor/schema.py), ConfigData.from_toml / validate, APIConfig,
TranslationConfig, RuntimeConfigFactory, the DI factories of Config,
ConfigLocations, RunOverrides and the log file name.

Loading from disk and the 'init' templates are tested in the integration tier.
"""
import dataclasses
import sys
from datetime import UTC, datetime

import pytest

from llm_conductor.cli.locations import ConfigLocations, resolve_app_dir
from llm_conductor.config import (
    APIConfig,
    Config,
    ConfigData,
    ConfigLogger,
    RunOverrides,
    RuntimeConfigFactory,
    TranslationConfig,
)
from llm_conductor.constants import (
    CONFIG_PATH_ENV_VAR,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TRANSLATION_API_TIMEOUT,
    DEFAULT_TRANSLATION_API_TYPE,
    DEFAULT_TRANSLATION_STRICT_TOP_P,
    DEFAULT_TRANSLATION_STYLE_TOP_P,
    DEFAULT_VERIFICATION_API_TIMEOUT,
    MAX_OPENAI_STOP_SEQUENCES,
    MODE_STRICT,
    MODE_STYLE,
    TRACE_LOG_FILE_PREFIX,
)
from llm_conductor.exceptions import ConfigValidationError
from llm_conductor.schema import CONDUCTOR_SCHEMA, LLM_SCHEMA
from tests.support import RecordingLogger, split_overrides


def _data(environ=None, **overrides) -> ConfigData:
    """ConfigData from empty documents plus double-underscore overrides."""
    llm, conductor = split_overrides(**overrides)
    return ConfigData.from_toml(llm, conductor, environ=environ if environ is not None else {})


# ============================================================================
# Schema - strict validation
# ============================================================================

def test_empty_documents_fall_back_to_defaults_and_validate():
    """Absent keys take DEFAULT_* values (checklist §11: an older file still loads)."""
    data = _data()

    assert data.translation_api.api_type == DEFAULT_TRANSLATION_API_TYPE
    assert data.translation_api.timeout == DEFAULT_TRANSLATION_API_TIMEOUT
    assert data.verification_api.timeout == DEFAULT_VERIFICATION_API_TIMEOUT
    assert data.cache_max_size == DEFAULT_CACHE_MAX_SIZE
    assert data.cache_dir == DEFAULT_CACHE_DIR
    assert data.log_level == DEFAULT_LOG_LEVEL


def test_default_translation_endpoint_matches_default_api_type():
    """The default URL is the native Ollama /api/chat, so the default type is 'ollama'."""
    api = _data().translation_api

    assert api.url.endswith("/api/chat")
    assert api.api_type == "ollama"


@pytest.mark.parametrize("llm_doc, fragment", [
    ({"translation": {"sampling": {"strict": {"tempreature": 0.1}}}}, "did you mean 'temperature'"),
    ({"translaton": {}}, "did you mean 'translation'"),
    ({"translation": {"api": {"zzz": 1}}}, "unknown key 'zzz' in [translation.api]"),
    ({"translation": {"api": "x"}}, "must be a table"),
])
def test_unknown_or_misplaced_keys_are_rejected(llm_doc, fragment):
    with pytest.raises(ConfigValidationError) as error:
        ConfigData.from_toml(llm_doc, {}, environ={})

    assert fragment in str(error.value)


@pytest.mark.parametrize("conductor_doc, fragment", [
    ({"cache": {"max_size": "big"}}, "expected integer"),
    ({"cache": {"enabled": 1}}, "expected boolean"),
    ({"limits": {"max_recursion_depth": True}}, "expected integer"),
])
def test_wrongly_typed_values_are_rejected(conductor_doc, fragment):
    with pytest.raises(ConfigValidationError, match=fragment):
        ConfigData.from_toml({}, conductor_doc, environ={})


def test_every_problem_of_a_file_is_reported_at_once():
    with pytest.raises(ConfigValidationError) as error:
        ConfigData.from_toml({}, {"cache": {"max_sise": 1, "enabled": "yes"}, "limitz": {}}, environ={})

    message = str(error.value)
    assert "max_sise" in message and "'enabled'" in message and "limitz" in message


def test_integer_is_accepted_where_a_float_is_expected():
    data = _data(translation__api__retry_backoff_factor=3)

    assert data.translation_api.retry_backoff_factor == 3.0
    assert isinstance(data.translation_api.retry_backoff_factor, float)


@pytest.mark.parametrize("schema", [LLM_SCHEMA, CONDUCTOR_SCHEMA], ids=lambda s: s.file_name)
def test_every_schema_default_passes_its_own_type_check(schema):
    for section in schema.sections:
        for setting in section.settings:
            for label, value in (("default", setting.default),
                                 ("template_default", setting.template_default)):
                if value is None and label == "template_default":
                    continue
                check = list(value) if isinstance(value, tuple) else value
                assert setting.kind.accepts(check), f"{section.name}.{setting.key} ({label})"


def test_from_toml_lowercases_api_type_and_uppercases_log_level():
    data = _data(translation__api__type="OpenAI", translation__api__key="sk-x", logging__level="debug")

    assert data.translation_api.api_type == "openai"
    assert data.log_level == "DEBUG"


def test_strict_and_style_sampling_are_independent():
    """Each translation mode has its own top_p/top_k/repeat_penalty; stop is shared."""
    data = _data(translation__sampling__strict__temperature=0.1,
                 translation__sampling__style__temperature=0.9,
                 translation__sampling__stop=["<|im_end|>"])

    assert (data.translation_strict.temp, data.translation_style.temp) == (0.1, 0.9)
    assert (data.translation_strict.top_p, data.translation_style.top_p) == (
        DEFAULT_TRANSLATION_STRICT_TOP_P, DEFAULT_TRANSLATION_STYLE_TOP_P)
    assert data.translation_strict.stop == data.translation_style.stop == ("<|im_end|>",)


@pytest.mark.parametrize("level, debug", [("DEBUG", True), ("VERBOSE", True), ("INFO", False)])
def test_is_debug_mode_follows_log_level(level, debug):
    assert _data(logging__level=level).is_debug_mode is debug


def test_relative_directories_are_anchored_at_base_dir(tmp_path):
    data = ConfigData.from_toml({}, {"cache": {"dir": "c"}, "logging": {"dir": str(tmp_path / "abs")}},
                                base_dir=str(tmp_path), environ={})

    assert data.cache_dir == str(tmp_path / "c")
    assert data.log_dir == str(tmp_path / "abs")


def test_config_data_is_immutable():
    data = _data()

    with pytest.raises(dataclasses.FrozenInstanceError):
        data.cache_dir = "elsewhere"
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.translation_api.model = "other"


# ============================================================================
# ConfigData.validate - system parameters
# ============================================================================

@pytest.mark.parametrize("overrides, message", [
    ({"cache__max_size": 0}, "cache_max_size"),
    ({"cache__expiry_days": 0}, "cache_expiry_days"),
    ({"cache__auto_save_interval": -1}, "cache_auto_save_interval"),
    ({"cache__glossary_patterns_max_size": 0}, "glossary_patterns_max_size"),
    ({"cache__dir": ""}, "cache_dir"),
    ({"logging__dir": ""}, "log_dir"),
    ({"logging__level": "LOUD"}, "log level"),
    ({"limits__max_recursion_depth": 0}, "max_recursion_depth"),
    ({"limits__dedup_log_interval": -5}, "dedup_log_interval"),
    ({"preflight__timeout": 0}, "preflight timeout"),
])
def test_validate_rejects_invalid_system_params(overrides, message):
    with pytest.raises(ConfigValidationError, match=message):
        _data(**overrides)


def test_validate_accepts_zero_auto_save_interval():
    """0 turns auto-save off; it is not an error."""
    _data(cache__auto_save_interval=0)


# ============================================================================
# APIConfig
# ============================================================================

def _api(**overrides):
    params = {"url": "http://localhost:11434/api/chat", "api_type": "ollama", "timeout": 30,
              "max_retries": 3, "retry_backoff_factor": 2.0, "model": "qwen"}
    params.update(overrides)
    return APIConfig(**params)


@pytest.mark.parametrize("overrides, message", [
    ({"url": ""}, "URL"),
    ({"api_type": "grpc"}, "api_type"),
    ({"timeout": 0}, "timeout"),
    ({"model": ""}, "model"),
    ({"api_type": "openai", "api_key": None}, "api_key is required"),
    ({"api_type": "openai", "api_key": ""}, "api_key is required"),
    ({"max_tokens_param": "max_output_tokens"}, "max_tokens_param"),
])
def test_api_config_validate_rejects(overrides, message):
    with pytest.raises(ConfigValidationError, match=message):
        _api(**overrides).validate()


def test_api_config_ollama_needs_no_key():
    _api(api_type="ollama", api_key=None).validate()


def test_api_config_headers_add_bearer_only_for_openai():
    assert "Authorization" not in _api(api_type="ollama", api_key="secret").get_headers()
    assert _api(api_type="openai", api_key="sk-1").get_headers()["Authorization"] == "Bearer sk-1"


@pytest.mark.parametrize("environ, expected", [
    ({"MY_KEY": "sk-env"}, "sk-env"),
    ({"MY_KEY": "   "}, "sk-literal"),
    ({}, "sk-literal"),
])
def test_key_env_has_priority_over_literal_key(environ, expected):
    data = _data(environ=environ, translation__api__key_env="MY_KEY", translation__api__key="sk-literal")

    assert data.translation_api.api_key == expected


# ============================================================================
# TranslationConfig
# ============================================================================

@pytest.mark.parametrize("overrides, message", [
    ({"temp": -0.1}, "temp"),
    ({"temp": 2.1}, "temp"),
    ({"top_p": 1.5}, "top_p"),
    ({"num_predict": 0}, "num_predict"),
    ({"num_ctx": 0}, "num_ctx"),
])
def test_translation_config_validate_rejects(overrides, message):
    params = {"prompt": "p", "temp": 0.3, "num_predict": 100, "num_ctx": 2048, "top_p": 0.9,
              "top_k": 40, "repeat_penalty": 1.1, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    params.update(overrides)

    with pytest.raises(ConfigValidationError, match=message):
        TranslationConfig(**params).validate()


# ============================================================================
# RuntimeConfigFactory
# ============================================================================

_RUNTIME_KEYS = {
    'api_url', 'api_type', 'api_key', 'api_timeout', 'max_retries', 'retry_backoff_factor',
    'model', 'system_prompt', 'keep_alive', 'max_tokens_param', 'prompt', 'temp',
    'num_predict', 'num_ctx', 'top_p', 'top_k', 'repeat_penalty', 'frequency_penalty',
    'presence_penalty', 'stop',
}


@pytest.fixture
def factory():
    return RuntimeConfigFactory(_data(
        translation__api__model="trans-model",
        translation__prompts__strict="STRICT",
        translation__prompts__style="STYLE",
        translation__sampling__strict__temperature=0.1,
        translation__sampling__style__temperature=0.9,
        verification__api__model="verify-model",
        verification__prompts__edit="VERIFY",
        verification__api__url="http://verify/api/chat",
    ))


def test_runtime_config_contains_full_contract_for_call_model_api(factory):
    assert set(factory.get_translation_config_for_mode(MODE_STRICT)) == _RUNTIME_KEYS
    assert set(factory.get_verification_config()) == _RUNTIME_KEYS


@pytest.mark.parametrize("mode, prompt, temp", [
    (MODE_STRICT, "STRICT", 0.1),
    (MODE_STYLE, "STYLE", 0.9),
])
def test_translation_runtime_config_picks_mode(factory, mode, prompt, temp):
    runtime = factory.get_translation_config_for_mode(mode)

    assert (runtime['prompt'], runtime['temp'], runtime['model']) == (prompt, temp, "trans-model")


@pytest.mark.characterization(reason="an unknown mode silently means style, without an error or a warning")
def test_unknown_mode_silently_falls_back_to_style(factory):
    runtime = factory.get_translation_config_for_mode("unknown-mode")

    assert runtime['prompt'] == "STYLE"


def test_verification_runtime_config_uses_verification_endpoint(factory):
    runtime = factory.get_verification_config()

    assert (runtime['prompt'], runtime['model'], runtime['api_url']) == (
        "VERIFY", "verify-model", "http://verify/api/chat")


def test_runtime_config_default_keep_alive_is_empty(factory):
    """Empty keep_alive means Ollama's server default; the field is omitted later."""
    assert factory.get_translation_config_for_mode(MODE_STRICT)['keep_alive'] == ''
    assert factory.get_verification_config()['keep_alive'] == ''


def test_runtime_config_passes_keep_alive_from_api_section():
    factory = RuntimeConfigFactory(_data(
        translation__api__model="trans-model",
        translation__api__keep_alive="5m",
        translation__prompts__strict="STRICT",
        translation__prompts__style="STYLE",
        verification__api__model="verify-model",
        verification__api__keep_alive="0",
        verification__prompts__edit="VERIFY",
    ))

    assert factory.get_translation_config_for_mode(MODE_STRICT)['keep_alive'] == "5m"
    assert factory.get_verification_config()['keep_alive'] == "0"


def test_runtime_config_default_max_tokens_param_is_max_tokens(factory):
    assert factory.get_translation_config_for_mode(MODE_STRICT)['max_tokens_param'] == 'max_tokens'
    assert factory.get_verification_config()['max_tokens_param'] == 'max_tokens'


def test_runtime_config_passes_max_tokens_param_from_api_section():
    factory = RuntimeConfigFactory(_data(
        translation__api__model="trans-model",
        translation__api__max_tokens_param="max_completion_tokens",
        translation__prompts__strict="STRICT",
        translation__prompts__style="STYLE",
        verification__api__model="verify-model",
        verification__api__max_tokens_param="max_tokens",
        verification__prompts__edit="VERIFY",
    ))

    assert (factory.get_translation_config_for_mode(MODE_STRICT)['max_tokens_param']
            == "max_completion_tokens")
    assert factory.get_verification_config()['max_tokens_param'] == "max_tokens"


def test_runtime_config_is_a_fresh_copy_each_time(factory):
    """Callers add 'mode' and may touch 'stop' - the shared configuration must not change."""
    first = factory.get_verification_config()
    first['stop'].append("mutated")
    first['mode'] = "x"

    second = factory.get_verification_config()
    assert "mutated" not in second['stop'] and 'mode' not in second


# ============================================================================
# Config - DI factories
# ============================================================================

def test_config_uses_injected_factories_instead_of_concrete_managers():
    data = _data(cache__glossary_patterns_max_size=7)
    created = {}

    def cache_manager_factory(config):
        created['cache_config'] = config
        return "fake-cache-manager"

    def glossary_cache_factory(max_size):
        created['glossary_max_size'] = max_size
        return "fake-glossary-cache"

    config = Config(
        data,
        RecordingLogger(),
        metrics_factory=lambda: "fake-metrics",
        cache_manager_factory=cache_manager_factory,
        glossary_cache_factory=glossary_cache_factory,
    )

    assert config.metrics == "fake-metrics"
    assert config.cache_manager == "fake-cache-manager"
    assert config.glossary_patterns_cache == "fake-glossary-cache"
    assert created == {'cache_config': config, 'glossary_max_size': 7}


def test_config_properties_delegate_to_config_data():
    data = _data(logging__level="DEBUG", cache__dir="/tmp/c", cache__max_size=5,
                 limits__max_error_display=3, meta__name="my profile")
    config = Config(data, RecordingLogger(), metrics_factory=lambda: None,
                    cache_manager_factory=lambda c: None, glossary_cache_factory=lambda n: None)

    assert config.is_debug_mode is True
    assert (config.cache_dir, config.cache_max_size) == ("/tmp/c", 5)
    assert config.max_error_display == 3
    assert config.profile_name == "my profile"
    assert config.translation_api is data.translation_api


# ============================================================================
# OpenAI caps the number of stop sequences
# ============================================================================

_FIVE_STOPS = ["a", "b", "c", "d", "e"]


def _for(api_type, **overrides):
    return _data(translation__api__type=api_type, verification__api__type=api_type,
                 translation__api__key="sk-test", verification__api__key="sk-test", **overrides)


def test_openai_limit_is_four_stop_sequences():
    assert MAX_OPENAI_STOP_SEQUENCES == 4


@pytest.mark.parametrize("key", ["translation__sampling__stop", "verification__sampling__stop"])
def test_openai_rejects_more_stop_sequences_than_the_provider_allows(key):
    with pytest.raises(ConfigValidationError, match="stop"):
        _for("openai", **{key: _FIVE_STOPS})


def test_openai_accepts_exactly_the_maximum_number_of_stop_sequences():
    _for("openai", translation__sampling__stop=_FIVE_STOPS[:MAX_OPENAI_STOP_SEQUENCES])


def test_ollama_has_no_stop_sequence_cap():
    _for("ollama", translation__sampling__stop=_FIVE_STOPS, verification__sampling__stop=_FIVE_STOPS)


# ============================================================================
# ConfigLocations
# ============================================================================

def test_config_files_default_to_the_program_directory(tmp_path):
    locations = ConfigLocations.resolve(str(tmp_path), environ={})

    assert locations.conductor_path == str(tmp_path / "conductor.toml")
    assert locations.llm_path == str(tmp_path / "llm.toml")


def test_llmc_config_moves_conductor_toml_and_the_default_profile(tmp_path):
    override = tmp_path / "elsewhere" / "conductor.toml"

    locations = ConfigLocations.resolve(str(tmp_path), environ={CONFIG_PATH_ENV_VAR: str(override)})

    assert locations.conductor_path == str(override)
    assert locations.llm_path == str(tmp_path / "elsewhere" / "llm.toml")


def test_llm_config_flag_replaces_only_the_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    locations = ConfigLocations.resolve(str(tmp_path / "app"), llm_override="profiles/ds.toml", environ={})

    assert locations.conductor_path == str(tmp_path / "app" / "conductor.toml")
    assert locations.llm_path == str(tmp_path / "profiles" / "ds.toml")


def test_resolve_app_dir_prefers_argv0_over_package_module(tmp_path, monkeypatch):
    shim = tmp_path / "bin" / "llm-conductor"
    shim.parent.mkdir()
    shim.write_text("", encoding="utf-8")
    package_main = tmp_path / "site-packages" / "llm_conductor" / "cli" / "main.py"
    package_main.parent.mkdir(parents=True)

    monkeypatch.setattr(sys, "argv", [str(shim), "check"])
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert resolve_app_dir(str(package_main)) == str(shim.parent)


def test_resolve_app_dir_falls_back_to_entry_file_when_argv0_is_not_a_file(
        tmp_path, monkeypatch):
    package_main = tmp_path / "pkg" / "cli" / "main.py"
    package_main.parent.mkdir(parents=True)
    package_main.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["-m", "llm_conductor"])
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert resolve_app_dir(str(package_main)) == str(package_main.parent)


def test_resolve_app_dir_uses_executable_directory_when_frozen(tmp_path, monkeypatch):
    """PyInstaller one-file: configs sit next to the exe, not the unpack dir."""
    exe = tmp_path / "dist" / "llm-conductor.exe"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    package_main = tmp_path / "_MEI" / "llm_conductor" / "cli" / "main.py"
    package_main.parent.mkdir(parents=True)
    package_main.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "argv", [str(package_main)])

    assert resolve_app_dir(str(package_main)) == str(exe.parent)


# ============================================================================
# RunOverrides and the log file name
# ============================================================================

def test_empty_overrides_return_the_same_configuration():
    data = _data()

    assert RunOverrides().apply(data) is data


def test_overrides_replace_only_what_they_name(tmp_path):
    data = _data(cache__max_size=7)

    overridden = RunOverrides(log_level="debug", cache_enabled=False,
                              cache_dir=str(tmp_path / "c")).apply(data)

    assert (overridden.log_level, overridden.cache_enabled, overridden.cache_dir) == \
        ("DEBUG", False, str(tmp_path / "c"))
    assert overridden.is_debug_mode
    assert overridden.cache_max_size == 7
    assert data.log_level == DEFAULT_LOG_LEVEL            # the loaded data is untouched


def test_overridden_values_are_validated():
    with pytest.raises(ConfigValidationError, match="log level"):
        RunOverrides(log_level="loud").apply(_data())


@pytest.mark.parametrize("log_name, expected", [
    ("ui.list", "conductor_ui.list_20261009_130501.log"),
    (None, "conductor_20261009_130501.log"),
])
def test_log_file_name_contains_the_target_and_the_start_time(log_name, expected):
    started = datetime(2026, 10, 9, 13, 5, 1, tzinfo=UTC)

    assert ConfigLogger.build_log_filename(log_name, started) == expected


def test_trace_log_file_name_uses_its_own_prefix():
    started = datetime(2026, 10, 9, 13, 5, 1, tzinfo=UTC)

    assert (ConfigLogger.build_log_filename("ui.list", started, prefix=TRACE_LOG_FILE_PREFIX)
            == "trace_ui.list_20261009_130501.log")


def test_had_errors_reflects_only_error_and_above():
    logger = ConfigLogger.create(log_to_file=False)

    logger.warning("not an error")
    assert logger.had_errors is False

    logger.error("boom")
    assert logger.had_errors is True
