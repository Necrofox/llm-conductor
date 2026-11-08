# -*- coding: utf-8 -*-
"""
Integration tests of the configuration files on disk: the templates of the
'init' command and Config.create_from_files with a real logger.
"""
import logging
import logging.handlers
import tomllib
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_conductor.cli.init import (
    TomlTemplateRenderer,
    create_config_files,
    format_toml_multiline,
    format_toml_value,
)
from llm_conductor.config import Config, ConfigFileLoader, ConfigLogger
from llm_conductor.constants import (
    DEFAULT_GLOSSARY_CACHE_MAX_SIZE,
    DEFAULT_PROFILE_NAME,
    DEFAULT_TRANSLATION_STRICT_PROMPT,
    DEFAULT_VALIDATION_ALLOWED_SCRIPTS,
    DEFAULT_VERIFICATION_SYSTEM_PROMPT,
    EXAMPLE_EN_RU_ALLOWED_SCRIPTS,
    MODE_STRICT,
    MODE_STYLE,
)
from llm_conductor.exceptions import ConfigValidationError, FileOperationError
from llm_conductor.schema import CONDUCTOR_SCHEMA, LLM_SCHEMA, ValueKind
from tests.support import base_llm_doc, deep_merge, split_overrides, write_configs

SCHEMAS = [LLM_SCHEMA, CONDUCTOR_SCHEMA]


def _document_from_init_values(schema):
    """Nested TOML-shaped dict using each setting's init_value (for template checks)."""
    document = {}
    for section in schema.sections:
        table = document
        for name in section.path:
            table = table.setdefault(name, {})
        for setting in section.settings:
            value = setting.init_value
            if setting.kind is ValueKind.STRING_LIST:
                value = list(value)
            table[setting.key] = value
    return document


# ============================================================================
# Templates of 'init'
# ============================================================================

@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.file_name)
def test_template_parses_back_to_init_values(schema):
    """init may write template_default; resolve of the template matches init_value."""
    parsed = tomllib.loads(TomlTemplateRenderer(schema).render())

    assert schema.resolve(parsed) == schema.resolve(_document_from_init_values(schema))


def test_absent_allowed_scripts_resolves_empty_not_example_profile():
    """Schema default stays empty; only the init line uses the EN→RU example."""
    assert LLM_SCHEMA.resolve({})["validation"]["allowed_scripts"] == DEFAULT_VALIDATION_ALLOWED_SCRIPTS
    assert DEFAULT_VALIDATION_ALLOWED_SCRIPTS == ()

    parsed = tomllib.loads(TomlTemplateRenderer(LLM_SCHEMA).render())
    assert tuple(parsed["validation"]["allowed_scripts"]) == EXAMPLE_EN_RU_ALLOWED_SCRIPTS


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.file_name)
def test_template_writes_every_schema_key_explicitly(schema):
    """A key the loader accepts but the template omits is invisible to the user (§10.3 step 10)."""
    parsed = tomllib.loads(TomlTemplateRenderer(schema).render())

    written = set()
    for section in schema.sections:
        table = parsed
        for name in section.path:
            table = table[name]
        written |= {f"{section.name}.{key}" for key in table if not isinstance(table[key], dict)}

    assert written == set(schema.all_keys())


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.file_name)
def test_every_template_key_has_a_comment_above(schema):
    lines = TomlTemplateRenderer(schema).render().splitlines()

    uncommented = [line for previous, line in pairwise(lines)
                   if " = " in line and not line.startswith(("#", " ")) and not previous.startswith("#")]
    assert uncommented == []


def test_glossary_cache_size_is_in_the_conductor_template():
    parsed = tomllib.loads(TomlTemplateRenderer(CONDUCTOR_SCHEMA).render())

    assert parsed["cache"]["glossary_patterns_max_size"] == DEFAULT_GLOSSARY_CACHE_MAX_SIZE


def test_template_prompts_survive_byte_for_byte():
    parsed = tomllib.loads(TomlTemplateRenderer(LLM_SCHEMA).render())

    assert parsed["translation"]["prompts"]["strict"] == DEFAULT_TRANSLATION_STRICT_PROMPT
    assert parsed["verification"]["prompts"]["system"] == DEFAULT_VERIFICATION_SYSTEM_PROMPT


@given(st.text())
def test_any_string_roundtrips_through_both_renderings(text):
    for rendered in (format_toml_value(text), format_toml_multiline(text)):
        assert tomllib.loads(f"v = {rendered}")["v"] == text


@pytest.mark.parametrize("text", ["has ''' inside", "ends with a quote'", "bell\x07", "cr\ronly", "del\x7f"])
def test_multiline_falls_back_to_a_basic_string_when_a_literal_cannot_hold_the_text(text):
    rendered = format_toml_multiline(text)

    assert not rendered.startswith("'''")
    assert tomllib.loads(f"v = {rendered}")["v"] == text


def test_init_writes_both_templates_that_load_into_a_valid_config(tmp_path, capsys):
    conductor, llm = tmp_path / "conductor.toml", tmp_path / "llm.toml"

    create_config_files(str(conductor), str(llm))

    data = ConfigFileLoader(environ={}).load(str(conductor), str(llm))
    assert data.profile_name == DEFAULT_PROFILE_NAME
    assert (data.translation_api.api_type, data.verification_api.api_type) == ("ollama", "ollama")
    assert data.cache_dir == str(tmp_path / "cache")
    assert f"Created {conductor}" in capsys.readouterr().out


def test_init_never_overwrites_an_existing_profile(tmp_path):
    conductor, llm = tmp_path / "conductor.toml", tmp_path / "llm.toml"
    llm.write_text("# hand-tuned profile\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Refusing to overwrite"):
        create_config_files(str(conductor), str(llm))

    assert llm.read_text(encoding="utf-8") == "# hand-tuned profile\n"
    assert not conductor.exists()


# ============================================================================
# Config.create_from_files
# ============================================================================

def _write(tmp_path, **overrides):
    """conductor.toml + llm.toml in tmp_path/app; returns (conductor path, llm path) as str."""
    llm_overrides, conductor_overrides = split_overrides(**overrides)
    conductor = write_configs(tmp_path / "app", deep_merge(base_llm_doc(), llm_overrides), conductor_overrides)
    return str(conductor), str(conductor.parent / "llm.toml")


def test_relative_directories_resolve_from_the_config_file_not_the_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    config = Config.create_from_files(*_write(tmp_path, logging__level="DEBUG"), log_to_file=True)

    assert config.is_debug_mode
    assert (tmp_path / "app" / "cache").is_dir()
    log_files = list((tmp_path / "app" / "logs").glob("conductor_*.log"))
    assert len(log_files) == 1
    assert list(work.iterdir()) == []

    config.logger.debug("debug line reaches the file")
    for handler in logging.getLogger('conductor').handlers:
        handler.flush()
    assert "debug line reaches the file" in log_files[0].read_text(encoding='utf-8')


def test_log_file_rotates_daily_with_the_configured_retention(tmp_path):
    Config.create_from_files(*_write(tmp_path, logging__retention_days=3), log_to_file=True)

    file_handlers = [h for h in logging.getLogger('conductor').handlers
                     if isinstance(h, logging.handlers.TimedRotatingFileHandler)]
    assert len(file_handlers) == 1
    assert (file_handlers[0].when, file_handlers[0].backupCount) == ('MIDNIGHT', 3)


def test_extended_diagnostic_writes_a_separate_trace_file(tmp_path):
    config = Config.create_from_files(*_write(tmp_path, logging__extended_diagnostic=True), log_to_file=True)

    config.logger.trace("full prompt: translate 'gold'")
    config.logger.debug("ordinary debug line")
    for handler in logging.getLogger('conductor').handlers:
        handler.flush()

    log_dir = tmp_path / "app" / "logs"
    trace_files = list(log_dir.glob("trace_*.log"))
    main_files = list(log_dir.glob("conductor_*.log"))
    assert len(trace_files) == 1
    trace_text = trace_files[0].read_text(encoding='utf-8')
    main_text = main_files[0].read_text(encoding='utf-8')
    assert "full prompt: translate 'gold'" in trace_text
    assert "full prompt: translate 'gold'" not in main_text    # TRACE never reaches the main file
    assert "ordinary debug line" in main_text


def test_clean_run_deletes_its_log_file(tmp_path):
    config = Config.create_from_files(*_write(tmp_path), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"
    config.logger.info("all good")

    config.close_log_files()

    assert list(log_dir.glob("conductor_*.log")) == []


def test_run_with_an_error_keeps_its_log_file(tmp_path):
    config = Config.create_from_files(*_write(tmp_path), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"
    config.logger.error("something broke")

    config.close_log_files()

    assert len(list(log_dir.glob("conductor_*.log"))) == 1


def test_always_keep_overrides_a_clean_run(tmp_path):
    config = Config.create_from_files(*_write(tmp_path), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"

    config.close_log_files(always_keep=True)

    assert len(list(log_dir.glob("conductor_*.log"))) == 1


def test_debug_level_keeps_a_clean_runs_log_in_full(tmp_path):
    config = Config.create_from_files(*_write(tmp_path, logging__level="DEBUG"), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"
    config.logger.info("all good")

    config.close_log_files()

    log_files = list(log_dir.glob("conductor_*.log"))
    assert len(log_files) == 1
    assert "all good" in log_files[0].read_text(encoding='utf-8')


def test_extended_diagnostic_keeps_every_file_even_at_a_quiet_level(tmp_path):
    config = Config.create_from_files(*_write(tmp_path, logging__extended_diagnostic=True), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"
    config.logger.info("all good")
    config.logger.trace("full prompt: translate 'gold'")

    config.close_log_files()

    assert len(list(log_dir.glob("conductor_*.log"))) == 1
    assert len(list(log_dir.glob("trace_*.log"))) == 1


def test_verbose_level_replaces_a_clean_runs_log_with_a_summary(tmp_path):
    config = Config.create_from_files(*_write(tmp_path, logging__level="VERBOSE"), log_to_file=True)
    log_dir = tmp_path / "app" / "logs"
    config.logger.debug("full prompt that would otherwise flood the log")

    config.close_log_files()

    log_files = list(log_dir.glob("conductor_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding='utf-8')
    assert "full prompt that would otherwise flood the log" not in content
    assert "RUN SUMMARY" in content
    assert config.profile_name in content


def test_config_logger_close_log_files_header_replaces_a_clean_runs_content(tmp_path):
    logger = ConfigLogger.create(log_to_file=True, log_dir=str(tmp_path))
    logger.info("detail that the header is about to replace")

    logger.close_log_files(always_keep=False, header="condensed summary")

    log_files = list(tmp_path.glob("conductor_*.log"))
    assert len(log_files) == 1
    assert log_files[0].read_text(encoding='utf-8') == "condensed summary"


def test_config_logger_close_log_files_keep_everything_ignores_header(tmp_path):
    logger = ConfigLogger.create(log_to_file=True, log_dir=str(tmp_path))
    logger.info("detail that must survive")

    logger.close_log_files(always_keep=True, header="should not be written")

    log_files = list(tmp_path.glob("conductor_*.log"))
    assert len(log_files) == 1
    assert "detail that must survive" in log_files[0].read_text(encoding='utf-8')
    assert "should not be written" not in log_files[0].read_text(encoding='utf-8')


def test_create_from_files_with_injected_factories(tmp_path):
    config = Config.create_from_files(
        *_write(tmp_path), log_to_file=False,
        metrics_factory=lambda: "metrics",
        cache_manager_factory=lambda c: "cache",
        glossary_cache_factory=lambda n: "glossary",
    )

    assert (config.metrics, config.cache_manager, config.glossary_patterns_cache) == (
        "metrics", "cache", "glossary")
    assert not (tmp_path / "app" / "cache").exists()      # no real CacheManager was built


@pytest.mark.parametrize("missing", ["conductor.toml", "llm.toml"])
def test_missing_file_raises(tmp_path, missing):
    paths = _write(tmp_path)
    (tmp_path / "app" / missing).unlink()

    with pytest.raises(FileOperationError, match="not found"):
        Config.create_from_files(*paths, log_to_file=False)


def test_broken_toml_names_the_file(tmp_path):
    conductor, llm = _write(tmp_path)
    (tmp_path / "app" / "llm.toml").write_text("[translation\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="not valid TOML") as error:
        Config.create_from_files(conductor, llm, log_to_file=False)
    assert "llm.toml" in str(error.value)


def test_misspelled_key_is_rejected_with_a_suggestion(tmp_path):
    paths = _write(tmp_path, translation__sampling__strict__tempreature=0.1)

    with pytest.raises(ConfigValidationError, match="did you mean 'temperature'"):
        Config.create_from_files(*paths, log_to_file=False)


@pytest.mark.parametrize("overrides, message", [
    ({"translation__api__type": "openai"}, "api_key is required"),
    ({"cache__max_size": 0}, "cache_max_size"),
    ({"cache__glossary_patterns_max_size": 0}, "glossary_patterns_max_size"),
    ({"limits__file_operation_timeout": 0}, "file_operation_timeout"),
    ({"logging__level": "LOUD"}, "log level"),
    ({"logging__retention_days": 0}, "log_retention_days"),
    ({"resilience__max_consecutive_failures": -1}, "max_consecutive_failures"),
    ({"resilience__max_failure_ratio": 1.5}, "max_failure_ratio"),
    ({"heartbeat__interval": -1}, "heartbeat.interval"),
    ({"validation__max_length_ratio": -1.0}, "max_length_ratio"),
    ({"validation__max_length_slack": -1}, "max_length_slack"),
    ({"validation__latin_text_policy": "ignore"}, "latin_text_policy"),
    ({"validation__allowed_scripts": ["runic"]}, "allowed_scripts"),
    ({"translation__api__max_tokens_param": "max_output_tokens"}, "max_tokens_param"),
])
def test_invalid_values_raise_validation_error(tmp_path, overrides, message):
    with pytest.raises(ConfigValidationError, match=message):
        Config.create_from_files(*_write(tmp_path, **overrides), log_to_file=False)


def test_allowed_scripts_is_read_from_llm_toml(tmp_path):
    config = Config.create_from_files(
        *_write(tmp_path, validation__allowed_scripts=["latin", "cjk"]),
        log_to_file=False)

    assert config.validation.allowed_scripts == ("latin", "cjk")


def test_limits_reach_the_components(tmp_path):
    config = Config.create_from_files(
        *_write(tmp_path, limits__file_operation_timeout=7, limits__large_file_threshold_mb=3,
                limits__max_recursion_depth=4),
        log_to_file=False)

    assert (config.file_ops.timeout, config.file_ops.large_file_threshold_mb) == (7, 3)
    assert config.max_recursion_depth == 4


def test_key_env_takes_priority_over_the_literal_key(tmp_path):
    paths = _write(tmp_path, translation__api__type="openai", translation__api__key_env="LLMC_TEST_KEY",
                   translation__api__key="sk-literal")

    data = ConfigFileLoader(environ={"LLMC_TEST_KEY": "sk-from-env"}).load(*paths)

    assert data.translation_api.api_key == "sk-from-env"


def test_unset_key_env_falls_back_to_the_literal_key(tmp_path):
    paths = _write(tmp_path, translation__api__type="openai", translation__api__key_env="LLMC_TEST_KEY",
                   translation__api__key="sk-literal")

    assert ConfigFileLoader(environ={}).load(*paths).translation_api.api_key == "sk-literal"


def test_openai_without_any_key_names_the_missing_variable(tmp_path):
    paths = _write(tmp_path, translation__api__type="openai", translation__api__key_env="LLMC_TEST_KEY")

    with pytest.raises(ConfigValidationError, match="'LLMC_TEST_KEY' is unset or empty"):
        ConfigFileLoader(environ={}).load(*paths)


def test_profile_without_keep_alive_loads_with_empty_default(tmp_path):
    """Profiles written before keep_alive existed must still load unchanged."""
    config = Config.create_from_files(*_write(tmp_path), log_to_file=False)

    assert config.translation_api.keep_alive == ''
    assert config.verification_api.keep_alive == ''
    assert config.get_translation_config_for_mode(MODE_STRICT)['keep_alive'] == ''


def test_keep_alive_from_llm_toml_reaches_runtime_config(tmp_path):
    config = Config.create_from_files(
        *_write(tmp_path, translation__api__keep_alive="5m", verification__api__keep_alive="0"),
        log_to_file=False)

    assert config.translation_api.keep_alive == "5m"
    assert config.get_translation_config_for_mode(MODE_STYLE)['keep_alive'] == "5m"
    assert config.get_verification_config()['keep_alive'] == "0"


def test_profile_without_max_tokens_param_loads_with_max_tokens_default(tmp_path):
    """Profiles written before max_tokens_param existed must still load unchanged."""
    config = Config.create_from_files(*_write(tmp_path), log_to_file=False)

    assert config.translation_api.max_tokens_param == 'max_tokens'
    assert config.verification_api.max_tokens_param == 'max_tokens'
    assert config.get_translation_config_for_mode(MODE_STRICT)['max_tokens_param'] == 'max_tokens'


def test_max_tokens_param_from_llm_toml_reaches_runtime_config(tmp_path):
    config = Config.create_from_files(
        *_write(tmp_path,
                translation__api__max_tokens_param="max_completion_tokens",
                verification__api__max_tokens_param="max_tokens"),
        log_to_file=False)

    assert config.translation_api.max_tokens_param == "max_completion_tokens"
    assert (config.get_translation_config_for_mode(MODE_STYLE)['max_tokens_param']
            == "max_completion_tokens")
    assert config.get_verification_config()['max_tokens_param'] == "max_tokens"
