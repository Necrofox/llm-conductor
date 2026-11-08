# -*- coding: utf-8 -*-
"""
Fixtures of the integration tier: a real Config with real managers on a
temporary directory, and HTTP interception through responses.
"""
import pytest
import responses

from llm_conductor.config import Config, ConfigData
from tests.support import RecordingLogger, base_llm_doc, deep_merge, split_overrides


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    """
    Factory of a real Config (CacheManager, GlossaryPatternCache,
    MetricsCollector from the default factories) built by ConfigData.from_toml.

    cache/log directories and the working directory are moved into tmp_path
    (AGENTS.md). The logger is a RecordingLogger without real handlers;
    key_env lookups see an empty environment.

    Returns:
        Function (**overrides) -> Config. Override names address TOML keys
        with double underscores: translation__api__max_retries=3,
        cache__enabled=True (see tests.support.split_overrides).
    """
    monkeypatch.chdir(tmp_path)

    def _make(**overrides):
        llm_overrides, conductor_overrides = split_overrides(**overrides)
        conductor = deep_merge({
            "logging": {"dir": str(tmp_path / "logs")},
            "cache": {"dir": str(tmp_path / "cache"), "auto_save_interval": 0},
        }, conductor_overrides)
        data = ConfigData.from_toml(deep_merge(base_llm_doc(), llm_overrides), conductor, environ={})
        return Config(data, RecordingLogger(is_debug_mode=data.is_debug_mode))

    return _make


@pytest.fixture
def http():
    """
    Intercept requests for the duration of a test. Any unregistered URL
    raises ConnectionError, so reaching the real network is impossible.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock
