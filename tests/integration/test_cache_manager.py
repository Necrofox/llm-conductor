# -*- coding: utf-8 -*-
"""
Integration tests for CacheManager on tmp_path (llm_conductor/core/cache.py, doc/pipeline.md):
full key, load/save, expiry, key version, eviction, secondary index and miss
diagnostics.
"""
import json
import time
from pathlib import Path

import pytest

from llm_conductor.constants import (
    CACHE_KEY_VERSION,
    CACHE_SAMPLING_IDENTITY_FIELDS,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.core.cache import build_sampling_identity

T = OPERATION_TRANSLATION

SAMPLE_RUNTIME = {
    'temp': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'repeat_penalty': 1.1,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'num_predict': 256,
    'num_ctx': 4096,
    'stop': ['\n\n'],
}

SAMPLE_IDENTITY = build_sampling_identity(SAMPLE_RUNTIME)

KEY_PARTS = {
    'operation_type': T,
    'api_url': "http://llm.test/api/chat",
    'model': "qwen",
    'system_prompt': "SYS",
    'enhanced_prompt': "Translate: ",
    'mode': "style",
    'value': "Hello",
    'sampling_identity': SAMPLE_IDENTITY,
}

FINGERPRINT_PARTS = {
    'operation_type': T,
    'api_url': KEY_PARTS['api_url'],
    'model': "qwen",
    'system_prompt': "SYS",
    'prompt_template': "Translate: ",
    'hints_text': "",
    'glossary_string': "",
    'mode': "style",
    'sampling_identity': SAMPLE_IDENTITY,
}


@pytest.fixture
def manager(make_config):
    return make_config().cache_manager


def _cache_file(manager, cache_type=T):
    return manager.get_cache_filename(cache_type)


def _write_cache_file(manager, entries, cache_type=T):
    with open(_cache_file(manager, cache_type), 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False)


def _entry(value, timestamp=None, version=CACHE_KEY_VERSION):
    entry = {'value': value, 'timestamp': time.time() if timestamp is None else timestamp}
    if version is not None:
        entry['cache_key_version'] = version
    return entry


def _store(manager, value="Hello", translated="Привет", **key_overrides):
    """Store an entry the same way _process_leaf does: key + fingerprints + source_value."""
    key = manager.create_cache_key(**{**KEY_PARTS, 'value': value, **key_overrides})
    fingerprint_overrides = {k: v for k, v in key_overrides.items() if k in FINGERPRINT_PARTS}
    fingerprints = manager.compute_fingerprints(**{**FINGERPRINT_PARTS, **fingerprint_overrides})
    manager.set(T, key, translated, {'fingerprints': fingerprints}, source_value=value)
    return key


# ============================================================================
# Full cache key
# ============================================================================

def test_cache_key_is_deterministic_md5(manager):
    key = manager.create_cache_key(**KEY_PARTS)

    assert key == manager.create_cache_key(**KEY_PARTS)
    assert len(key) == 32 and int(key, 16) >= 0


@pytest.mark.parametrize("component", sorted(KEY_PARTS))
def test_every_request_component_changes_the_key(manager, component):
    """Regression: model/prompts/hints/sampling used not to be part of the key."""
    changed = {**KEY_PARTS, component: KEY_PARTS[component] + "_changed"}

    assert manager.create_cache_key(**changed) != manager.create_cache_key(**KEY_PARTS)


def test_allowed_scripts_separate_cached_replies(manager):
    key = manager.create_cache_key(**KEY_PARTS, allowed_scripts=("cyrillic", "latin"))
    assert key == manager.create_cache_key(**KEY_PARTS, allowed_scripts=("latin", "cyrillic"))
    assert key != manager.create_cache_key(**KEY_PARTS, allowed_scripts=("latin",))


@pytest.mark.parametrize("field_name,runtime_key", CACHE_SAMPLING_IDENTITY_FIELDS)
def test_each_sampling_field_changes_the_key(manager, field_name, runtime_key):
    """Issue #12: changing any generation-affecting param must miss the cache."""
    altered = dict(SAMPLE_RUNTIME)
    if field_name == 'stop':
        altered[runtime_key] = list(altered[runtime_key]) + ['</s>']
    elif isinstance(altered[runtime_key], float):
        altered[runtime_key] = altered[runtime_key] + 0.05
    else:
        altered[runtime_key] = altered[runtime_key] + 1

    changed_identity = build_sampling_identity(altered)
    assert changed_identity != SAMPLE_IDENTITY
    assert manager.create_cache_key(
        **{**KEY_PARTS, 'sampling_identity': changed_identity}
    ) != manager.create_cache_key(**KEY_PARTS)


def test_restoring_sampling_parameters_restores_the_key(manager):
    """Issue #12 acceptance: same sampling again → same key (cache hit path)."""
    again = build_sampling_identity(dict(SAMPLE_RUNTIME))
    assert again == SAMPLE_IDENTITY
    assert manager.create_cache_key(
        **{**KEY_PARTS, 'sampling_identity': again}
    ) == manager.create_cache_key(**KEY_PARTS)


def test_fingerprints_cover_all_components_with_fixed_length(manager):
    fingerprints = manager.compute_fingerprints(**FINGERPRINT_PARTS)

    assert set(fingerprints) == {
        'cache_key_version', 'operation_type', 'model', 'system_prompt',
        'prompt_template', 'hints', 'glossary', 'mode', 'sampling',
        'api_url', 'allowed_scripts',
    }
    assert {len(fp) for fp in fingerprints.values()} == {12}


# ============================================================================
# get / set / delete
# ============================================================================

def test_set_then_get_counts_hits_and_misses(manager):
    manager.set(T, "k", "Привет")

    assert manager.get(T, "k") == "Привет"
    assert manager.get(T, "missing") is None
    stats = manager.get_stats(T)
    assert (stats['hits'], stats['misses'], stats['cache_size']) == (1, 1, 1)


def test_get_returns_deep_copy_of_mutable_value(manager):
    manager.set(T, "k", {"nested": ["a"]})

    manager.get(T, "k")["nested"].append("mutated")

    assert manager.get(T, "k") == {"nested": ["a"]}


def test_translation_and_verification_caches_are_independent(manager):
    manager.set(T, "k", "перевод")

    assert manager.get(OPERATION_VERIFICATION, "k") is None


def test_set_respects_max_size_fifo(make_config):
    manager = make_config(cache__max_size=2).cache_manager
    for key in ("a", "b", "c"):
        manager.set(T, key, key)

    assert manager.get(T, "a") is None
    assert (manager.get(T, "b"), manager.get(T, "c")) == ("b", "c")


def test_delete_removes_entry_and_its_value_index(manager):
    key = _store(manager)

    assert manager.delete(T, key) is True
    assert manager.delete(T, key) is False
    assert manager.get(T, key) is None
    assert manager.diagnose_miss(T, "Hello", manager.compute_fingerprints(**FINGERPRINT_PARTS)) == 'new_value'


def test_unknown_cache_type_is_rejected_gracefully(manager):
    manager.set("summaries", "k", "v")

    assert manager.get("summaries", "k") is None
    assert manager.load("summaries") is False
    assert manager.logger.contains("Invalid cache type: summaries", level='ERROR')


# ============================================================================
# save / load
# ============================================================================

def test_save_then_load_in_fresh_manager_restores_entries(make_config):
    first = make_config().cache_manager
    key = _store(first)
    assert first.save(T) is True

    second = make_config().cache_manager
    assert second.load(T) is True

    assert second.get(T, key) == "Привет"


def test_saved_entries_carry_key_version_and_value_hash(manager):
    _store(manager)
    manager.save(T)

    with open(_cache_file(manager), encoding='utf-8') as f:
        (entry,) = json.load(f).values()
    assert entry['cache_key_version'] == CACHE_KEY_VERSION
    assert entry['value'] == "Привет"
    assert 'value_hash' in entry and 'fingerprints' in entry


def test_load_without_file_starts_empty(manager):
    assert manager.load(T) is True
    assert manager.get_stats(T)['cache_size'] == 0


def test_disabled_cache_neither_creates_reads_nor_writes_the_disk(make_config, tmp_path):
    """[cache].enabled = false and --no-cache: the cache directory is never touched."""
    enabled = make_config().cache_manager
    _store(enabled)
    enabled.save(T)
    cache_file = Path(_cache_file(enabled))
    before = cache_file.read_text(encoding='utf-8')

    disabled = make_config(cache__enabled=False).cache_manager
    assert disabled.load(T) is True
    assert disabled.get_stats(T)['cache_size'] == 0            # the file on disk was not read
    _store(disabled, value="Bye", translated="Пока")
    assert disabled.save(T) is True

    assert cache_file.read_text(encoding='utf-8') == before    # and not overwritten


def test_disabled_cache_does_not_create_its_directory(make_config, tmp_path):
    make_config(cache__enabled=False, cache__dir=str(tmp_path / "never"))

    assert not (tmp_path / "never").exists()


def test_load_drops_expired_entries(make_config):
    manager = make_config(cache__expiry_days=30).cache_manager
    _write_cache_file(manager, {
        "fresh": _entry("свежий"),
        "old": _entry("старый", timestamp=time.time() - 31 * 86400),
    })

    manager.load(T)

    assert manager.get(T, "fresh") == "свежий"
    assert manager.get(T, "old") is None


def test_load_discards_entries_of_previous_key_format(manager):
    """Entries missing or with a stale cache_key_version are dropped with an INFO line."""
    _write_cache_file(manager, {
        "current": _entry("ok"),
        "no_version": _entry("old", version=None),
        "old_version": _entry("old", version=CACHE_KEY_VERSION - 1),
        "bare_legacy": "голое значение",
    })

    manager.load(T)

    assert manager.get_stats(T)['cache_size'] == 1
    assert manager.get(T, "current") == "ok"
    assert manager.logger.contains("Discarded 3 cache entries with outdated cache key format", level='INFO')


def test_load_respects_max_size_keeping_last_entries(make_config):
    """Unified eviction: after load the size is exactly max_size."""
    manager = make_config(cache__max_size=3).cache_manager
    _write_cache_file(manager, {f"k{i}": _entry(str(i)) for i in range(5)})

    manager.load(T)

    assert manager.get_stats(T)['cache_size'] == 3
    assert [manager.get(T, f"k{i}") for i in range(5)] == [None, None, "2", "3", "4"]


@pytest.mark.parametrize("content", ['["not", "a", "dict"]', '{broken json'])
def test_load_rejects_corrupted_file_without_raising(manager, content):
    with open(_cache_file(manager), 'w', encoding='utf-8') as f:
        f.write(content)

    assert manager.load(T) is False
    assert manager.get_stats(T)['cache_size'] == 0


def test_save_purges_entries_expired_in_memory(make_config, monkeypatch):
    manager = make_config(cache__expiry_days=1).cache_manager
    manager.set(T, "k", "v")
    real_time = time.time
    monkeypatch.setattr(time, 'time', lambda: real_time() + 2 * 86400)

    manager.save(T)

    assert manager.get(T, "k") is None
    with open(_cache_file(manager), encoding='utf-8') as f:
        assert json.load(f) == {}


def test_save_failure_is_logged_and_returns_false(manager, monkeypatch):
    def failing_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manager.file_ops, 'write_json_atomic', failing_write)
    manager.set(T, "k", "v")

    assert manager.save(T) is False
    assert manager.logger.contains("disk full", level='WARNING')


# ============================================================================
# Miss diagnostics
# ============================================================================

def _miss_reason(manager, value="Hello", **fingerprint_overrides):
    fingerprints = manager.compute_fingerprints(**{**FINGERPRINT_PARTS, **fingerprint_overrides})
    return manager.diagnose_miss(T, value, fingerprints)


def test_diagnose_new_value(manager):
    _store(manager, value="Hello")

    assert _miss_reason(manager, value="Goodbye") == 'new_value'


def test_diagnose_names_changed_components(manager):
    _store(manager)

    assert _miss_reason(manager, model="other") == 'model'
    assert _miss_reason(manager, system_prompt="SYS2", hints_text="Use MASCULINE") == 'hints, system_prompt'
    assert _miss_reason(manager, sampling_identity=SAMPLE_IDENTITY + "_x") == 'sampling'


def test_diagnose_prompt_assembly_when_all_fingerprints_match(manager):
    """All fingerprints matched but the full key diverged — signal to extend the fingerprint list."""
    _store(manager)

    assert _miss_reason(manager) == 'prompt_assembly'


def test_diagnose_survives_reload_from_disk(make_config):
    first = make_config().cache_manager
    _store(first)
    first.save(T)
    second = make_config().cache_manager
    second.load(T)

    assert _miss_reason(second, model="other") == 'model'


def test_miss_reasons_are_summarised_in_stats(manager):
    _store(manager)
    manager.get(T, "unknown-key")
    _miss_reason(manager, model="other")
    _miss_reason(manager, model="other")
    _miss_reason(manager, value="Bye")

    manager.print_stats(T)

    assert manager.get_stats(T)['miss_reasons'] == {'model': 2, 'new_value': 1}
    assert manager.logger.contains("cache misses 3: model — 2, new_value — 1", level='INFO')
