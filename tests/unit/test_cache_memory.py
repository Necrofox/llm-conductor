# -*- coding: utf-8 -*-
"""
Unit tests for in-memory caches only (llm_conductor/core/cache.py):
GlossaryPatternCache, _CacheWriteCoordinator, _OperationCache.

CacheManager (disk cache) lives in the integration tier: its constructor
creates cache_dir, and load()/save() touch files.
"""
import threading

import pytest

from llm_conductor.core.cache import (
    GlossaryPatternCache,
    _CacheWriteCoordinator,
    _OperationCache,
)
from llm_conductor.exceptions import FileDurabilityError
from tests.support import DeadlockGuardLock

# ============================================================================
# GlossaryPatternCache
# ============================================================================

def test_glossary_cache_rejects_non_positive_size():
    with pytest.raises(ValueError):
        GlossaryPatternCache(max_size=0)


def test_get_or_compile_compiles_once_then_hits():
    cache = GlossaryPatternCache(max_size=10)

    first = cache.get_or_compile("gold")
    second = cache.get_or_compile("gold")

    assert first is second
    stats = cache.get_stats()
    assert (stats['hits'], stats['misses'], stats['cache_size']) == (1, 1, 1)


def test_word_boundary_flag_is_part_of_the_key():
    cache = GlossaryPatternCache(max_size=10)

    bounded = cache.get_or_compile("gold", use_word_boundaries=True)
    unbounded = cache.get_or_compile("gold", use_word_boundaries=False)

    assert bounded is not unbounded
    assert bounded.search("goldsmith") is None
    assert unbounded.search("goldsmith") is not None
    assert len(cache) == 2


def test_compiled_pattern_is_case_insensitive_and_escaped():
    cache = GlossaryPatternCache(max_size=10)

    assert cache.get_or_compile("Guild").search("the GUILD hall") is not None
    # re.escape: term metacharacters must not become regex syntax
    assert cache.get_or_compile("a.b", use_word_boundaries=False).search("axb") is None


def test_lru_eviction_drops_least_recently_used():
    cache = GlossaryPatternCache(max_size=2)
    cache.get_or_compile("a")
    cache.get_or_compile("b")
    cache.get_or_compile("a")          # 'a' becomes most recently used

    cache.get_or_compile("c")          # evicts 'b'

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get_stats()['evictions'] == 1


def test_put_rejects_non_pattern():
    with pytest.raises(TypeError):
        GlossaryPatternCache(max_size=2).put("a", "not a pattern")


def test_resize_evicts_down_to_new_limit():
    cache = GlossaryPatternCache(max_size=5)
    for term in "abcde":
        cache.get_or_compile(term)

    cache.resize(2)

    assert len(cache) == 2
    assert cache.get("d") is not None and cache.get("e") is not None


def test_clear_resets_entries_and_metrics():
    cache = GlossaryPatternCache(max_size=5)
    cache.get_or_compile("a")

    cache.clear()

    assert cache.get_stats()['cache_size'] == 0
    assert cache.get_stats()['misses'] == 0


def test_task36_repr_and_len_are_safe_inside_locked_section():
    """repr()/len() must not hang when called from code that already holds cache._lock."""
    cache = GlossaryPatternCache(max_size=5)
    cache._lock = DeadlockGuardLock(timeout=0.2)

    with cache._lock:
        repr(cache)
        len(cache)


# ============================================================================
# _CacheWriteCoordinator
# ============================================================================

def test_coordinator_versions_are_monotonic():
    coordinator = _CacheWriteCoordinator()

    assert [coordinator.next_version() for _ in range(3)] == [1, 2, 3]


def test_coordinator_drops_stale_snapshot_written_after_newer_one():
    """Reentrant signal_handler scenario: v2 lands on disk before v1."""
    coordinator = _CacheWriteCoordinator()
    written = []
    v1 = coordinator.next_version()
    v2 = coordinator.next_version()

    assert coordinator.attempt_write(v2, lambda: written.append("v2")) is True
    assert coordinator.attempt_write(v1, lambda: written.append("v1")) is False
    assert written == ["v2"]


def test_coordinator_failed_write_does_not_advance_version():
    """A failed IO must not block a retry with the same version number."""
    coordinator = _CacheWriteCoordinator()
    version = coordinator.next_version()

    def failing_io():
        raise OSError("disk full")

    with pytest.raises(OSError):
        coordinator.attempt_write(version, failing_io)

    assert coordinator.attempt_write(version, lambda: None) is True


def test_coordinator_durability_failure_prevents_older_snapshot_overwrite():
    coordinator = _CacheWriteCoordinator()
    older = coordinator.next_version()
    replaced = coordinator.next_version()
    written = []

    def replaced_but_not_synced():
        written.append("new")
        raise FileDurabilityError("Replaced cache, directory sync failed")

    with pytest.raises(FileDurabilityError):
        coordinator.attempt_write(replaced, replaced_but_not_synced)
    assert coordinator.attempt_write(older, lambda: written.append("old")) is False
    newer = coordinator.next_version()
    assert coordinator.attempt_write(newer, lambda: written.append("newer")) is True
    assert written == ["new", "newer"]


# ============================================================================
# _OperationCache
# ============================================================================

def _operation_cache(max_size):
    return _OperationCache(max_size=max_size, lock=threading.Lock())


def test_ensure_capacity_keeps_size_exactly_at_limit_after_insert():
    """Unified eviction: after insert the size is exactly max_size, not max_size-1."""
    op = _operation_cache(max_size=3)
    for i in range(4):
        op.ensure_capacity()
        op.data[f"k{i}"] = {'value': i}

    assert list(op.data) == ["k1", "k2", "k3"]


def test_evict_oldest_maintains_value_index():
    op = _operation_cache(max_size=10)
    op.data["k1"] = {'value': "a", 'value_hash': "h"}
    op.data["k2"] = {'value': "b", 'value_hash': "h"}
    op.index_value("h", "k1")
    op.index_value("h", "k2")

    assert op.evict_oldest() == "k1"
    assert op.value_index == {"h": {"k2"}}

    op.evict_oldest()
    assert op.value_index == {}
    assert op.evict_oldest() is None
