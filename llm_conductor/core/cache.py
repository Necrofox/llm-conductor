# -*- coding: utf-8 -*-
import copy
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Set

from llm_conductor.constants import (
    CACHE_FINGERPRINT_LENGTH,
    CACHE_KEY_VERSION,
    CACHE_SAMPLING_IDENTITY_FIELDS,
    DEFAULT_GLOSSARY_CACHE_MAX_SIZE,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.core.io import safe_read_file
from llm_conductor.interfaces import CacheConfigProtocol


def build_sampling_identity(runtime_config: Mapping[str, Any]) -> str:
    """
    Canonical JSON of generation-affecting sampling parameters.

    Used as one component of the cache key and as the ``sampling`` fingerprint.
    Field list: ``CACHE_SAMPLING_IDENTITY_FIELDS`` in constants (see also
    ``doc/pipeline.md``). ``stop`` order is preserved; outer keys are sorted.

    Args:
        runtime_config: Flattened mode config from ConfigBuilder (``temp``,
            ``top_p``, …, ``stop``).

    Returns:
        Stable JSON string suitable for hashing.
    """
    payload: Dict[str, Any] = {}
    for name, runtime_key in CACHE_SAMPLING_IDENTITY_FIELDS:
        value = runtime_config[runtime_key]
        if name == 'stop':
            value = list(value) if value is not None else []
        payload[name] = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class GlossaryPatternCache:
    """
    LRU cache of compiled glossary regex patterns.

    Uses an OrderedDict for LRU eviction, tracks hit/miss counters and is
    thread-safe through a single non-reentrant threading.Lock.

    Thread-safety contract: get_stats() takes self._lock for a consistent
    snapshot. __len__() and __repr__() are deliberately lock-free instead -
    debug output has no reason to risk a deadlock (self._lock is not
    reentrant, so calling them from code that already holds it would hang
    forever) or to block on it; they may rarely race with a concurrent
    mutation and read a size or counters a moment out of sync, which is
    acceptable for a size check or a repr(). Code that needs the same
    fields while already holding the lock calls the private
    _stats_unlocked() directly, the same way __repr__() does.
    """

    def __init__(self, max_size: int = DEFAULT_GLOSSARY_CACHE_MAX_SIZE):
        """
        Args:
            max_size: Maximum number of compiled patterns to keep.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self._cache: OrderedDict[str, re.Pattern] = OrderedDict()
        self._max_size: int = max_size
        self._lock: threading.Lock = threading.Lock()

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def get(self, term: str, use_word_boundaries: bool = True) -> Optional[re.Pattern]:
        """
        Return a compiled pattern from the LRU cache, or None on a miss.

        A hit moves the entry to the most-recently-used end.

        Args:
            term: Glossary term.
            use_word_boundaries: Whether the cached pattern used word boundaries.

        Returns:
            Compiled re.Pattern, or None if the term is not cached.
        """
        cache_key = self._build_cache_key(term, use_word_boundaries)

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                self._hits += 1
                return self._cache[cache_key]

            self._misses += 1
            return None

    def put(self, term: str, pattern: re.Pattern, use_word_boundaries: bool = True) -> None:
        """
        Insert a compiled pattern, evicting the oldest entry when full.

        Args:
            term: Glossary term.
            pattern: Compiled regex pattern.
            use_word_boundaries: Whether the pattern is anchored to word boundaries.
        """
        if not isinstance(pattern, re.Pattern):
            raise TypeError(f"pattern must be re.Pattern, got {type(pattern).__name__}")

        cache_key = self._build_cache_key(term, use_word_boundaries)

        with self._lock:
            # Another thread may have inserted the same key between get() and put().
            if cache_key in self._cache:
                self._cache[cache_key] = pattern
                self._cache.move_to_end(cache_key)
                return

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[cache_key] = pattern

    def get_or_compile(self, term: str, use_word_boundaries: bool = True) -> re.Pattern:
        """
        Get a compiled pattern from the cache, or compile and cache a new one.

        The main entry point for the cache; compilation and storage are
        handled automatically. Thread-safe.

        Args:
            term: Glossary term.
            use_word_boundaries: Whether to anchor the term to non-word
                boundaries.

        Returns:
            Compiled re.Pattern.
        """
        pattern = self.get(term, use_word_boundaries)
        if pattern is not None:
            return pattern

        if use_word_boundaries:
            # Lookaround, not \b: \b never matches next to punctuation, so a
            # term with a non-word character at either edge (C++, Dr., .NET)
            # would otherwise never be found.
            pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(term), re.IGNORECASE)

        self.put(term, pattern, use_word_boundaries)

        return pattern

    def clear(self) -> None:
        """Clear all patterns and reset hit/miss/eviction counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def get_stats(self) -> dict:
        """
        Return usage statistics of the cache.

        Returns:
            dict with hits, misses, hit_rate, size and evictions.
        """
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict:
        """Same as get_stats(), for callers that already hold self._lock."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_size': self._max_size,
            'evictions': self._evictions
        }

    def resize(self, new_max_size: int) -> None:
        """
        Change the capacity, evicting oldest entries that no longer fit.

        Args:
            new_max_size: New maximum number of patterns.
        """
        if new_max_size <= 0:
            raise ValueError(f"new_max_size must be positive, got {new_max_size}")

        with self._lock:
            self._max_size = new_max_size

            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def _build_cache_key(self, term: str, use_word_boundaries: bool) -> str:
        """
        Build the OrderedDict key from the term and the boundary flag.

        Args:
            term: Glossary term.
            use_word_boundaries: Whether the pattern uses word boundaries.

        Returns:
            Cache key string (`wb:` / `nb:` prefix + term).
        """
        # Prefix keeps the same term with and without boundaries as distinct entries.
        prefix = "wb:" if use_word_boundaries else "nb:"
        return f"{prefix}{term}"

    def __len__(self) -> int:
        """Current number of cached patterns. Lock-free - see the class docstring."""
        return len(self._cache)

    def __repr__(self) -> str:
        """Debug representation. Lock-free (_stats_unlocked()) - see the class docstring."""
        stats = self._stats_unlocked()
        return (
            f"GlossaryPatternCache(size={stats['cache_size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1f}%)"
        )

class _CacheWriteCoordinator:
    """
    Orders disk writes of one cache type so a newer snapshot is never
    overwritten by an older one.

    CacheManager.save() builds the snapshot under the shared data lock and
    performs IO outside it (see ``doc/pipeline.md``).
    Two save() calls can therefore leave the snapshot section in one order
    and reach disk in the reverse order. A monotonic snapshot version plus a
    write lock separate from the data lock prevent that:

      - next_version() allocates the number in snapshot-freshness order
        (caller must hold the data lock).
      - attempt_write() serializes the IO and silently drops a snapshot if a
        newer one is already on disk — expected reordering protection, not
        an error.

    One instance per cache_type (not merged into CacheManager._lock): this
    serializes IO of distinct on-disk files, not shared in-memory data.
    On the current single-threaded pipeline the practical reordering source
    is a signal handler re-entering save() between snapshot and write.
    """

    def __init__(self):
        self._version_lock = threading.Lock()
        self._next_version = 0

        self._write_lock = threading.Lock()
        self._last_written_version = 0

    def next_version(self) -> int:
        """
        Allocate the next monotonic snapshot number.

        Returns:
            A version unique and greater than every previous allocation.
        """
        with self._version_lock:
            self._next_version += 1
            return self._next_version

    def attempt_write(self, version: int, io_func: Callable[[], None]) -> bool:
        """
        Run io_func() if version is newer than anything already written.

        Args:
            version: Snapshot number from next_version().
            io_func: Zero-argument side effect that writes to disk.

        Returns:
            True if io_func() ran; False if the snapshot was dropped as stale
            (io_func() was not called).

        Raises:
            Any exception from io_func() is re-raised unchanged;
            _last_written_version is left untouched so a retry with the same
            or a newer version is not wrongly treated as stale.
        """
        with self._write_lock:
            if version <= self._last_written_version:
                # A newer snapshot is already on disk — this write is stale.
                return False
            io_func()
            self._last_written_version = version
            return True


class _OperationCache:
    """
    Cache of one operation type (translation or verification).

    Encapsulates its data, hit/miss counters and size limit in one object,
    the same way GlossaryPatternCache does above. The lock is injected:
    CacheManager builds a single threading.Lock shared by both operations
    (translation and verification never run concurrently - see
    CacheManager), so this class holds the lock it was given rather than
    owning one of its own.

    value_index and miss_reasons are cache-miss diagnostics: value_index
    maps md5(value) to the set of cache keys that share that value (so a
    miss can look up candidates without scanning the whole cache),
    miss_reasons accumulates miss causes for the print_stats() summary.

    ensure_capacity()/evict_oldest() are the single eviction implementation,
    used by both set() and load().

    Thread-safety contract: get_stats() takes `lock`. Code that already
    holds it (CacheManager.load()/save()/set()) must call the private
    _stats_unlocked() instead - `lock` is not reentrant.
    """

    def __init__(self, max_size: int, lock: threading.Lock):
        self.data: 'OrderedDict[str, dict]' = OrderedDict()
        self.max_size = max_size
        self.lock = lock

        self.hits: int = 0
        self.misses: int = 0

        # md5(value) -> set of cache keys that share that source value.
        self.value_index: Dict[str, Set[str]] = {}
        # miss reason -> count (aggregated for print_stats()).
        self.miss_reasons: Dict[str, int] = {}

    def index_value(self, value_hash: str, cache_key: str) -> None:
        """Register cache_key under value_hash in the secondary index."""
        self.value_index.setdefault(value_hash, set()).add(cache_key)

    def deindex_key(self, cache_key: str, value_hash: Optional[str]) -> None:
        """Remove cache_key from the secondary index for value_hash."""
        if not value_hash:
            return
        keys = self.value_index.get(value_hash)
        if keys is None:
            return
        keys.discard(cache_key)
        if not keys:
            del self.value_index[value_hash]

    def evict_oldest(self) -> Optional[str]:
        """
        Evict the oldest entry (FIFO) and keep value_index in sync.

        Returns:
            Evicted cache key, or None if the cache is empty.
        """
        if not self.data:
            return None
        cache_key, entry = self.data.popitem(last=False)
        value_hash = entry.get('value_hash') if isinstance(entry, dict) else None
        self.deindex_key(cache_key, value_hash)
        return cache_key

    def ensure_capacity(self) -> int:
        """
        Evict until there is room for one more entry under max_size.

        Called before every insert from both set() and load() — the single
        size-limit implementation for this class (see the class docstring).

        Returns:
            Number of entries evicted.
        """
        evicted = 0
        while len(self.data) >= self.max_size:
            self.evict_oldest()
            evicted += 1
        return evicted

    def get_stats(self) -> dict:
        """Return cache statistics, including a breakdown of miss reasons."""
        with self.lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict:
        """Same as get_stats(), for callers that already hold `lock`."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.data),
            'miss_reasons': dict(self.miss_reasons),
        }


class CacheManager:
    """
    Disk cache of model replies.

    A facade over one _OperationCache per OPERATION_TRANSLATION /
    OPERATION_VERIFICATION, each encapsulating its data, statistics and size
    limit; callers address an operation by its cache_type string. With
    cache_enabled off nothing is read from or written to the disk.

    Thread-safety contract: every public method takes self._lock on its
    own and returns; none of them calls another public method of this
    class, or of the _OperationCache it delegates to, while still holding
    the lock - self._lock is not reentrant.
    """

    def __init__(self, config: CacheConfigProtocol):
        """
        Args:
            config: Cache settings and services - logger, size and expiry
                limits, cache_dir, file_ops (used by load()/save()) and
                cache_enabled. A disabled cache never touches the disk:
                load()/save() do nothing and cache_dir is not created.
        """
        self.logger = config.logger
        self.cache_max_size = config.cache_max_size
        self.cache_expiry_days = config.cache_expiry_days
        self.is_debug = config.is_debug_mode
        self.cache_dir = config.cache_dir
        self.file_ops = config.file_ops
        self.persistent = config.cache_enabled

        if self.persistent:
            os.makedirs(self.cache_dir, exist_ok=True)

        # One lock for both operations: translation and verification never
        # run concurrently (single-threaded pipeline today; see doc/pipeline.md).
        self._lock = threading.Lock()

        self._operations: Dict[str, _OperationCache] = {
            OPERATION_TRANSLATION: _OperationCache(self.cache_max_size, self._lock),
            OPERATION_VERIFICATION: _OperationCache(self.cache_max_size, self._lock),
        }

        # One write coordinator per file: it orders disk writes of the same
        # cache_type and is independent of _lock, which guards memory only.
        self._write_coordinators = {
            OPERATION_TRANSLATION: _CacheWriteCoordinator(),
            OPERATION_VERIFICATION: _CacheWriteCoordinator()
        }

    def _get_operation(self, cache_type: str) -> Optional[_OperationCache]:
        """Return the _OperationCache for cache_type, or None after logging an error."""
        op = self._operations.get(cache_type)
        if op is None:
            self.logger.error(f"Invalid cache type: {cache_type}")
        return op

    @staticmethod
    def _hash_value(value: str) -> str:
        """MD5 of value — secondary-index key used by miss diagnostics."""
        return hashlib.md5(value.encode('utf-8')).hexdigest()

    def get_cache_filename(self, cache_type: str) -> str:
        """Return the on-disk path for cache_type under self.cache_dir."""
        return os.path.join(self.cache_dir, f"{cache_type}_cache.json")

    def load(self, cache_type: str) -> bool:
        """
        Load one cache file into memory, thread-safely.

        The read and the replacement of the in-memory data happen under one
        lock. Expired entries, malformed entries and entries without the
        current CACHE_KEY_VERSION are dropped; a missing file means an empty
        cache. A disabled (non-persistent) cache reads nothing.

        Args:
            cache_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.

        Returns:
            True on success, including a missing file and a disabled cache;
            False on an unknown cache type or an unreadable file.
        """
        op = self._get_operation(cache_type)
        if op is None:
            return False

        if not self.persistent:
            if self.is_debug:
                self.logger.debug(f"Disk cache disabled, {cache_type} cache is not loaded.")
            return True

        with self._lock:
            cache_filename = self.get_cache_filename(cache_type)

            if not os.path.exists(cache_filename):
                if self.is_debug:
                    self.logger.debug(
                        f"Cache file {cache_filename} does not exist, starting with empty cache."
                    )
                return True

            # critical=False: safe_read_file has already logged the failure.
            data = safe_read_file(cache_filename, f"{cache_type} cache file", self.file_ops,
                                   self.logger, critical=False)
            if data is None:
                return False

            if not isinstance(data, dict):
                self.logger.warning(
                    f"Cache file {cache_filename} contains invalid format "
                    f"(expected dict, got {type(data).__name__}). Starting with empty cache."
                )
                return False

            # Entries are collected into a staging _OperationCache, which reuses
            # the eviction of set(); its data and value_index then replace the
            # live ones without resetting the hit/miss counters.
            current_time = time.time()
            expiry_seconds = self.cache_expiry_days * 86400
            staging = _OperationCache(self.cache_max_size, self._lock)
            loaded_count = 0
            expired_count = 0
            invalid_count = 0
            stale_version_count = 0

            for hash_key, v in data.items():
                if not isinstance(hash_key, str):
                    invalid_count += 1
                    continue

                if isinstance(v, dict):
                    if 'timestamp' in v:
                        if current_time - v['timestamp'] < expiry_seconds:
                            entry = v
                        else:
                            expired_count += 1
                            continue
                    elif 'value' in v:
                        entry = {**v, 'timestamp': current_time}
                    else:
                        invalid_count += 1
                        continue
                else:
                    entry = {'value': v, 'timestamp': current_time}

                # An entry of another key format may answer a different
                # request under the same key - never a safe hit.
                if entry.get('cache_key_version') != CACHE_KEY_VERSION:
                    stale_version_count += 1
                    continue

                evicted = staging.ensure_capacity()
                if evicted and self.is_debug:
                    self.logger.debug(
                        "Cache size limit reached during load, removed oldest entry"
                    )

                staging.data[hash_key] = entry
                loaded_count += 1

                value_hash = entry.get('value_hash')
                if value_hash:
                    staging.index_value(value_hash, hash_key)

            # Only the data moves; counters of lookups made before load() stay.
            op.data = staging.data
            op.value_index = staging.value_index

            if self.is_debug:
                self.logger.debug(
                    f"Loaded {loaded_count} entries from {cache_filename}. "
                    f"Skipped: {expired_count} expired, {invalid_count} invalid, "
                    f"{stale_version_count} stale cache-key version."
                )
            else:
                if invalid_count > 0:
                    self.logger.warning(
                        f"Skipped {invalid_count} invalid cache entries from {cache_filename}"
                    )
                if stale_version_count > 0:
                    self.logger.info(
                        f"Discarded {stale_version_count} cache entries with outdated cache "
                        f"key format from {cache_filename} (will be regenerated)."
                    )

            return True

    def save(self, cache_type: str) -> bool:
        """
        Write one cache to disk, thread-safely.

        The snapshot is built and expired entries are pruned under the data
        lock; the disk write happens outside it and is ordered by the
        cache_type's _CacheWriteCoordinator. A disabled (non-persistent)
        cache writes nothing.

        Args:
            cache_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.

        Returns:
            True on a write, on a stale snapshot dropped by the coordinator
            and on a disabled cache; False on an unknown cache type or an IO error.
        """
        op = self._get_operation(cache_type)
        if op is None:
            return False

        if not self.persistent:
            return True

        cache_filename = self.get_cache_filename(cache_type)

        with self._lock:
            # value_index is rebuilt from the pruned snapshot: an expired entry
            # must not stay a candidate of miss diagnostics.
            current_time = time.time()
            expiry_seconds = self.cache_expiry_days * 86400
            cleaned_cache: 'OrderedDict[str, dict]' = OrderedDict()
            cleaned_value_index: Dict[str, Set[str]] = {}
            expired_count = 0

            for hash_key, v in op.data.items():
                if isinstance(v, dict) and 'timestamp' in v:
                    if current_time - v['timestamp'] < expiry_seconds:
                        entry = v
                    else:
                        expired_count += 1
                        continue
                else:
                    entry = {'value': v, 'timestamp': current_time}

                cleaned_cache[hash_key] = entry
                value_hash = entry.get('value_hash')
                if value_hash:
                    cleaned_value_index.setdefault(value_hash, set()).add(hash_key)

            if expired_count > 0 and self.is_debug:
                self.logger.debug(
                    f"Removed {expired_count} expired entries from {cache_type} cache before saving."
                )

            cache_to_save = dict(cleaned_cache)  # a copy written outside the lock

            op.data = cleaned_cache
            op.value_index = cleaned_value_index

            # Allocated under the data lock, so version order matches snapshot freshness.
            version = self._write_coordinators[cache_type].next_version()

        # A stale snapshot (a newer one is already on disk) is dropped silently.
        try:
            written = self._write_coordinators[cache_type].attempt_write(
                version, lambda: self.file_ops.write_json_atomic(cache_filename, cache_to_save)
            )

            if self.is_debug:
                if written:
                    self.logger.debug(
                        f"Saved {len(cache_to_save)} entries to {cache_filename} (snapshot v{version})."
                    )
                else:
                    self.logger.debug(
                        f"Skipped stale snapshot v{version} for {cache_filename} "
                        f"(a newer snapshot is already on disk)."
                    )

            return True

        except Exception as e:  # noqa: BLE001 - save failure is reported, not fatal to the run
            self.logger.warning(f"Failed to save {cache_filename}. Error: {e}")
            return False

    def get(self, cache_type: str, cache_key: str) -> Optional[Any]:
        """
        Return a cached value, thread-safely.

        dict/list values are deep-copied so callers cannot mutate the stored
        entry without holding the lock.
        """
        op = self._operations.get(cache_type)
        if op is None:
            return None

        with self._lock:
            entry = op.data.get(cache_key)

            if entry is not None:
                op.hits += 1

                if isinstance(entry, dict) and 'value' in entry:
                    value = entry['value']
                else:
                    value = entry

                if isinstance(value, (dict, list)):
                    return copy.deepcopy(value)
                # Strings and other immutables need no copy.
                return value

            op.misses += 1
            return None

    def delete(self, operation_type: str, key: str) -> bool:
        """
        Remove one cache entry.

        Used to drop corrupted entries (for example a tuple left by an older
        translation-cache format).

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            key: Cache key to remove.

        Returns:
            True if an entry was removed; False if it was absent.
        """
        op = self._operations.get(operation_type)
        if op is None:
            self.logger.warning(f"Unknown operation type for cache deletion: {operation_type}")
            return False

        with self._lock:
            entry = op.data.pop(key, None)
            if entry is None:
                return False

            value_hash = entry.get('value_hash') if isinstance(entry, dict) else None
            op.deindex_key(key, value_hash)

            if self.is_debug:
                self.logger.debug(
                    f"Deleted cache entry: operation={operation_type}, key={key[:50]}..."
                )

            return True

    def set(self, cache_type: str, cache_key: str, value: Any,
            metadata: Optional[dict] = None, source_value: Optional[str] = None) -> None:
        """
        Store a value in the cache, thread-safely.

        Args:
            cache_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            cache_key: Full cache key from create_cache_key().
            value: Value to store (translated or verified string).
            metadata: Optional extra entry fields (original_key, path,
                fingerprints — see RecursiveProcessor._process_leaf).
            source_value: Unmasked leaf source text used to maintain the
                secondary value_index for miss diagnostics. Without it the
                entry is not indexed by value.
        """
        op = self._operations.get(cache_type)
        if op is None:
            return

        with self._lock:
            evicted = op.ensure_capacity()
            if evicted and self.is_debug:
                self.logger.debug(f"Cache {cache_type} limit reached, removed oldest entry")

            cache_entry: dict = {
                'value': value,
                'timestamp': time.time(),
                'cache_key_version': CACHE_KEY_VERSION,
            }

            if metadata:
                cache_entry.update(metadata)

            if source_value is not None:
                value_hash = self._hash_value(source_value)
                cache_entry['value_hash'] = value_hash
                op.index_value(value_hash, cache_key)

            op.data[cache_key] = cache_entry

    def create_cache_key(self, operation_type: str, model: str, system_prompt: str,
                          enhanced_prompt: str, mode: str, value: str,
                          sampling_identity: str) -> str:
        """
        Build an MD5 cache key from the full request sent to the model.

        The key is MD5 of the canonical JSON
        ``[CACHE_KEY_VERSION, operation_type, model, system_prompt,
        enhanced_prompt, mode, value, sampling_identity]``.
        ``enhanced_prompt`` is the already assembled prompt (linguistic hints
        and glossary included — see RecursiveProcessor._build_enhanced_prompt)
        and is hashed whole, not by component, so any future prompt ingredient
        joins the key automatically (see ``doc/pipeline.md``). ``value`` is
        the unmasked leaf source (pipeline invariant: cache stores unmasked
        results). ``sampling_identity`` is the canonical JSON from
        ``build_sampling_identity`` (no cross-parameter reuse).

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            model: API model name.
            system_prompt: System prompt.
            enhanced_prompt: Assembled user prompt (hints and glossary included).
            mode: Mode string (strict / style / verification).
            value: Leaf source value.
            sampling_identity: Canonical sampling JSON from
                ``build_sampling_identity``.

        Returns:
            32-character MD5 hex digest.
        """
        cache_key_data = [
            CACHE_KEY_VERSION, operation_type, model, system_prompt,
            enhanced_prompt, mode, value, sampling_identity,
        ]
        cache_key_str = json.dumps(cache_key_data, ensure_ascii=False)
        return hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()

    def compute_fingerprints(self, operation_type: str, model: str, system_prompt: str,
                              prompt_template: str, hints_text: str, glossary_string: str,
                              mode: str, sampling_identity: str) -> dict:
        """
        Build short component fingerprints for miss diagnostics.

        Fingerprints are not part of the cache key (see create_cache_key) —
        only the full key decides a hit. They are stored in entry metadata
        and used by diagnose_miss() to name which component changed (model,
        system prompt, prompt template, hints, glossary, mode, or sampling)
        without logging the component contents.

        Args:
            operation_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            model: API model name.
            system_prompt: System prompt.
            prompt_template: Base user prompt before hint/glossary injection
                (RuntimeConfig['prompt']).
            hints_text: Joined linguistic-hint text.
            glossary_string: Assembled glossary string.
            mode: Mode string (strict / style / verification).
            sampling_identity: Canonical sampling JSON from
                ``build_sampling_identity``.

        Returns:
            Mapping of component name to truncated hex digest.
        """
        components = {
            'cache_key_version': str(CACHE_KEY_VERSION),
            'operation_type': operation_type,
            'model': model,
            'system_prompt': system_prompt,
            'prompt_template': prompt_template,
            'hints': hints_text,
            'glossary': glossary_string,
            'mode': mode,
            'sampling': sampling_identity,
        }
        return {
            name: hashlib.md5(text.encode('utf-8')).hexdigest()[:CACHE_FINGERPRINT_LENGTH]
            for name, text in components.items()
        }

    def diagnose_miss(self, cache_type: str, value: str, fingerprints: dict) -> str:
        """
        Determine and log why a cache lookup missed.

        The hit/miss decision was already made by the full key
        (create_cache_key) before this call — diagnose_miss only explains the
        miss. Candidates are looked up via the secondary index (entries that
        share the same source value — typically a few: strict/style, different
        paths), not by scanning the whole cache (see ``doc/pipeline.md``).

        Args:
            cache_type: OPERATION_TRANSLATION or OPERATION_VERIFICATION.
            value: Leaf source value that missed.
            fingerprints: Current-request fingerprints from compute_fingerprints().

        Returns:
            Miss reason: ``'new_value'`` (no candidates for this value),
            ``'prompt_assembly'`` (all known fingerprints matched but the full
            key diverged — the fingerprint list needs extending), or a
            comma-separated list of differing components, e.g.
            ``'model, system_prompt'``.
        """
        op = self._operations.get(cache_type)
        if op is None:
            return 'unknown_operation'

        value_hash = self._hash_value(value)

        with self._lock:
            candidate_keys = op.value_index.get(value_hash, ())
            candidates = [op.data[k] for k in candidate_keys if k in op.data]

            if not candidates:
                reason = 'new_value'
            else:
                diffs: Set[str] = set()
                for candidate in candidates:
                    candidate_fingerprints = candidate.get('fingerprints') or {}
                    for name, fp in fingerprints.items():
                        if candidate_fingerprints.get(name) != fp:
                            diffs.add(name)
                reason = ', '.join(sorted(diffs)) if diffs else 'prompt_assembly'

            op.miss_reasons[reason] = op.miss_reasons.get(reason, 0) + 1

        if self.is_debug:
            self.logger.debug(f"Cache miss ({cache_type}): {reason}")

        return reason

    def get_stats(self, cache_type: str) -> dict:
        """Return hit/miss statistics for the given cache type."""
        op = self._operations.get(cache_type)
        if op is None:
            return {'hits': 0, 'misses': 0, 'total_requests': 0, 'hit_rate': 0.0,
                    'cache_size': 0, 'miss_reasons': {}}
        return op.get_stats()

    def print_stats(self, cache_type: str):
        """Log cache statistics and a breakdown of diagnosed miss reasons."""
        stats = self.get_stats(cache_type)

        if stats['total_requests'] > 0:
            self.logger.info(
                f"{cache_type.capitalize()} cache: {stats['total_requests']} requests, "
                f"{stats['hits']} hits ({stats['hit_rate']:.2f}%), "
                f"cache size: {stats['cache_size']}"
            )

            miss_reasons = stats.get('miss_reasons') or {}
            if miss_reasons:
                total_diagnosed = sum(miss_reasons.values())
                breakdown = ', '.join(
                    f"{reason} — {count}"
                    for reason, count in sorted(miss_reasons.items(), key=lambda kv: kv[1], reverse=True)
                )
                self.logger.info(
                    f"{cache_type.capitalize()} cache misses {total_diagnosed}: {breakdown}"
                )

    def clear(self, cache_type: Optional[str] = None):
        """Clear cache data and statistics for one type, or for all types."""
        targets = [cache_type] if cache_type else list(self._operations.keys())

        for ct in targets:
            op = self._operations.get(ct)
            if op is None:
                continue
            with self._lock:
                op.data.clear()
                op.value_index.clear()
                op.miss_reasons.clear()
                op.hits = 0
                op.misses = 0
