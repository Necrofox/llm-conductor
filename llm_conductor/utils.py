# -*- coding: utf-8 -*-
import itertools
import re
import signal as signal_module
import string
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Tuple,
    TypeVar,
)

from llm_conductor.constants import DEFAULT_MARKER_PREFIX, TAG_MASK_PATTERN
from llm_conductor.exceptions import TimeoutError as ConductorTimeoutError

if TYPE_CHECKING:
    from llm_conductor.interfaces import ProcessorConfigProtocol

# ============================================================================
# TIMEOUT UTILITIES
# ============================================================================
T = TypeVar('T')

@contextmanager
def timeout_context(seconds: int):
    """
    SIGALRM-based timeout context manager (POSIX only).

    Used by SignalTimeoutStrategy.run(); calling it directly instead of
    run_with_timeout() is discouraged — on Windows (no SIGALRM) it is a
    no-op and gives the caller no time protection.

    Advantages over ThreadPoolExecutor:
    - No thread-creation overhead
    - Runs in the same thread
    - Minimal resource use

    Limits:
    - POSIX only
    - Not safe in a multi-threaded process
    - One active timeout per process

    Args:
        seconds: Seconds until timeout.

    Raises:
        TimeoutError: When the enclosed work overruns ``seconds``.
    """
    # Windows: no-op — signal-based timeouts are unavailable here.
    # Time protection on Windows is ThreadPoolTimeoutStrategy, not this path.
    if not hasattr(signal_module, 'SIGALRM'):
        yield
        return

    def timeout_handler(signum, frame):
        raise ConductorTimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal_module.signal(signal_module.SIGALRM, timeout_handler)
    # alarm exists only when SIGALRM does; the hasattr guard above is the
    # runtime gate. Windows typeshed omits it — silence the checker here.
    signal_module.alarm(seconds)  # type: ignore[attr-defined]

    try:
        yield
    finally:
        signal_module.alarm(0)  # type: ignore[attr-defined]
        signal_module.signal(signal_module.SIGALRM, old_handler)

class TimeoutStrategy(ABC):
    """
    Strategy for running a function under a time limit.

    Platform implementations (SignalTimeoutStrategy, ThreadPoolTimeoutStrategy)
    hide POSIX/Windows differences behind one interface — callers of
    run_with_timeout() do not know which strategy is active.
    """

    @abstractmethod
    def run(self, func: Callable[[], T], timeout_seconds: int,
            operation_name: str = "operation") -> T:
        """
        Run ``func()`` under a time limit.

        Args:
            func: Zero-argument callable (callers wrap args in a closure/lambda).
            timeout_seconds: Maximum runtime in seconds.
            operation_name: Name used in the timeout error message.

        Raises:
            TimeoutError: On overrun (llm_conductor.exceptions.TimeoutError).
        """
        raise NotImplementedError


class SignalTimeoutStrategy(TimeoutStrategy):
    """
    POSIX strategy based on SIGALRM.

    Runs in the same thread with no thread-pool overhead — see
    timeout_context() for the trade-offs.
    """

    def run(self, func: Callable[[], T], timeout_seconds: int,
            operation_name: str = "operation") -> T:
        with timeout_context(timeout_seconds):
            return func()


class ThreadPoolTimeoutStrategy(TimeoutStrategy):
    """
    Windows (and any platform without SIGALRM) strategy via ThreadPoolExecutor.

    A worker stuck on a blocking call (e.g. a hung network volume) is not
    killed — Python cannot safely abort a thread. The caller stops waiting
    and gets TimeoutError; the worker finishes in the background and is
    collected afterward.
    """

    def run(self, func: Callable[[], T], timeout_seconds: int,
            operation_name: str = "operation") -> T:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                raise ConductorTimeoutError(
                    f"{operation_name} timed out after {timeout_seconds}s")
        finally:
            executor.shutdown(wait=False)


# Pick the strategy once at import time from platform capabilities.
# Strategy objects are stateless across run() calls, so they are
# thread-safe by construction without locks.
_TIMEOUT_STRATEGY: TimeoutStrategy = (
    SignalTimeoutStrategy() if hasattr(signal_module, 'SIGALRM') else ThreadPoolTimeoutStrategy()
)


def run_with_timeout(func: Callable[[], T], timeout_seconds: int,
                      operation_name: str = "operation") -> T:
    """
    Run ``func()`` with a cross-platform time limit.

    Single entry point over TimeoutStrategy: SIGALRM in-process on POSIX,
    ThreadPoolExecutor on Windows. Callers need not know which is active.

    Args:
        func: Zero-argument callable.
        timeout_seconds: Maximum runtime in seconds.
        operation_name: Name used in the timeout error message.

    Returns:
        The result of ``func()``.

    Raises:
        TimeoutError: On overrun (llm_conductor.exceptions.TimeoutError).
    """
    return _TIMEOUT_STRATEGY.run(func, timeout_seconds, operation_name)

# ============================================================================
# MASQUERADE UTILITIES
# ============================================================================

def _marker_prefix_candidates() -> Iterator[str]:
    """
    Yield letter prefixes to try for a collision-free tag marker.

    DEFAULT_MARKER_PREFIX ("T") first, then every other single uppercase
    letter, then two-letter combinations, and so on - growing without bound,
    though a finite source text can only contain finitely many distinct
    bracket-letter-digit substrings, so a real string never needs more than
    a handful of candidates.
    """
    yield DEFAULT_MARKER_PREFIX
    length = 1
    while True:
        for combo in itertools.product(string.ascii_uppercase, repeat=length):
            prefix = ''.join(combo)
            if prefix != DEFAULT_MARKER_PREFIX:
                yield prefix
        length += 1


def _select_marker_prefix(text: str) -> str:
    """
    Pick a marker prefix with no literal "[<prefix><digits>]" already in text.

    Needed because mask_tags() markers are plain substitutions: a literal
    "[T1]" already present in the source text is indistinguishable from a
    generated marker and would be corrupted by unmask_tags().

    Args:
        text: Source text about to be masked.

    Returns:
        A prefix guaranteed not to collide with text.
    """
    for prefix in _marker_prefix_candidates():
        if not re.search(r'\[' + re.escape(prefix) + r'\d+\]', text):
            return prefix
    raise AssertionError(  # pragma: no cover - unreachable, see docstring above
        "No collision-free marker prefix found; this should never happen for finite text"
    )


def mask_tags(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Mask formatting tags so a model call cannot alter them.

    Replaces the following with temporary "[<prefix><n>]" markers:
    - {0}, {1}, ... (placeholders)
    - <tag>, </tag> (HTML/XML tags)
    - \\n (newlines)

    The marker prefix defaults to "T" but switches to a collision-free one
    (see _select_marker_prefix()) if the text already contains a literal
    "[T<digits>]"-shaped substring, so that unmask_tags() cannot confuse a
    generated marker with text that was already there. Skipped entirely
    when there is nothing to mask, so most leaves pay no cost.

    Args:
        text: Source text.

    Returns:
        Tuple (masked_text, tag_map), tag_map mapping each marker to the
        original tag it replaced.
    """
    if not TAG_MASK_PATTERN.search(text):
        return text, {}

    prefix = _select_marker_prefix(text)
    tag_map: Dict[str, str] = {}
    counter = itertools.count()

    def replace_func(match: 're.Match') -> str:
        original = match.group(0)
        marker = f"[{prefix}{next(counter)}]"
        tag_map[marker] = original
        return marker

    masked_text = TAG_MASK_PATTERN.sub(replace_func, text)
    return masked_text, tag_map

class UnmaskResult(NamedTuple):
    """
    Result of unmask_tags().

    missing_markers and orphaned_markers are always populated (not only in
    debug mode), since the content validator (llm_conductor/l10n/validation.py) needs
    them regardless of the log level; the warnings logged from them stay
    gated on is_debug_mode to avoid noise in normal runs.
    """

    text: str
    missing_markers: List[str]
    orphaned_markers: List[str]


def unmask_tags(masked_text: str, tag_map: Dict[str, str], config: 'ProcessorConfigProtocol') -> UnmaskResult:
    """
    Restore the original tags in a model reply.

    Markers missing from the reply (the model dropped them) and markers left
    after the replacement (the model invented them) are always returned to
    the caller; a warning is logged for them in debug mode only. Processing
    is never interrupted here - rejecting the reply is the caller's call.

    Args:
        masked_text: Reply carrying "[<prefix><n>]" markers.
        tag_map: Marker -> original tag, as returned by mask_tags(). The
            marker prefix (default "T") is read back from an existing key;
            an empty tag_map falls back to DEFAULT_MARKER_PREFIX.
        config: Supplies the logger, the debug flag and max_error_display.

    Returns:
        An UnmaskResult with the original tags restored.
    """
    limit = config.max_error_display

    missing_markers = [marker for marker in tag_map if marker not in masked_text]
    if missing_markers and config.is_debug_mode:
        config.logger.warning(
            f"Unmasking: {len(missing_markers)} marker(s) not found in text: {', '.join(missing_markers[:limit])}"
            f"{'...' if len(missing_markers) > limit else ''}. Tags may have been lost during processing.")

    # Longest first; '[T1]' is no prefix of '[T10]' thanks to the bracket,
    # so the order is harmless rather than protective.
    sorted_markers = sorted(tag_map.keys(), key=len, reverse=True)

    result = masked_text
    for marker in sorted_markers:
        result = result.replace(marker, tag_map[marker])

    if tag_map:
        prefix_match = re.match(r'\[([A-Za-z]+)\d+\]', next(iter(tag_map)))
        prefix = prefix_match.group(1) if prefix_match else DEFAULT_MARKER_PREFIX
    else:
        prefix = DEFAULT_MARKER_PREFIX

    orphaned_markers = re.findall(r'\[' + re.escape(prefix) + r'\d+\]', result)
    if orphaned_markers and config.is_debug_mode:
        config.logger.warning(
            f"Unmasking: found {len(orphaned_markers)} orphaned marker(s) not in tag_map: "
            f"{', '.join(orphaned_markers[:limit])}"
            f"{'...' if len(orphaned_markers) > limit else ''}")

    return UnmaskResult(result, missing_markers, orphaned_markers)
