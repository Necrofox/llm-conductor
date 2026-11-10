# -*- coding: utf-8 -*-
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Union,
)

from llm_conductor.constants import GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE

if TYPE_CHECKING:
    from llm_conductor.interfaces import ProcessorConfigProtocol

@dataclass(frozen=True)
class KeyPath:
    """
    Immutable, thread-safe path to a node in a JSON tree.

    Examples:
        KeyPath(('items', 'sword', 'name')) → "items.sword.name"
        KeyPath(('dialogues', 0, 'text')) → "dialogues[0].text"
        KeyPath(()) → "root"

    Notes:
        - Frozen dataclass for safe sharing across threads
        - Segments are dict keys (str) or list indices (int)
        - Hashable, so usable as a dict key
    """
    segments: Tuple[Union[str, int], ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        """
        Human-readable path string.

        Returns:
            A string such as ``"root.key1.key2[0].key3"``.
        """
        if not self.segments:
            return "root"

        parts = []
        for segment in self.segments:
            if isinstance(segment, int):
                parts.append(f"[{segment}]")
            else:
                if parts:  # Not the first segment — insert a dot separator.
                    parts.append(f".{segment}")
                else:
                    parts.append(segment)

        return ''.join(parts)

    def report_key(self) -> str:
        """
        Canonical leaf identity for sidecars and ``run_report.json``.

        Encodes segments as a compact JSON array so a literal object key
        ``"a.b"`` (``["a.b"]``) cannot collide with nested ``a → b``
        (``["a","b"]``). Human logs keep using ``str(self)``.
        """
        return json.dumps(list(self.segments), separators=(',', ':'), ensure_ascii=False)

    def __repr__(self) -> str:
        """Debug representation."""
        return f"KeyPath({self.segments!r})"

    def get_last_segment(self) -> Optional[Union[str, int]]:
        """Return the last path segment (leaf key or index), or None for root."""
        return self.segments[-1] if self.segments else None

    def get_parent(self) -> Optional['KeyPath']:
        """
        Return the parent path.

        Returns:
            A KeyPath without the last segment, or None for root.
        """
        if not self.segments:
            return None
        return KeyPath(self.segments[:-1])

    def append(self, segment: Union[str, int]) -> 'KeyPath':
        """
        Return a new KeyPath with ``segment`` appended.

        Args:
            segment: Dict key (str) or list index (int).

        Returns:
            A new immutable KeyPath.
        """
        return KeyPath(self.segments + (segment,))

    def matches_suffix(self, suffix: str) -> bool:
        """
        Return True if the last string segment ends with ``suffix``.

        Args:
            suffix: Suffix to test (e.g. ``"_plural"``).

        Returns:
            True when the last segment is a str ending with ``suffix``.
        """
        last = self.get_last_segment()
        return isinstance(last, str) and last.endswith(suffix)

    def matches_prefix(self, prefix: str) -> bool:
        """
        Return True if the last string segment starts with ``prefix``.

        Args:
            prefix: Prefix to test (e.g. ``"speaker_"``).

        Returns:
            True when the last segment is a str starting with ``prefix``.
        """
        last = self.get_last_segment()
        return isinstance(last, str) and last.startswith(prefix)

    def contains_segment(self, segment: str) -> bool:
        """
        Return True if ``segment`` appears anywhere in the path.

        Args:
            segment: Segment to look for.

        Returns:
            True if the segment is present.
        """
        return segment in self.segments

    def depth(self) -> int:
        """Return nesting depth (number of segments)."""
        return len(self.segments)

    def iter_ancestors(self, include_self: bool = False) -> List['KeyPath']:
        """
        Return every ancestor path from root toward the current node.

        Args:
            include_self: Whether to append the current path.

        Returns:
            KeyPath list from root to (optionally) self.

        Example:
            >>> path = KeyPath(('dialogues', 'greeting_male', 'text'))
            >>> path.iter_ancestors()
            [KeyPath(()), KeyPath(('dialogues',)), KeyPath(('dialogues', 'greeting_male'))]
        """
        ancestors = []
        for i in range(len(self.segments)):
            ancestors.append(KeyPath(self.segments[:i]))
        if include_self:
            ancestors.append(self)
        return ancestors

    def get_string_segments(self, include_indices: bool = False) -> List[str]:
        """
        Return only the string segments (drop numeric indices by default).

        Args:
            include_indices: When True, stringify indices instead of dropping them.

        Returns:
            List of string segments.
        """
        if include_indices:
            return [str(seg) for seg in self.segments]
        else:
            return [seg for seg in self.segments if isinstance(seg, str)]

    def find_segment_with_pattern(self, pattern: str, check_suffix: bool = True) -> Optional[str]:
        """
        Find the first string segment matching ``pattern`` (leaf toward root).

        Args:
            pattern: Suffix or prefix to match (e.g. ``"_male"``).
            check_suffix: True for suffix match, False for prefix match.

        Returns:
            The matching segment, or None.

        Example:
            >>> path = KeyPath(('dialogues', 'greeting_male', 'text'))
            >>> path.find_segment_with_pattern('_male')
            'greeting_male'
        """
        string_segments = self.get_string_segments()

        # Leaf-first: nearer ancestors win over distant ones.
        for segment in reversed(string_segments):
            segment_lower = segment.lower()
            pattern_lower = pattern.lower()

            if check_suffix and segment_lower.endswith(pattern_lower) or not check_suffix and segment_lower.startswith(pattern_lower):
                return segment

        return None

class HintType(Enum):
    """Kinds of linguistic hints injected into the model prompt."""
    PLURAL = "plural"
    GENDER = "gender"
    CASE = "case"
    FORMALITY = "formality"
    CUSTOM = "custom"

@dataclass(frozen=True)
class LinguisticHint:
    """
    Immutable linguistic hint for the translation model.

    Attributes:
        hint_type: Hint kind (plural / gender / case / ...).
        instruction: English instruction text for the model.
        confidence: Detector confidence in ``[0.0, 1.0]``.
        priority: Application priority (higher wins when ordering hints).
    """
    hint_type: HintType
    instruction: str
    confidence: float = 1.0
    priority: int = 0

    def __post_init__(self):
        """Validate fields on construction."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        if not self.instruction.strip():
            raise ValueError("instruction cannot be empty")

    def __lt__(self, other: 'LinguisticHint') -> bool:
        """Order by priority (used when sorting hints)."""
        return self.priority < other.priority

class LinguisticContext:
    """
    Thread-safe analyzer of JSON key paths, extracting linguistic patterns.

    Supported patterns (extensible via register_pattern()):
      - Plurals: *_plural, *_many, *_few, *_zero, count_*
      - Gender: *_male, *_female, *_m, *_f, speaker_male, speaker_female
      - Formality: *_formal, *_informal

    Features:
      - Custom detectors registered through register_pattern()
      - Unknown patterns logged automatically (see _log_unknown_pattern())
      - Two independent threading.Lock instances (_unknown_lock, _cache_lock)
      - LRU cache of analysis results

    Thread-safety contract: get_unknown_patterns() and clear_cache() each
    take their own lock and must never be called from code that already
    holds it - neither lock is reentrant. _log_unknown_pattern() follows
    the same rule: it releases _unknown_lock before logging, so the logger
    call below never runs with the lock held.
    """
    
    def __init__(self, config: 'ProcessorConfigProtocol'):
        """
        Args:
            config: Processor-facing config (logger and related settings).
        """
        self.config = config

        # name -> detector
        self._pattern_rules: Dict[str, Callable[[KeyPath], Optional[LinguisticHint]]] = {}

        # Unknown-pattern audit set (thread-safe).
        self._unknown_patterns: set = set()
        self._unknown_lock = threading.Lock()

        # LRU of analysis results.
        self._analysis_cache: OrderedDict[str, List[LinguisticHint]] = OrderedDict()
        self._cache_max_size = 5000
        self._cache_lock = threading.Lock()

        self._register_builtin_patterns()

    def _register_builtin_patterns(self):
        """Register the built-in plural / gender / formality detectors."""
        self.register_pattern("plural", self._detect_plural)
        self.register_pattern("gender", self._detect_gender)
        self.register_pattern("formality", self._detect_formality)

    def register_pattern(self, name: str, detector: Callable[[KeyPath], Optional[LinguisticHint]]):
        """
        Register a custom pattern detector.

        Args:
            name: Unique detector name.
            detector: ``(KeyPath) -> Optional[LinguisticHint]``.

        Example:
            def detect_my_pattern(path: KeyPath) -> Optional[LinguisticHint]:
                if path.matches_suffix("_custom"):
                    return LinguisticHint(
                        hint_type=HintType.CUSTOM,
                        instruction="Use custom translation rules",
                        priority=10
                    )
                return None

            context.register_pattern("my_pattern", detect_my_pattern)
        """
        if not callable(detector):
            raise TypeError(f"detector must be callable, got {type(detector).__name__}")

        self._pattern_rules[name] = detector

        if self.config.is_debug_mode:
            self.config.logger.debug(f"Registered linguistic pattern detector: {name}")

    def analyze(self, key_path: KeyPath, value: str) -> List[LinguisticHint]:
        """
        Analyze a key path and return linguistic hints.

        Args:
            key_path: Path to the JSON node.
            value: Leaf text reserved for future context-aware analysis.

        Returns:
            LinguisticHint list sorted by priority (descending).
        """
        cache_key = str(key_path)

        with self._cache_lock:
            if cache_key in self._analysis_cache:
                self._analysis_cache.move_to_end(cache_key)
                return self._analysis_cache[cache_key]

        hints = []
        seen_types = set()

        try:
            # 1. Local analysis (current segment only).
            for rule_name, detector in self._pattern_rules.items():
                try:
                    hint = detector(key_path)
                    if hint and hint.hint_type not in seen_types:
                        hints.append(hint)
                        seen_types.add(hint.hint_type)
                except Exception as e:  # noqa: BLE001 - a broken detector must not abort analysis
                    self.config.logger.warning(
                        f"Pattern detector '{rule_name}' failed for {key_path}: {e}"
                    )

            # 2. Vertical scan of ancestors when some hint types are still missing.
            if len(seen_types) < len(HintType):
                inherited_hints = self._scan_ancestors(key_path, seen_types)
                hints.extend(inherited_hints)

            hints.sort(reverse=True)

            # 3. Audit unknown patterns: log suspicious keys that yielded no hints.
            if not hints and self._is_suspicious_key(key_path):
                self._log_unknown_pattern(key_path)

            if self.config.is_debug_mode and any(h.priority < 10 for h in hints):
                inherited = [h for h in hints if h.priority < 10]
                if inherited:
                    self.config.logger.debug(
                        f"Inherited {len(inherited)} hint(s) from ancestors for {key_path}"
                    )

        except Exception as e:  # noqa: BLE001 - analysis failure degrades to no hints, not a crash
            self.config.logger.error(
                f"Critical error in pattern analysis for {key_path}: {e}",
                include_traceback=self.config.is_debug_mode
            )

        with self._cache_lock:
            if len(self._analysis_cache) >= self._cache_max_size:
                self._analysis_cache.popitem(last=False)  # FIFO
            self._analysis_cache[cache_key] = hints

        return hints

    def _scan_ancestors(self, key_path: KeyPath, skip_types: set) -> List[LinguisticHint]:
        """
        Scan ancestor segments for inherited hints.

        Algorithm:
        1. List ancestors from root to the current node.
        2. Run detectors on each ancestor (nearest parent first).
        3. Lower inherited priority by 5 and confidence by 0.8×.
        4. Skip hint types already found locally (``skip_types``).

        Args:
            key_path: Leaf path being analyzed.
            skip_types: HintType set already satisfied locally (mutated in place).

        Returns:
            Inherited LinguisticHint list with reduced priority.
        """
        inherited_hints = []
        ancestors = key_path.iter_ancestors(include_self=False)

        # Nearest parent first (proximity priority).
        for ancestor_path in reversed(ancestors):
            if not ancestor_path.segments:  # Skip root.
                continue

            for rule_name, detector in self._pattern_rules.items():
                try:
                    hint = detector(ancestor_path)

                    if hint and hint.hint_type not in skip_types:
                        inherited_hint = LinguisticHint(
                            hint_type=hint.hint_type,
                            instruction=f"(inherited from parent) {hint.instruction}",
                            confidence=hint.confidence * 0.8,
                            priority=max(hint.priority - 5, 1),
                        )

                        inherited_hints.append(inherited_hint)
                        skip_types.add(hint.hint_type)

                        if self.config.is_debug_mode:
                            self.config.logger.debug(
                                f"Inherited {hint.hint_type.value} from '{ancestor_path}' "
                                f"for leaf '{key_path}'"
                            )

                except Exception as e:  # noqa: BLE001 - same policy as analyze(): skip this detector
                    self.config.logger.warning(
                        f"Pattern detector '{rule_name}' failed while scanning "
                        f"ancestor '{ancestor_path}' for {key_path}: {e}"
                    )

        return inherited_hints

    def _detect_plural(self, key_path: KeyPath) -> Optional[LinguisticHint]:
        """
        Detect number (plural/singular/few/zero/countable) from the key suffix.

        Matches are anchored to the end of the last segment (`endswith`), not
        a substring search - otherwise e.g. "button_name" would spuriously
        match "_n" and "phone_number" would match "_m".

        Patterns:
          - *_plural / *_many / *_multiple -> many items
          - *_single / *_one / *_singular -> one item
          - *_few -> few items
          - *_zero / *_none / *_empty -> zero items
          - count_* -> countable form
        """
        last = key_path.get_last_segment()
        if not isinstance(last, str):
            return None

        last_lower = last.lower()

        # Plural
        if any(last_lower.endswith(suffix) for suffix in ['_plural', '_many', '_multiple']):
            return LinguisticHint(
                hint_type=HintType.PLURAL,
                instruction="Translate in PLURAL form (many items) if the target language distinguishes number",
                confidence=1.0,
                priority=5
            )

        # Singular
        if any(last_lower.endswith(suffix) for suffix in ['_single', '_one', '_singular']):
            return LinguisticHint(
                hint_type=HintType.PLURAL,
                instruction="Translate in SINGULAR form (one item) if the target language distinguishes number",
                confidence=1.0,
                priority=5
            )

        # Small count (e.g. 2-4, for languages with a dedicated "few" form)
        if last_lower.endswith('_few'):
            return LinguisticHint(
                hint_type=HintType.PLURAL,
                instruction="Translate for FEW items (a small count, e.g. 2-4) if the target language has a "
                            "dedicated form for it",
                confidence=0.9,
                priority=5
            )

        # Zero items
        if any(last_lower.endswith(suffix) for suffix in ['_zero', '_none', '_empty']):
            return LinguisticHint(
                hint_type=HintType.PLURAL,
                instruction="Translate for ZERO items (no items) if the target language has a distinct form for it",
                confidence=0.9,
                priority=5
            )

        # count_ prefix
        if last_lower.startswith('count_'):
            return LinguisticHint(
                hint_type=HintType.PLURAL,
                instruction="Translate as a COUNTABLE noun",
                confidence=0.8,
                priority=3
            )

        return None

    def _detect_gender(self, key_path: KeyPath) -> Optional[LinguisticHint]:
        """
        Detect grammatical gender from the key suffix.

        Matches are anchored to the end of the last segment (`endswith`), not
        a substring search. The speaker_male/speaker_female prefix
        is checked first and wins over the generic suffix rules below: a key
        like "speaker_female" also legitimately ends with "_female", so
        checking the generic rule first would make this specific branch
        unreachable.

        Patterns:
          - speaker_male / speaker_female -> speaker's own gender
          - *_male / *_m / *_masculine -> masculine
          - *_female / *_f / *_feminine -> feminine
          - *_neutral / *_n / *_neuter -> neuter
        """
        last = key_path.get_last_segment()
        if not isinstance(last, str):
            return None

        last_lower = last.lower()

        # speaker_* has its own wording and takes priority over the generic
        # suffix rules below.
        if last_lower.startswith('speaker_'):
            if last_lower.endswith('_male'):
                return LinguisticHint(
                    hint_type=HintType.GENDER,
                    instruction="Speaker is MALE - use masculine verb endings and pronouns if the target "
                                "language marks gender there",
                    confidence=1.0,
                    priority=10
                )
            if last_lower.endswith('_female'):
                return LinguisticHint(
                    hint_type=HintType.GENDER,
                    instruction="Speaker is FEMALE - use feminine verb endings and pronouns if the target "
                                "language marks gender there",
                    confidence=1.0,
                    priority=10
                )

        # Masculine
        if any(last_lower.endswith(suffix) for suffix in ['_male', '_m', '_masculine']):
            return LinguisticHint(
                hint_type=HintType.GENDER,
                instruction="Use MASCULINE grammatical forms if the target language has grammatical gender "
                            "(male speaker/subject)",
                confidence=1.0,
                priority=10
            )

        # Feminine
        if any(last_lower.endswith(suffix) for suffix in ['_female', '_f', '_feminine']):
            return LinguisticHint(
                hint_type=HintType.GENDER,
                instruction="Use FEMININE grammatical forms if the target language has grammatical gender "
                            "(female speaker/subject)",
                confidence=1.0,
                priority=10
            )

        # Neuter
        if any(last_lower.endswith(suffix) for suffix in ['_neutral', '_n', '_neuter']):
            return LinguisticHint(
                hint_type=HintType.GENDER,
                instruction="Use NEUTER grammatical forms if the target language has grammatical gender",
                confidence=0.9,
                priority=10
            )

        return None

    def _detect_formality(self, key_path: KeyPath) -> Optional[LinguisticHint]:
        """
        Detect formal/informal address from the key suffix.

        Matches are anchored to the end of the last segment (`endswith`), not
        a substring search.

        Patterns:
          - *_formal / *_polite / *_respectful -> formal address
          - *_informal / *_casual / *_friendly -> informal address
        """
        last = key_path.get_last_segment()
        if not isinstance(last, str):
            return None

        last_lower = last.lower()

        if any(last_lower.endswith(suffix) for suffix in ['_formal', '_polite', '_respectful']):
            return LinguisticHint(
                hint_type=HintType.FORMALITY,
                instruction="Use FORMAL address forms if the target language distinguishes formal/informal "
                            "address (formal tone)",
                confidence=0.9,
                priority=7
            )

        if any(last_lower.endswith(suffix) for suffix in ['_informal', '_casual', '_friendly']):
            return LinguisticHint(
                hint_type=HintType.FORMALITY,
                instruction="Use INFORMAL address forms if the target language distinguishes formal/informal "
                            "address (casual tone)",
                confidence=0.9,
                priority=7
            )

        return None
    
    def _is_suspicious_key(self, key_path: KeyPath) -> bool:
        """
        Heuristic: whether a key looks like an unknown linguistic pattern.

        Suspicious when any of:
          - a ``_``-suffix of length >= 3 that is not a known detector suffix
          - a system-like prefix (``meta_``, ``sys_``, ``internal_``, ``config_``, …)
          - special characters in the key name (other than ``_`` / word chars)

        Returns:
            True if the key should be audited as unknown.
        """
        last = key_path.get_last_segment()
        if not isinstance(last, str):
            return False

        if last.lower() in ['name', 'description', 'text', 'title', 'message', 'value']:
            return False

        last_lower = last.lower()

        if re.search(r'_[a-z]{3,}$', last_lower):
            known_suffixes = [
                'plural', 'single', 'many', 'few', 'zero',
                'male', 'female', 'neutral',
                'formal', 'informal',
                'name', 'desc', 'text', 'title'
            ]

            match = re.search(r'_([a-z]{3,})$', last_lower)
            if match:
                suffix = match.group(1)
                if suffix not in known_suffixes:
                    return True

        if re.match(r'^(meta|sys|internal|config|debug|temp)_', last_lower):
            return True

        return bool(re.search(r'[^\w_]', last))

    def _log_unknown_pattern(self, key_path: KeyPath):
        """
        Log an unknown key pattern once (thread-safe).

        Warning shape::

            Unknown key pattern detected:
              Path: ...
              Last segment: '...'
              Action: Proceeding with default translation (no linguistic hints)
              Recommendation: ...
        """
        pattern_id = str(key_path)

        with self._unknown_lock:
            if pattern_id in self._unknown_patterns:
                return

            self._unknown_patterns.add(pattern_id)

        # Log outside the lock (see class docstring thread-safety contract).
        last_segment = key_path.get_last_segment()

        self.config.logger.warning(
            f"Unknown key pattern detected:\n"
            f"  Path: {key_path}\n"
            f"  Last segment: '{last_segment}'\n"
            f"  Action: Proceeding with default translation (no linguistic hints)\n"
            f"  Recommendation: If this is a recurring pattern, consider extending\n"
            f"                  LinguisticContext.register_pattern() with custom detector"
        )
    
    def get_unknown_patterns(self) -> FrozenSet[str]:
        """
        Return an immutable snapshot of the unknown key patterns seen so far.

        Takes self._unknown_lock; must not be called while it is held.

        Returns:
            Frozen set of KeyPath string representations.
        """
        with self._unknown_lock:
            return frozenset(self._unknown_patterns)

    def clear_cache(self):
        """Clear the analysis cache (for tests). Takes self._cache_lock; must not be called while it is held."""
        with self._cache_lock:
            self._analysis_cache.clear()

def build_glossary_string(text: str, glossary: dict, config: 'ProcessorConfigProtocol',
                         use_word_boundaries: bool = True) -> str:
    """
    Build a glossary definitions block for the terms found in text.

    Uses GlossaryPatternCache's LRU cache of compiled regex patterns for
    performance; it manages eviction of rarely used patterns on its own.

    Args:
        text: Text to search for glossary terms.
        glossary: Glossary {term: translation}.
        config: Processor-facing configuration.
        use_word_boundaries: Whether to anchor each term to non-word
            boundaries (lookaround, not \\b, so a term starting or ending
            with punctuation such as "C++" or "Dr." still matches).

    Returns:
        A string formatted with GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE
        (entries joined as "term = translation; ..."), or an empty string
        if nothing matched.
    """
    if not glossary:
        return ""

    # Cache not initialized: fall back to uncached direct compilation.
    if config.glossary_patterns_cache is None:
        config.logger.warning("Glossary pattern cache not initialized, using direct regex compilation")
        local_glossary = {}
        for term, translation in glossary.items():
            if use_word_boundaries:
                # Lookaround, not \b: \b never matches next to punctuation,
                # so a term with a non-word character at either edge (C++,
                # Dr., .NET) would otherwise never be found.
                pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(term), re.IGNORECASE)

            if pattern.search(text):
                local_glossary[term] = translation
        
        if local_glossary:
            glossary_entries = [f"{term} = {translation}" for term, translation in local_glossary.items()]
            return GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE.format(
                entries='; '.join(glossary_entries)
            )
        return ""
    
    local_glossary = {}
    
    if use_word_boundaries:
        for term, translation in glossary.items():
            # Cached pattern (compiled on first use); see get_or_compile().
            pattern = config.glossary_patterns_cache.get_or_compile(term, use_word_boundaries=True)

            if pattern.search(text):
                local_glossary[term] = translation
    else:
        # Plain substring search, no word boundaries (also cached).
        for term, translation in glossary.items():
            pattern = config.glossary_patterns_cache.get_or_compile(term, use_word_boundaries=False)
            
            if pattern.search(text):
                local_glossary[term] = translation

    if local_glossary:
        glossary_entries = [f"{term} = {translation}" for term, translation in local_glossary.items()]
        return GLOSSARY_DEFINITIONS_BLOCK_TEMPLATE.format(
            entries='; '.join(glossary_entries)
        )
    else:
        return ""
