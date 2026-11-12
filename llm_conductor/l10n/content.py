"""Immutable content eligibility and protected-fragment rules for string leaves."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import regex

from llm_conductor.constants import (
    CONTENT_MASK_REPLACEMENT,
    DEFAULT_MASK_PATTERN,
    DEFAULT_REQUIRED_CONTENT_PATTERN,
)
from llm_conductor.exceptions import ConfigValidationError


def _compile_pattern(value: str, key: str) -> Optional[regex.Pattern[str]]:
    """Compile a nonempty setting or raise a named configuration error."""
    if not value:
        return None
    try:
        return regex.compile(value, regex.VERSION0)
    except (regex.error, ValueError) as error:
        raise ConfigValidationError(f"Invalid [processing].{key}: {error}") from error


@dataclass(frozen=True)
class ContentRules:
    """Compiled rules shared by eligibility, masking, and validation.

    Args:
        required_content_pattern: Search expression on the unprotected view;
            empty admits every nonblank leaf.
        mask_pattern: Complete protected-fragment expression; empty disables
            masking. Only nonempty, non-overlapping matches are protected.
    """

    required_content_pattern: str = DEFAULT_REQUIRED_CONTENT_PATTERN
    mask_pattern: str = DEFAULT_MASK_PATTERN
    _required: Optional[regex.Pattern[str]] = field(init=False, repr=False, compare=False)
    _mask: Optional[regex.Pattern[str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_required', _compile_pattern(
            self.required_content_pattern, 'required_content_pattern',
        ))
        object.__setattr__(self, '_mask', _compile_pattern(self.mask_pattern, 'mask_pattern'))

    @property
    def identity(self) -> Tuple[str, str]:
        """Return the two source expressions for cache and dedup identity."""
        return self.required_content_pattern, self.mask_pattern

    def protected_spans(self, text: str) -> List[Tuple[int, int]]:
        """Return ordered nonempty protected spans, ignoring capture groups."""
        if self._mask is None:
            return []
        return sorted(
            (match.start(), match.end())
            for match in self._mask.finditer(text)
            if match.start() < match.end()
        )

    def protected_fragments(self, text: str) -> List[str]:
        """Return exact complete matches, including repeated occurrences."""
        return [text[start:end] for start, end in self.protected_spans(text)]

    def eligibility_text(self, text: str) -> str:
        """Peel boundaries and replace each protected span with one space."""
        core = text.strip()
        parts: List[str] = []
        previous = 0
        for start, end in self.protected_spans(core):
            parts.extend((core[previous:start], CONTENT_MASK_REPLACEMENT))
            previous = end
        parts.append(core[previous:])
        return ''.join(parts)

    def accepts(self, text: str) -> bool:
        """Admit a nonblank leaf when the required expression is found."""
        if not text.strip():
            return False
        if self._required is None:
            return True
        return self._required.search(self.eligibility_text(text)) is not None


DEFAULT_CONTENT_RULES = ContentRules()
