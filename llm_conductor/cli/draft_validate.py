# -*- coding: utf-8 -*-
"""
Structural and marker comparison of an original JSON tree and a frozen draft.

Draft verification (#14) maps leaves by full JSON key path. Missing keys,
extra keys, incompatible value types, and tag/placeholder multiset mismatches
are collected before any model call.
"""
from collections import Counter
from typing import Any, List

from llm_conductor.constants import TAG_MASK_PATTERN
from llm_conductor.l10n.context import KeyPath


def _type_name(value: Any) -> str:
    """Short type label for mismatch messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _compatible_types(left: Any, right: Any) -> bool:
    """True when both values are the same JSON shape family (dict/list/scalar class)."""
    if isinstance(left, dict) and isinstance(right, dict):
        return True
    if isinstance(left, list) and isinstance(right, list):
        return True
    if isinstance(left, str) and isinstance(right, str):
        return True
    if isinstance(left, bool) and isinstance(right, bool):
        return True
    # bool is a subclass of int — reject bool/int mixing.
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return True
    return left is None and right is None


def compare_json_structure(original: Any, draft: Any) -> List[str]:
    """
    Compare two JSON values by full key path.

    Args:
        original: Source JSON (English).
        draft: Frozen translated draft.

    Returns:
        Human-readable issues (empty when structures match). Each issue names
        a KeyPath and whether the key is missing, extra or type-mismatched.
    """
    issues: List[str] = []
    _walk_structure(KeyPath(), original, draft, issues)
    return issues


def compare_draft_markers(original: Any, draft: Any) -> List[str]:
    """
    Compare tag/placeholder multisets on matching string leaves.

    Uses the same ``TAG_MASK_PATTERN`` as reply masking (``{N}``, newlines,
    XML-like tags). Call after ``compare_json_structure`` succeeds so every
    string leaf is present on both sides.

    Args:
        original: Source JSON (English).
        draft: Frozen translated draft.

    Returns:
        Human-readable marker mismatch issues (empty when multisets match).
    """
    issues: List[str] = []
    _walk_markers(KeyPath(), original, draft, issues)
    return issues


def _marker_multiset(text: str) -> Counter:
    """Count structural markers that must be preserved across draft verify."""
    return Counter(TAG_MASK_PATTERN.findall(text))


def _walk_structure(path: KeyPath, original: Any, draft: Any, issues: List[str]) -> None:
    """Recurse; append structure issues for this node and descendants."""
    if not _compatible_types(original, draft):
        issues.append(
            f"{path}: type mismatch (original {_type_name(original)}, draft {_type_name(draft)})"
        )
        return

    if isinstance(original, dict):
        original_keys = set(original)
        draft_keys = set(draft)
        for key in sorted(original_keys - draft_keys):
            issues.append(f"{path.append(key)}: missing in draft")
        for key in sorted(draft_keys - original_keys):
            issues.append(f"{path.append(key)}: extra in draft")
        for key in sorted(original_keys & draft_keys):
            _walk_structure(path.append(key), original[key], draft[key], issues)
        return

    if isinstance(original, list):
        if len(original) != len(draft):
            issues.append(
                f"{path}: array length mismatch (original {len(original)}, draft {len(draft)})"
            )
            # Still compare the overlapping prefix so callers see leaf issues too.
        for index, (left, right) in enumerate(zip(original, draft)):
            _walk_structure(path.append(index), left, right, issues)
        return


def _walk_markers(path: KeyPath, original: Any, draft: Any, issues: List[str]) -> None:
    """Recurse matching trees; compare marker multisets on string leaves."""
    if not _compatible_types(original, draft):
        return

    if isinstance(original, dict):
        for key in sorted(set(original) & set(draft)):
            _walk_markers(path.append(key), original[key], draft[key], issues)
        return

    if isinstance(original, list):
        for index, (left, right) in enumerate(zip(original, draft)):
            _walk_markers(path.append(index), left, right, issues)
        return

    if isinstance(original, str) and isinstance(draft, str):
        original_markers = _marker_multiset(original)
        draft_markers = _marker_multiset(draft)
        if original_markers != draft_markers:
            missing = original_markers - draft_markers
            extra = draft_markers - original_markers
            parts = []
            if missing:
                parts.append(
                    "missing in draft: "
                    + ", ".join(f"{marker!r}×{count}" for marker, count in sorted(missing.items()))
                )
            if extra:
                parts.append(
                    "extra in draft: "
                    + ", ".join(f"{marker!r}×{count}" for marker, count in sorted(extra.items()))
                )
            detail = "; ".join(parts)
            issues.append(f"{path}: marker mismatch ({detail})")


def format_draft_structure_errors(target_index: int, issues: List[str]) -> str:
    """
    Format structural draft issues for a TargetError message.

    Args:
        target_index: 1-based ``[[targets]]`` index.
        issues: Lines from ``compare_json_structure``.

    Returns:
        Multi-line details block.
    """
    header = f"  target {target_index}: draft structure does not match original"
    body = "\n".join(f"    {issue}" for issue in issues)
    return f"{header}\n{body}"


def format_draft_marker_errors(target_index: int, issues: List[str]) -> str:
    """
    Format draft marker-multiset issues for a TargetError message.

    Args:
        target_index: 1-based ``[[targets]]`` index.
        issues: Lines from ``compare_draft_markers``.

    Returns:
        Multi-line details block.
    """
    header = f"  target {target_index}: draft markers do not match original"
    body = "\n".join(f"    {issue}" for issue in issues)
    return f"{header}\n{body}"
