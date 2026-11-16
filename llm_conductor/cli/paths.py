# -*- coding: utf-8 -*-
"""
Path containment helpers shared by run manifests and result layout.

Manifest-relative paths must stay inside the manifest directory: absolute
paths, drive prefixes and ``..`` escapes would make the result layout
ambiguous or overwrite a source file.

Machine-readable file identities (``run_report.json`` keys, sidecar
``relative_path``) use POSIX separators; host filesystem joins use
``host_relative_path``.
"""
import os
from typing import Iterable, Optional

from llm_conductor.constants import MANIFEST_DRIVE_PATTERN
from llm_conductor.exceptions import TargetError


def validate_preview_output(output_path: Optional[str], *,
                            input_paths: Iterable[str], reserved_dirs: Iterable[str]) -> None:
    """Reject a preview destination that could overwrite run data.

    Args:
        output_path: JSONL destination, or None when export is disabled.
        input_paths: Original, draft, manifest, and loaded configuration files.
        reserved_dirs: Result, cache, and log directories used by this run.

    Raises:
        TargetError: If the destination aliases an input or lies in a reserved directory.
    """
    if output_path is None:
        return
    output_key = os.path.normcase(os.path.realpath(output_path))
    for path in input_paths:
        aliases_input = output_key == os.path.normcase(os.path.realpath(path))
        if not aliases_input:
            try:
                aliases_input = os.path.samefile(output_path, path)
            except OSError:
                pass  # Nonexistent destinations are compared by their resolved paths.
        if aliases_input:
            raise TargetError(f"--preview-output must not overwrite an input or configuration file: {path}")
    for directory in reserved_dirs:
        if is_inside(output_key, os.path.realpath(directory)):
            raise TargetError(f"--preview-output must stay outside the reserved directory {directory}")


def portable_relative_path(path: str) -> str:
    """
    Normalize a target-relative path to POSIX separators.

    Used as the stable file identity in ``run_report.json``, leaves
    sidecars, and heartbeat snapshots so the same nested manifest input
    serializes identically on Windows and POSIX. Accepts older native
    separator forms (``\\\\``) so consumers can migrate without dropping
    records.
    """
    return path.replace('\\', '/')


def host_relative_path(portable: str) -> str:
    """
    Convert a portable relative path to a host-OS relative path.

    Args:
        portable: Target-relative path with ``/`` (or legacy ``\\\\``) separators.

    Returns:
        Path suitable for ``os.path.join`` on the current platform.

    Raises:
        ValueError: If the path is empty, rooted, contains a drive prefix,
            or contains ``.`` or ``..`` components.
    """
    normalized = portable_relative_path(portable)
    parts = [part for part in normalized.split('/') if part]
    if (not parts or normalized.startswith('/') or any(
            part in (os.curdir, os.pardir)
            or os.path.splitdrive(part)[0]
            or MANIFEST_DRIVE_PATTERN.match(part)
            for part in parts)):
        raise ValueError(
            "Expected a non-empty relative file path without roots, drive prefixes or dot components")
    return os.path.join(*parts)


def manifest_relative_identity(absolute_path: str, base_dir: str) -> str:
    """
    Build the portable file identity of ``absolute_path`` under ``base_dir``.

    ``os.path.relpath`` yields host separators; this always returns POSIX form.
    """
    return portable_relative_path(os.path.relpath(absolute_path, base_dir))


def is_inside(path: str, directory: str) -> bool:
    """
    Whether path lies inside directory (or is the directory itself).

    Both are compared absolute, normalized and case-folded where the file
    system is case-insensitive.
    """
    path_key = os.path.normcase(os.path.abspath(path))
    dir_key = os.path.normcase(os.path.abspath(directory))
    try:
        return os.path.commonpath([path_key, dir_key]) == dir_key
    except ValueError:  # different drives
        return False


def resolve_under_base(base_dir: str, relative: str) -> str:
    """
    Resolve a relative path that must stay inside ``base_dir``.

    Args:
        base_dir: Absolute directory that bounds the path.
        relative: Path relative to ``base_dir`` (``\\`` and ``/`` both separate).

    Returns:
        Absolute normalized path under ``base_dir``.

    Raises:
        ValueError: When the path is absolute, a drive path, leaves ``base_dir``,
            or resolves to ``base_dir`` itself.
    """
    portable = portable_relative_path(relative)
    drive, _ = os.path.splitdrive(portable)
    if drive or MANIFEST_DRIVE_PATTERN.match(portable) or os.path.isabs(portable) or portable.startswith('/'):
        raise ValueError("absolute path; manifest paths must be relative to the manifest directory")

    base_dir = os.path.abspath(base_dir)
    source_path = os.path.normpath(os.path.join(base_dir, portable))
    if os.path.normcase(source_path) == os.path.normcase(base_dir) or not is_inside(source_path, base_dir):
        raise ValueError(f"outside the manifest directory {base_dir}")
    return source_path
