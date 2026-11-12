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

from llm_conductor.constants import MANIFEST_DRIVE_PATTERN


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
    """
    parts = [part for part in portable_relative_path(portable).split('/') if part]
    if not parts:
        return portable_relative_path(portable)
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
        ValueError: When the path is absolute, a drive path, or leaves ``base_dir``.
    """
    portable = portable_relative_path(relative)
    drive, _ = os.path.splitdrive(portable)
    if drive or MANIFEST_DRIVE_PATTERN.match(portable) or os.path.isabs(portable) or portable.startswith('/'):
        raise ValueError("absolute path; manifest paths must be relative to the manifest directory")

    source_path = os.path.normpath(os.path.join(base_dir, portable))
    if source_path == base_dir or not is_inside(source_path, base_dir):
        raise ValueError(f"outside the manifest directory {base_dir}")
    return source_path
