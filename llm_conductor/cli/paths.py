# -*- coding: utf-8 -*-
"""
Path containment helpers shared by run manifests and result layout.

Manifest-relative paths must stay inside the manifest directory: absolute
paths, drive prefixes and ``..`` escapes would make the result layout
ambiguous or overwrite a source file.
"""
import os

from llm_conductor.constants import MANIFEST_DRIVE_PATTERN


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
    portable = relative.replace('\\', '/')
    drive, _ = os.path.splitdrive(portable)
    if drive or MANIFEST_DRIVE_PATTERN.match(portable) or os.path.isabs(portable) or portable.startswith('/'):
        raise ValueError("absolute path; manifest paths must be relative to the manifest directory")

    source_path = os.path.normpath(os.path.join(base_dir, portable))
    if source_path == base_dir or not is_inside(source_path, base_dir):
        raise ValueError(f"outside the manifest directory {base_dir}")
    return source_path
