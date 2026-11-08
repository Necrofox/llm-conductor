# -*- coding: utf-8 -*-
"""
What a run processes and where its results go.

A target is one JSON file or a .list manifest of JSON files. Manifest paths
are relative to the manifest directory and must stay inside it; results
mirror those relative paths under result/{translated,redacted,final}/ with
the original file names.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from llm_conductor.constants import (
    MANIFEST_COMMENT_PREFIX,
    MANIFEST_DRIVE_PATTERN,
    MANIFEST_EXTENSION,
    RESULT_DIR_NAME,
    RESULT_OWNER_FILENAME,
    UTF8_BOM,
)
from llm_conductor.core.io import safe_read_file, safe_write_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


def is_manifest_path(path: str) -> bool:
    """True when the path names a manifest (by its extension)."""
    return os.path.splitext(path)[1].lower() == MANIFEST_EXTENSION


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


@dataclass(frozen=True)
class TargetEntry:
    """One input file of a run."""

    source_path: str      # absolute path of the input file
    relative_path: str    # its path under every result/<stage>/ directory


@dataclass(frozen=True)
class ManifestViolation:
    """A manifest line that breaks the path rules."""

    line_number: int
    line: str
    reason: str

    def __str__(self) -> str:
        return f"  line {self.line_number}: {self.line} - {self.reason}"


class ManifestParser:
    """
    Turns manifest text into run entries.

    Rules: one path per line; surrounding whitespace is stripped; blank
    lines and '#' comments are skipped; '\\' and '/' both separate path
    parts. Every path must be relative and resolve inside the manifest
    directory - an absolute path or one escaping through '..' would make
    the result layout ambiguous or overwrite a source file. Two lines
    naming the same file after normalization are an error too.
    """

    def __init__(self, manifest_path: str):
        """
        Args:
            manifest_path: Path of the manifest; relative lines resolve from its directory.
        """
        self.manifest_path = os.path.abspath(manifest_path)
        self.base_dir = os.path.dirname(self.manifest_path)

    def parse(self, text: str) -> Tuple[TargetEntry, ...]:
        """
        Parse the manifest, checking every line before reporting.

        Args:
            text: Manifest content.

        Returns:
            The entries in manifest order.

        Raises:
            TargetError: Listing every violating line, or when the manifest
                names no file at all.
        """
        entries: List[TargetEntry] = []
        violations: List[ManifestViolation] = []
        first_line_of: Dict[str, int] = {}

        for number, raw_line in enumerate(text.lstrip(UTF8_BOM).splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith(MANIFEST_COMMENT_PREFIX):
                continue

            try:
                entry = self._resolve_line(line)
            except ValueError as e:
                violations.append(ManifestViolation(number, line, str(e)))
                continue

            key = os.path.normcase(entry.source_path)
            if key in first_line_of:
                violations.append(ManifestViolation(number, line, f"duplicate of line {first_line_of[key]}"))
                continue

            first_line_of[key] = number
            entries.append(entry)

        if violations:
            details = "\n".join(str(violation) for violation in violations)
            raise TargetError(f"Manifest {self.manifest_path} has invalid lines:\n{details}")
        if not entries:
            raise TargetError(f"Manifest {self.manifest_path} lists no files")
        return tuple(entries)

    def _resolve_line(self, line: str) -> TargetEntry:
        """
        Resolve one manifest path.

        Raises:
            ValueError: With the reason when the path breaks the rules.
        """
        portable = line.replace('\\', '/')
        drive, _ = os.path.splitdrive(portable)
        if drive or MANIFEST_DRIVE_PATTERN.match(portable) or os.path.isabs(portable) or portable.startswith('/'):
            raise ValueError("absolute path; manifest paths must be relative to the manifest directory")

        source_path = os.path.normpath(os.path.join(self.base_dir, portable))
        if source_path == self.base_dir or not is_inside(source_path, self.base_dir):
            raise ValueError(f"outside the manifest directory {self.base_dir}")

        return TargetEntry(source_path, os.path.relpath(source_path, self.base_dir))


@dataclass(frozen=True)
class RunTarget:
    """The resolved target of a run: the file or manifest and its entries."""

    path: str
    entries: Tuple[TargetEntry, ...]

    @property
    def name(self) -> str:
        """File name of the target; names its log and its default result directory."""
        return os.path.basename(self.path)

    @property
    def directory(self) -> str:
        """Directory of the target file or manifest."""
        return os.path.dirname(self.path)

    @property
    def is_manifest(self) -> bool:
        """True for a manifest target."""
        return is_manifest_path(self.path)


class TargetResolver:
    """Checks the --target path and builds the RunTarget, reading a manifest if needed."""

    def __init__(self, file_ops: FileOperationsProtocol, logger: Logger):
        """
        Args:
            file_ops: File operations for reading the manifest.
            logger: Logger of the run.
        """
        self._file_ops = file_ops
        self._logger = logger

    def resolve(self, target_path: str) -> RunTarget:
        """
        Resolve the target.

        Args:
            target_path: Value of --target, relative to the working directory.

        Returns:
            The target with its entries; a single file is its own only entry,
            with its file name as the relative path.

        Raises:
            TargetError: If the target is missing, unreadable or an invalid manifest.
        """
        path = os.path.abspath(target_path)
        if not os.path.isfile(path):
            raise TargetError(f"Target {path} not found")

        if not is_manifest_path(path):
            return RunTarget(path, (TargetEntry(path, os.path.basename(path)),))

        try:
            text = cast(str, safe_read_file(path, f"manifest {path}", self._file_ops, self._logger,
                                            critical=True, as_text=True))
        except FileOperationError as e:
            raise TargetError(str(e)) from e
        return RunTarget(path, ManifestParser(path).parse(text))


@dataclass(frozen=True)
class ResultLayout:
    """Paths of a run's results: <root>/<stage>/<relative path of the entry>."""

    root: str
    # True for the shared result/<target name>/ layout: the root is claimed
    # by one absolute target path and locked for the run. False for
    # --result2target (result/ next to the target; no owner marker).
    shared_by_name: bool = False

    @classmethod
    def for_target(cls, target: RunTarget, default_base_dir: str, next_to_target: bool) -> 'ResultLayout':
        """
        Choose the result root of a target.

        Args:
            target: The run target.
            default_base_dir: Directory holding the shared result/ directory
                (the directory of conductor.toml).
            next_to_target: --result2target - result/ next to the target
                instead of result/<target name>/ in default_base_dir.

        Returns:
            The layout.
        """
        if next_to_target:
            return cls(os.path.join(target.directory, RESULT_DIR_NAME), shared_by_name=False)
        return cls(
            os.path.join(os.path.abspath(default_base_dir), RESULT_DIR_NAME, target.name),
            shared_by_name=True,
        )

    def stage_path(self, stage: str, entry: TargetEntry) -> str:
        """
        Path of an entry's result of one stage.

        Args:
            stage: STAGE_TRANSLATED, STAGE_REDACTED or STAGE_FINAL.
            entry: The input file.

        Returns:
            The path; its directory may not exist yet (see prepare()).
        """
        return os.path.join(self.root, stage, entry.relative_path)

    def prepare(self, stage: str, entry: TargetEntry) -> str:
        """
        Path of an entry's result of one stage, with its directory created.

        Raises:
            OSError: If the directory cannot be created.
        """
        path = self.stage_path(stage, entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def check_disjoint(self, target: RunTarget) -> None:
        """
        Make sure no input lies under the result root, where a result could overwrite it.

        Raises:
            TargetError: Naming the inputs inside the result root.
        """
        clashing = [entry.source_path for entry in target.entries if is_inside(entry.source_path, self.root)]
        if clashing:
            raise TargetError(
                f"Inputs inside the result directory {self.root} would be overwritten: " + ", ".join(clashing))

    def claim_owner(self, target_path: str, file_ops: FileOperationsProtocol,
                    logger: Optional[Logger] = None) -> None:
        """
        Claim or verify ownership of a shared result root.

        Writes ``.llmc.target`` with the absolute, case-folded target path
        when the marker is missing. A matching marker is accepted; an empty,
        multi-line, unreadable or foreign marker is refused. No-op when the
        layout is not shared by name (``--result2target``).

        Args:
            target_path: Absolute or relative path of the run target.
            file_ops: File operations used to read and write the marker.
            logger: Optional logger for IO helpers.

        Raises:
            TargetError: If the directory already belongs to another target
                or the marker is corrupt / unreadable / unwritable.
        """
        if not self.shared_by_name:
            return

        canonical = os.path.normcase(os.path.abspath(target_path))
        marker_path = os.path.join(self.root, RESULT_OWNER_FILENAME)
        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(marker_path):
            # safe_write_file may return False or raise FileOperationError
            # (including after a fallback copy next to the path). Both mean
            # the marker is not at marker_path — map to TargetError so the
            # entry point exits EXIT_USAGE_ERROR (2), not INTERNAL (1).
            try:
                wrote = safe_write_file(
                    marker_path, canonical + '\n',
                    f"result owner marker {marker_path}",
                    file_ops, logger)
            except FileOperationError as e:
                raise TargetError(
                    f"Cannot write the owner marker of the result directory "
                    f"{self.root}: {e}"
                ) from e
            if not wrote:
                raise TargetError(
                    f"Cannot write the owner marker of the result directory {self.root}")
            return

        try:
            text = cast(str, safe_read_file(
                marker_path, f"result owner marker {marker_path}",
                file_ops, logger, critical=True, as_text=True))
        except FileOperationError as e:
            raise TargetError(
                f"Result directory {self.root} has an unreadable owner marker "
                f"({RESULT_OWNER_FILENAME}): {e}"
            ) from e

        content = text.strip()
        if not content or '\n' in content or '\r' in content:
            raise TargetError(
                f"Result directory {self.root} has a corrupt owner marker "
                f"({RESULT_OWNER_FILENAME}); delete it or use --result2target"
            )

        recorded = os.path.normcase(os.path.abspath(content))
        if recorded != canonical:
            raise TargetError(
                f"Result directory {self.root} belongs to {recorded}, not {canonical}; "
                f"delete that directory or use --result2target"
            )
