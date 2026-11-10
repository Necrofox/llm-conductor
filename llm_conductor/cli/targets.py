# -*- coding: utf-8 -*-
"""
What a run processes and where its results go.

A target is one JSON file or a TOML run manifest (``[[targets]]``). Manifest
paths are relative to the manifest directory and must stay inside it; results
mirror those relative paths under result/{translated,redacted,final}/ with
the original file names.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, cast

from llm_conductor.cli.paths import is_inside
from llm_conductor.cli.run_manifest import (
    RunManifestParser,
    RunSettings,
    is_legacy_list_path,
    is_run_manifest_path,
)
from llm_conductor.constants import (
    LEAVES_DIR_NAME,
    LEGACY_LIST_EXTENSION,
    MANIFEST_EXTENSION,
    RESULT_DIR_NAME,
    RESULT_OWNER_FILENAME,
    RUN_STAGE_ALL,
)
from llm_conductor.core.io import safe_read_file, safe_write_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger

# Re-export for callers that imported containment from this module.
__all__ = [
    'ResultLayout',
    'RunTarget',
    'TargetEntry',
    'TargetResolver',
    'is_inside',
    'is_manifest_path',
]


def is_manifest_path(path: str) -> bool:
    """True when the path names a TOML run manifest (by its extension)."""
    return is_run_manifest_path(path)


@dataclass(frozen=True)
class TargetEntry:
    """One input file of a run, with effective stage settings."""

    source_path: str      # absolute path of the input file
    relative_path: str    # its path under every result/<stage>/ directory
    index: int = 1        # 1-based [[targets]] index (1 for a lone JSON)
    draft_path: Optional[str] = None
    stage: str = RUN_STAGE_ALL
    strict: bool = False
    llm_config_path: Optional[str] = None  # absolute llm.toml; None → bootstrap Config
    glossary_path: Optional[str] = None    # absolute glossary; None → empty terms


@dataclass(frozen=True)
class RunTarget:
    """The resolved target of a run: the file or manifest and its entries."""

    path: str
    entries: Tuple[TargetEntry, ...]
    # Populated for TOML manifests so the CLI can read layout / verb defaults.
    settings: Optional[RunSettings] = None

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
        """True for a TOML run-manifest target."""
        return is_manifest_path(self.path)


class TargetResolver:
    """Checks the --target path and builds the RunTarget, reading a manifest if needed."""

    def __init__(self, file_ops: FileOperationsProtocol, logger: Logger,
                 conductor_dir: Optional[str] = None):
        """
        Args:
            file_ops: File operations for reading the manifest.
            logger: Logger of the run.
            conductor_dir: Directory of conductor.toml (default llm profile anchor).
                Required when resolving a TOML run manifest.
        """
        self._file_ops = file_ops
        self._logger = logger
        self._conductor_dir = conductor_dir

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

        if is_legacy_list_path(path):
            raise TargetError(
                f"{LEGACY_LIST_EXTENSION} manifests are no longer supported "
                f"(target was {path}). Migrate each line to a {MANIFEST_EXTENSION} "
                f"run manifest:\n"
                f"  [[targets]]\n"
                f"  original = \"path/from/line.json\"\n"
                f"See doc/cli.md."
            )

        if not is_manifest_path(path):
            return RunTarget(path, (TargetEntry(path, os.path.basename(path)),))

        if self._conductor_dir is None:
            raise TargetError(
                f"Internal error: conductor directory required to resolve run manifest {path}"
            )

        settings = RunManifestParser(
            path, self._conductor_dir, self._file_ops, self._logger
        ).parse_file()
        entries = tuple(target.as_entry() for target in settings.targets)
        return RunTarget(path, entries, settings=settings)


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

    def leaves_path(self, entry: TargetEntry) -> str:
        """
        Path of the per-leaf status sidecar for one input file.

        Sidecars live under ``leaves/`` next to translated/redacted/final so
        a standalone ``--stage verify`` can load translation_degraded flags.
        """
        return os.path.join(self.root, LEAVES_DIR_NAME, entry.relative_path)

    def prepare_leaves(self, entry: TargetEntry) -> str:
        """
        Sidecar path with its directory created.

        Raises:
            OSError: If the directory cannot be created.
        """
        path = self.leaves_path(entry)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def check_disjoint(self, target: RunTarget) -> None:
        """
        Make sure no input lies under the result root, where a result could overwrite it.

        Raises:
            TargetError: Naming the inputs inside the result root.
        """
        clashing = []
        for entry in target.entries:
            if is_inside(entry.source_path, self.root):
                clashing.append(entry.source_path)
            if entry.draft_path is not None and is_inside(entry.draft_path, self.root):
                clashing.append(entry.draft_path)
        if clashing:
            raise TargetError(
                f"Inputs inside the result directory {self.root} would be overwritten: "
                + ", ".join(clashing))

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
        if not self.shared_by_name or self.check_owner(target_path, file_ops, logger):
            return

        canonical = os.path.normcase(os.path.abspath(target_path))
        marker_path = os.path.join(self.root, RESULT_OWNER_FILENAME)
        os.makedirs(self.root, exist_ok=True)

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

    def check_owner(self, target_path: str, file_ops: FileOperationsProtocol,
                    logger: Optional[Logger] = None) -> bool:
        """Check an existing shared result owner without creating any files.

        Args:
            target_path: Absolute or relative path of the run target.
            file_ops: File operations used to read the marker.
            logger: Optional logger for IO helpers.

        Returns:
            True if the marker matches. False if absent or ownership is unused.

        Raises:
            TargetError: If the marker is corrupt, unreadable or foreign.
        """
        if not self.shared_by_name:
            return False

        marker_path = os.path.join(self.root, RESULT_OWNER_FILENAME)
        if not os.path.exists(marker_path):
            return False

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

        canonical = os.path.normcase(os.path.abspath(target_path))
        recorded = os.path.normcase(os.path.abspath(content))
        if recorded != canonical:
            raise TargetError(
                f"Result directory {self.root} belongs to {recorded}, not {canonical}; "
                f"delete that directory or use --result2target"
            )
        return True
