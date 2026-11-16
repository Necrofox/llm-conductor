# -*- coding: utf-8 -*-
"""
Where the key-context catalog of a run comes from.

The lookup chain mirrors the glossary, then falls back to context.toml
written by ``init`` beside the program. A selected file is the whole
catalog: its rules are not merged with the init file and missing rows are
not filled from the schema.
"""
import os
import tomllib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from llm_conductor.cli.targets import RunTarget
from llm_conductor.constants import (
    CONTEXT_FILENAME,
    CONTEXT_ORIGIN_OPTION,
    CONTEXT_ORIGIN_PROGRAM,
    CONTEXT_ORIGIN_TARGET,
    UTF8_BOM,
)
from llm_conductor.core.io import safe_read_file
from llm_conductor.exceptions import (
    ConfigValidationError,
    FileOperationError,
    TargetError,
)
from llm_conductor.interfaces import FileOperationsProtocol, Logger
from llm_conductor.l10n.context import (
    ContextRule,
    compile_context_rules,
    compile_leaf_context,
)
from llm_conductor.schema import CONTEXT_SCHEMA


@dataclass(frozen=True)
class ContextSource:
    """The key-context catalog of a run and where it was found."""

    rules: Tuple[ContextRule, ...] = ()
    path: Optional[str] = None
    origin: Optional[str] = None

    def describe(self) -> str:
        """One log line naming the source and how many rules it holds."""
        if self.path is None:
            return "Context: none given and no context.toml next to the target or the program"
        return f"Context: {self.path} ({self.origin}), {len(self.rules)} rule(s)"


class ContextResolver:
    """Walks the lookup chain and loads the catalog it finds."""

    def __init__(self, file_ops: FileOperationsProtocol, logger: Logger,
                 program_dir: Optional[str] = None):
        """
        Args:
            file_ops: File operations for reading the catalog.
            logger: Logger of the run.
            program_dir: Directory of conductor.toml, where ``init`` writes context.toml.
        """
        self._file_ops = file_ops
        self._logger = logger
        self._program_dir = program_dir

    def resolve(self, explicit_path: Optional[str], target: RunTarget) -> ContextSource:
        """
        Find and load the catalog of a lone JSON target.

        Args:
            explicit_path: Value of --context, relative to the working directory.
            target: The run target; its directory holds the implicit context.toml.

        Returns:
            The loaded catalog, or an empty one when the chain finds nothing.

        Raises:
            TargetError: If the explicit file is missing, or a found file is invalid.
        """
        if explicit_path:
            path = os.path.abspath(explicit_path)
            if not os.path.isfile(path):
                raise TargetError(f"Context {path} not found")
            return self._load(path, CONTEXT_ORIGIN_OPTION)

        beside_target = os.path.join(target.directory, CONTEXT_FILENAME)
        if os.path.isfile(beside_target):
            return self._load(beside_target, CONTEXT_ORIGIN_TARGET)
        return self._program_or_empty()

    def load_path(self, path: str, origin: str = CONTEXT_ORIGIN_TARGET) -> ContextSource:
        """
        Load a catalog from an absolute path already resolved by the manifest.

        Raises:
            TargetError: If the file is missing or invalid.
        """
        if not os.path.isfile(path):
            raise TargetError(f"Context {path} not found")
        return self._load(path, origin)

    def program_default(self) -> ContextSource:
        """The init catalog beside the program, or an empty catalog."""
        return self._program_or_empty()

    def _program_or_empty(self) -> ContextSource:
        if self._program_dir:
            candidate = os.path.join(self._program_dir, CONTEXT_FILENAME)
            if os.path.isfile(candidate):
                return self._load(candidate, CONTEXT_ORIGIN_PROGRAM)
        return ContextSource()

    def _load(self, path: str, origin: str) -> ContextSource:
        """Read and check one context.toml."""
        try:
            text = safe_read_file(path, f"context {path}", self._file_ops, self._logger,
                                  critical=True, as_text=True)
        except FileOperationError as e:
            raise TargetError(str(e)) from e
        if not isinstance(text, str):
            raise TargetError(f"Context {path} must be a TOML document")
        try:
            document = tomllib.loads(text.lstrip(UTF8_BOM))
        except tomllib.TOMLDecodeError as e:
            raise TargetError(f"Context {path} is not valid TOML: {e}") from e
        try:
            rows, templates = CONTEXT_SCHEMA.resolve_catalog(document, source=path)
        except ConfigValidationError as e:
            raise TargetError(str(e)) from e
        try:
            rules = (compile_leaf_context(CONTEXT_SCHEMA.resolve_leaves(document), templates)
                     + compile_context_rules(rows, templates))
        except ValueError as e:
            raise TargetError(f"Invalid {path}: {e}") from e
        return ContextSource(rules, path, origin)


def load_contexts_for_target(
        target: RunTarget,
        file_ops: FileOperationsProtocol,
        logger: Logger,
        *,
        cli_context: Optional[str] = None,
        program_dir: Optional[str] = None,
) -> Dict[Optional[str], ContextSource]:
    """
    Load every context catalog referenced by the target.

    ``None`` is the catalog used when an entry has no explicit path: the
    JSON lookup chain, or the program file for a manifest entry that left
    ``context`` unset and has no context.toml beside the manifest.
    """
    resolver = ContextResolver(file_ops, logger, program_dir=program_dir)
    found: Dict[Optional[str], ContextSource] = {}

    if not target.is_manifest:
        source = resolver.resolve(cli_context, target)
        found[None] = source
        if source.path is not None:
            found[source.path] = source
        return found

    if any(entry.context_path is None for entry in target.entries):
        found[None] = resolver.program_default()
    for entry in target.entries:
        path = entry.context_path
        if path is None or path in found:
            continue
        found[path] = resolver.load_path(path)
    return found
