# -*- coding: utf-8 -*-
"""
Where the glossary of a run comes from.

The lookup chain has two levels: --glossary PATH (a missing file is an
error), then glossary.json next to the target (file or manifest), then no
glossary at all. A glossary is project data, so there is no shared fallback
next to the executable - it would leak one project's terms into another.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from llm_conductor.cli.targets import RunTarget
from llm_conductor.constants import (
    GLOSSARY_FILENAME,
    GLOSSARY_ORIGIN_OPTION,
    GLOSSARY_ORIGIN_TARGET,
)
from llm_conductor.core.io import safe_read_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


@dataclass(frozen=True)
class GlossarySource:
    """The glossary of a run and where it was found."""

    terms: Dict[str, str] = field(default_factory=dict)
    path: Optional[str] = None
    origin: Optional[str] = None

    def describe(self) -> str:
        """One log line naming the source, its level of the chain and the number of terms."""
        if self.path is None:
            return "Glossary: none given and no glossary.json next to the target - working without terms"
        return f"Glossary: {self.path} ({self.origin}), {len(self.terms)} term(s)"


class GlossaryResolver:
    """Walks the lookup chain and loads the glossary it finds."""

    def __init__(self, file_ops: FileOperationsProtocol, logger: Logger):
        """
        Args:
            file_ops: File operations for reading the glossary.
            logger: Logger of the run.
        """
        self._file_ops = file_ops
        self._logger = logger

    def resolve(self, explicit_path: Optional[str], target: RunTarget) -> GlossarySource:
        """
        Find and load the glossary of a run.

        Args:
            explicit_path: Value of --glossary, relative to the working directory.
            target: The run target; its directory holds the implicit glossary.json.

        Returns:
            The loaded glossary, or an empty one when the chain finds nothing.

        Raises:
            TargetError: If the explicit glossary is missing, or a found
                glossary is unreadable or not an object of strings.
        """
        if explicit_path:
            path = os.path.abspath(explicit_path)
            if not os.path.isfile(path):
                raise TargetError(f"Glossary {path} not found")
            return self._load(path, GLOSSARY_ORIGIN_OPTION)

        candidate = os.path.join(target.directory, GLOSSARY_FILENAME)
        if os.path.isfile(candidate):
            return self._load(candidate, GLOSSARY_ORIGIN_TARGET)
        return GlossarySource()

    def _load(self, path: str, origin: str) -> GlossarySource:
        """
        Read and check one glossary file.

        Raises:
            TargetError: If it is unreadable or not a {term: translation} object of strings.
        """
        try:
            data = safe_read_file(path, f"glossary {path}", self._file_ops, self._logger,
                                  critical=True, as_text=False)
        except FileOperationError as e:
            raise TargetError(str(e)) from e

        if not isinstance(data, dict) or not all(
                isinstance(term, str) and isinstance(translation, str) for term, translation in data.items()):
            raise TargetError(f"Glossary {path} must be a JSON object mapping terms to translation strings")
        return GlossarySource(data, path, origin)
