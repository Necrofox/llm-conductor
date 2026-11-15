# -*- coding: utf-8 -*-
"""
TOML run manifest: parse, validate and resolve effective per-target settings.

A ``.toml`` ``--target`` must contain a non-empty ``[[targets]]`` array.
Job fields come from the manifest (precedence B); session flags ``--verb`` /
``--no-cache`` still apply on top. Path rules match the former ``.list``
contract: relative, no ``..`` escape, inside the manifest directory.
"""
import os
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from llm_conductor.cli.paths import (
    is_inside,
    manifest_relative_identity,
    portable_relative_path,
    resolve_under_base,
)
from llm_conductor.constants import (
    GLOSSARY_FILENAME,
    LEGACY_LIST_EXTENSION,
    LLM_CONFIG_FILENAME,
    MANIFEST_EXTENSION,
    RESULT_DIR_NAME,
    RUN_MANIFEST_RUN_KEYS,
    RUN_MANIFEST_TARGET_KEYS,
    RUN_STAGE_ALL,
    RUN_STAGE_VERIFY,
    UTF8_BOM,
    VALID_LOG_LEVELS,
    VALID_RUN_STAGES,
)
from llm_conductor.core.io import safe_read_file
from llm_conductor.exceptions import FileOperationError, TargetError
from llm_conductor.interfaces import FileOperationsProtocol, Logger


@dataclass(frozen=True)
class ResolvedTarget:
    """One ``[[targets]]`` entry after defaults and path resolution."""

    index: int
    original: str
    relative_path: str  # portable (POSIX) identity under result/<stage>/
    draft: Optional[str]
    stage: str
    strict: bool
    llm_config: str
    glossary: Optional[str]

    def as_entry(self):
        """Convert to the runner's ``TargetEntry``."""
        from llm_conductor.cli.targets import TargetEntry
        return TargetEntry(
            source_path=self.original,
            relative_path=self.relative_path,
            index=self.index,
            draft_path=self.draft,
            stage=self.stage,
            strict=self.strict,
            llm_config_path=self.llm_config,
            glossary_path=self.glossary,
        )


@dataclass(frozen=True)
class RunSettings:
    """Resolved run manifest ready for layout, Config bootstrap and the runner."""

    path: str
    result2target: bool
    cache2target: bool
    verb: Optional[str]
    targets: Tuple[ResolvedTarget, ...]

    @property
    def directory(self) -> str:
        """Directory of the manifest file."""
        return os.path.dirname(self.path)

    @property
    def name(self) -> str:
        """Basename of the manifest (log / default result root)."""
        return os.path.basename(self.path)


@dataclass(frozen=True)
class _RawTarget:
    """Unchecked fields of one ``[[targets]]`` table."""

    index: int
    original: Optional[Any]
    draft: Optional[Any]
    stage: Optional[Any]
    strict: Optional[Any]
    llm_config: Optional[Any]
    glossary: Optional[Any]


@dataclass(frozen=True)
class _RawManifest:
    """Unchecked top-level run manifest document."""

    verb: Optional[Any]
    result2target: Optional[Any]
    cache2target: Optional[Any]
    stage: Optional[Any]
    strict: Optional[Any]
    llm_config: Optional[Any]
    glossary: Optional[Any]
    targets: Tuple[_RawTarget, ...]


class ManifestViolation:
    """One collected validation fault (run-level or per-target)."""

    def __init__(self, message: str, target_index: Optional[int] = None):
        self.message = message
        self.target_index = target_index

    def __str__(self) -> str:
        if self.target_index is None:
            return f"  {self.message}"
        return f"  target {self.target_index}: {self.message}"


def is_run_manifest_path(path: str) -> bool:
    """True when the path extension is the TOML run-manifest extension."""
    return os.path.splitext(path)[1].lower() == MANIFEST_EXTENSION


def is_legacy_list_path(path: str) -> bool:
    """True when the path uses the removed ``.list`` extension."""
    return os.path.splitext(path)[1].lower() == LEGACY_LIST_EXTENSION


class RunManifestParser:
    """
    Parse and resolve a TOML run manifest.

    Collects every schema, path and collision error before raising one
    ``TargetError`` (exit 2). Does not call the API or write results.
    """

    def __init__(self, manifest_path: str, conductor_dir: str,
                 file_ops: FileOperationsProtocol, logger: Logger):
        """
        Args:
            manifest_path: Absolute or relative path of the ``.toml`` file.
            conductor_dir: Directory of ``conductor.toml`` (default llm profile).
            file_ops: File operations for reading the manifest.
            logger: Logger for IO helpers.
        """
        self.manifest_path = os.path.abspath(manifest_path)
        self.base_dir = os.path.dirname(self.manifest_path)
        self.conductor_dir = os.path.abspath(conductor_dir)
        self._file_ops = file_ops
        self._logger = logger
        self._violations: List[ManifestViolation] = []

    def parse_file(self) -> RunSettings:
        """
        Read, parse and resolve the manifest on disk.

        Raises:
            TargetError: On IO, TOML decode, schema or path faults.
        """
        try:
            text = cast(str, safe_read_file(
                self.manifest_path, f"run manifest {self.manifest_path}",
                self._file_ops, self._logger, critical=True, as_text=True))
        except FileOperationError as e:
            raise TargetError(str(e)) from e
        return self.parse_text(text)

    def parse_text(self, text: str) -> RunSettings:
        """
        Parse manifest text into effective settings.

        Raises:
            TargetError: Listing every collected fault.
        """
        self._violations = []
        try:
            document = tomllib.loads(text.lstrip(UTF8_BOM))
        except tomllib.TOMLDecodeError as e:
            raise TargetError(
                f"Target {self.manifest_path} is not a run manifest: invalid TOML ({e})"
            ) from e

        if not isinstance(document, dict):
            raise TargetError(
                f"Target {self.manifest_path} is not a run manifest "
                f"(expected a TOML table with [[targets]])"
            )

        raw = self._extract_raw(document)
        if self._violations:
            self._raise()

        if not raw.targets:
            raise TargetError(
                f"Target {self.manifest_path} is not a run manifest "
                f"(non-empty [[targets]] required)"
            )

        return self._resolve(raw)

    def _extract_raw(self, document: Dict[str, Any]) -> _RawManifest:
        """Pull typed-ish fields; record unknown keys and wrong container shapes."""
        unknown = sorted(set(document) - RUN_MANIFEST_RUN_KEYS)
        for key in unknown:
            self._violations.append(ManifestViolation(f"unknown key '{key}'"))

        targets_value = document.get('targets')
        raw_targets: List[_RawTarget] = []
        if targets_value is None:
            self._violations.append(ManifestViolation(
                "missing [[targets]]; not a run manifest"))
        elif not isinstance(targets_value, list):
            self._violations.append(ManifestViolation(
                "'targets' must be an array of tables ([[targets]])"))
        else:
            for index, item in enumerate(targets_value, 1):
                if not isinstance(item, dict):
                    self._violations.append(ManifestViolation(
                        "entry must be a table", target_index=index))
                    continue
                unknown_t = sorted(set(item) - RUN_MANIFEST_TARGET_KEYS)
                for key in unknown_t:
                    self._violations.append(ManifestViolation(
                        f"unknown key '{key}'", target_index=index))
                raw_targets.append(_RawTarget(
                    index=index,
                    original=item.get('original'),
                    draft=item.get('draft'),
                    stage=item.get('stage'),
                    strict=item.get('strict'),
                    llm_config=item.get('llm_config'),
                    glossary=item.get('glossary'),
                ))

        return _RawManifest(
            verb=document.get('verb'),
            result2target=document.get('result2target'),
            cache2target=document.get('cache2target'),
            stage=document.get('stage'),
            strict=document.get('strict'),
            llm_config=document.get('llm_config'),
            glossary=document.get('glossary'),
            targets=tuple(raw_targets),
        )

    def _resolve(self, raw: _RawManifest) -> RunSettings:
        """Apply defaults, resolve paths, check duplicates and collisions."""
        verb = self._optional_string(raw.verb, 'verb', allowed=VALID_LOG_LEVELS)
        if verb is not None:
            verb = verb.upper()
        result2target = self._optional_bool(raw.result2target, 'result2target')
        cache2target = self._optional_bool(raw.cache2target, 'cache2target')
        run_stage = self._optional_string(raw.stage, 'stage', allowed=VALID_RUN_STAGES)
        run_strict = self._optional_bool(raw.strict, 'strict')
        run_llm = self._optional_string(raw.llm_config, 'llm_config')
        run_glossary = self._optional_string(raw.glossary, 'glossary')

        default_llm = os.path.join(self.conductor_dir, LLM_CONFIG_FILENAME)
        run_llm_path = self._resolve_optional_path(run_llm, 'llm_config') if run_llm else None
        run_glossary_path = (
            self._resolve_optional_path(run_glossary, 'glossary') if run_glossary else None
        )

        resolved: List[ResolvedTarget] = []
        first_original: Dict[str, int] = {}
        result_keys: Dict[str, int] = {}

        for raw_target in raw.targets:
            target = self._resolve_target(
                raw_target,
                run_stage=run_stage,
                run_strict=run_strict,
                run_llm_path=run_llm_path,
                run_glossary_path=run_glossary_path,
                default_llm=default_llm,
            )
            if target is None:
                continue

            original_key = os.path.normcase(target.original)
            if original_key in first_original:
                self._violations.append(ManifestViolation(
                    f"duplicate original of target {first_original[original_key]}",
                    target_index=target.index,
                ))
            else:
                first_original[original_key] = target.index

            result_key = os.path.normcase(
                portable_relative_path(target.relative_path))
            if result_key in result_keys:
                self._violations.append(ManifestViolation(
                    f"result path collides with target {result_keys[result_key]} "
                    f"({target.relative_path})",
                    target_index=target.index,
                ))
            else:
                result_keys[result_key] = target.index

            resolved.append(target)

        if result2target is None:
            result2target = False
        if cache2target is None:
            cache2target = False

        # Inputs must not sit under the chosen result root.
        result_root = (
            os.path.join(self.base_dir, RESULT_DIR_NAME)
            if result2target
            else os.path.join(self.conductor_dir, RESULT_DIR_NAME, os.path.basename(self.manifest_path))
        )
        for target in resolved:
            for label, path in (('original', target.original), ('draft', target.draft)):
                if path is not None and is_inside(path, result_root):
                    self._violations.append(ManifestViolation(
                        f"{label} lies inside the result directory {result_root}",
                        target_index=target.index,
                    ))

        if self._violations:
            self._raise()

        return RunSettings(
            path=self.manifest_path,
            result2target=result2target,
            cache2target=cache2target,
            verb=verb,
            targets=tuple(resolved),
        )

    def _resolve_target(
            self,
            raw: _RawTarget,
            *,
            run_stage: Optional[str],
            run_strict: Optional[bool],
            run_llm_path: Optional[str],
            run_glossary_path: Optional[str],
            default_llm: str,
    ) -> Optional[ResolvedTarget]:
        """Resolve one target; return None when original is unusable."""
        index = raw.index
        original_raw = self._required_string(raw.original, 'original', target_index=index)
        if original_raw is None:
            return None

        try:
            original = resolve_under_base(self.base_dir, original_raw)
        except ValueError as e:
            self._violations.append(ManifestViolation(str(e), target_index=index))
            return None

        # Conflict keys off the draft *field*, not a successful path resolve,
        # so a bad draft path still collects "draft implies verify" with stage.
        has_draft_field = raw.draft is not None
        draft_raw = self._optional_string(raw.draft, 'draft', target_index=index)
        draft: Optional[str] = None
        if draft_raw is not None:
            try:
                draft = resolve_under_base(self.base_dir, draft_raw)
            except ValueError as e:
                self._violations.append(ManifestViolation(str(e), target_index=index))

        target_stage = self._optional_string(
            raw.stage, 'stage', allowed=VALID_RUN_STAGES, target_index=index)
        target_strict = self._optional_bool(raw.strict, 'strict', target_index=index)
        target_llm = self._optional_string(raw.llm_config, 'llm_config', target_index=index)
        target_glossary = self._optional_string(raw.glossary, 'glossary', target_index=index)

        if has_draft_field and target_stage is not None:
            self._violations.append(ManifestViolation(
                "draft implies verify; omit stage on this target",
                target_index=index,
            ))

        if has_draft_field:
            stage = RUN_STAGE_VERIFY
        else:
            stage = target_stage if target_stage is not None else (
                run_stage if run_stage is not None else RUN_STAGE_ALL
            )

        strict = target_strict if target_strict is not None else (
            run_strict if run_strict is not None else False
        )

        if target_llm is not None:
            llm_config = self._resolve_optional_path(
                target_llm, 'llm_config', target_index=index) or default_llm
        elif run_llm_path is not None:
            llm_config = run_llm_path
        else:
            llm_config = default_llm

        if target_glossary is not None:
            glossary = self._resolve_optional_path(
                target_glossary, 'glossary', target_index=index)
        elif run_glossary_path is not None:
            glossary = run_glossary_path
        else:
            candidate = os.path.join(self.base_dir, GLOSSARY_FILENAME)
            glossary = candidate if os.path.isfile(candidate) else None

        return ResolvedTarget(
            index=index,
            original=original,
            relative_path=manifest_relative_identity(original, self.base_dir),
            draft=draft,
            stage=stage,
            strict=strict,
            llm_config=llm_config,
            glossary=glossary,
        )

    def _resolve_optional_path(
            self, relative: str, field: str,
            target_index: Optional[int] = None) -> Optional[str]:
        """Resolve a manifest-relative path field; record violations."""
        try:
            return resolve_under_base(self.base_dir, relative)
        except ValueError as e:
            self._violations.append(ManifestViolation(
                f"{field}: {e}", target_index=target_index))
            return None

    def _required_string(
            self, value: Any, field: str,
            target_index: Optional[int] = None) -> Optional[str]:
        """Require a non-empty string field."""
        if value is None:
            self._violations.append(ManifestViolation(
                f"missing '{field}'", target_index=target_index))
            return None
        if not isinstance(value, str) or not value.strip():
            self._violations.append(ManifestViolation(
                f"'{field}' must be a non-empty string", target_index=target_index))
            return None
        return value

    def _optional_string(
            self, value: Any, field: str, *,
            allowed: Optional[Tuple[str, ...]] = None,
            target_index: Optional[int] = None) -> Optional[str]:
        """Optional string; optionally restricted to ``allowed`` (case-sensitive)."""
        if value is None:
            return None
        if not isinstance(value, str):
            self._violations.append(ManifestViolation(
                f"'{field}' must be a string", target_index=target_index))
            return None
        if allowed is not None and value not in allowed and value.upper() not in allowed:
            # verb is stored uppercased; accept any VALID_LOG_LEVELS case.
            if field == 'verb' and value.upper() in allowed:
                return value
            self._violations.append(ManifestViolation(
                f"'{field}' must be one of {', '.join(allowed)}",
                target_index=target_index,
            ))
            return None
        return value

    def _optional_bool(
            self, value: Any, field: str,
            target_index: Optional[int] = None) -> Optional[bool]:
        """Optional boolean field."""
        if value is None:
            return None
        if not isinstance(value, bool):
            self._violations.append(ManifestViolation(
                f"'{field}' must be a boolean", target_index=target_index))
            return None
        return value

    def _raise(self) -> None:
        """Raise TargetError with every collected violation."""
        details = "\n".join(str(v) for v in self._violations)
        raise TargetError(
            f"Run manifest {self.manifest_path} is invalid:\n{details}"
        )
