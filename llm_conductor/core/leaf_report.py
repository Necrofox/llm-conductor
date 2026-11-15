# -*- coding: utf-8 -*-
"""
Per-leaf outcome recording, sidecars, and the run-level leaf report.

Status comes from request/validation outcomes only — never from comparing
the output string to the source (names and ids may legally stay identical).

Leaf keys in schema v2 are ``KeyPath.report_key()`` JSON segment arrays.
Schema v1 dotted keys (``ui.title``) are rejected at load time — re-run
``translate`` to regenerate v2 sidecars. New writes always use v2 keys.

File identities (``run_report.json`` ``files`` keys and sidecar
``relative_path``) use POSIX separators. Legacy native-separator identities
are accepted on read and rewritten in portable form; schema version stays 2.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_conductor.constants import (
    LEAF_ERROR_KIND_SOURCE_STALE,
    LEAF_REPORT_SCHEMA_VERSION,
    LEAF_SKIP_REASON_TRANSLATION_DEGRADED,
    LEAF_STATUS_CACHE_HIT,
    LEAF_STATUS_DEDUP,
    LEAF_STATUS_DEGRADED,
    LEAF_STATUS_OK,
    LEAF_STATUS_SKIPPED,
    LEAF_STATUS_VERIFY_SKIPPED,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)


def hash_text(value: str) -> str:
    """Stable hex digest of a leaf string for machine reports (not for cache keys)."""
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class LeafStageRecord:
    """Outcome of one pipeline stage for one leaf path."""

    status: str
    cache_hit: bool = False
    degraded: bool = False
    error_kind: Optional[str] = None
    source_hash: Optional[str] = None
    output_hash: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form (omit null optional fields)."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def make_ok(source: str, output: str, *, cache_hit: bool = False,
            dedup: bool = False) -> LeafStageRecord:
    """Accepted leaf (model, cache hit, or in-run dedup)."""
    if cache_hit:
        status = LEAF_STATUS_CACHE_HIT
    elif dedup:
        status = LEAF_STATUS_DEDUP
    else:
        status = LEAF_STATUS_OK
    return LeafStageRecord(
        status=status,
        cache_hit=cache_hit,
        degraded=False,
        source_hash=hash_text(source),
        output_hash=hash_text(output),
    )


def make_skipped(source: str) -> LeafStageRecord:
    """Blank or ineligible leaf under the active content policy; no API call."""
    digest = hash_text(source)
    return LeafStageRecord(
        status=LEAF_STATUS_SKIPPED,
        source_hash=digest,
        output_hash=digest,
    )


def make_degraded(source: str, error_kind: Optional[str]) -> LeafStageRecord:
    """Transport/validation/unexpected failure; stage input kept (or restored)."""
    digest = hash_text(source)
    return LeafStageRecord(
        status=LEAF_STATUS_DEGRADED,
        degraded=True,
        error_kind=error_kind,
        source_hash=digest,
        output_hash=digest,
    )


def make_verify_skipped(
        source: str,
        *,
        reason: str = LEAF_SKIP_REASON_TRANSLATION_DEGRADED,
) -> LeafStageRecord:
    """Normal verification skipped (degraded translation or never attempted)."""
    digest = hash_text(source)
    return LeafStageRecord(
        status=LEAF_STATUS_VERIFY_SKIPPED,
        source_hash=digest,
        output_hash=digest,
        reason=reason,
    )


def make_source_stale(original: str) -> LeafStageRecord:
    """Verification refused: translation ``source_hash`` no longer matches original."""
    return make_degraded(original, LEAF_ERROR_KIND_SOURCE_STALE)


def portable_file_identity(relative_path: str) -> str:
    """
    Normalize a target-relative file identity to POSIX separators.

    Machine reports and sidecars always emit this form. Older Windows
    artifacts that used ``\\\\`` are accepted by mapping them here so
    lookup and merge do not drop records (#33).
    """
    return relative_path.replace('\\', '/')


def _is_v2_report_key(key: str) -> bool:
    """True when ``key`` is a schema v2 ``KeyPath.report_key()`` JSON array."""
    try:
        parsed = json.loads(key)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(parsed, list)


def legacy_sidecar_v1_reason(payload: Dict[str, Any]) -> Optional[str]:
    """
    Return a short reason when ``payload`` is a schema v1 leaves sidecar.

    Detection: explicit ``schema_version == 1``, or any leaf identity that is
    not a v2 ``KeyPath.report_key()`` JSON array (dotted keys like ``ui.title``).
    """
    version = payload.get('schema_version')
    if version == 1:
        return "schema_version is 1 (dotted leaf keys)"
    leaves = payload.get('leaves')
    if isinstance(leaves, dict):
        for key in leaves:
            if isinstance(key, str) and not _is_v2_report_key(key):
                return (
                    f"leaf key {key!r} uses schema v1 dotted identity; "
                    f"expected KeyPath.report_key() JSON arrays"
                )
    return None


def _record_from_dict(raw: Any) -> Optional[LeafStageRecord]:
    """Parse a sidecar stage dict; ignore malformed entries."""
    if not isinstance(raw, dict):
        return None
    status = raw.get('status')
    if not isinstance(status, str):
        return None
    return LeafStageRecord(
        status=status,
        cache_hit=bool(raw.get('cache_hit', False)),
        degraded=bool(raw.get('degraded', False)),
        error_kind=raw.get('error_kind') if isinstance(raw.get('error_kind'), str) else None,
        source_hash=raw.get('source_hash') if isinstance(raw.get('source_hash'), str) else None,
        output_hash=raw.get('output_hash') if isinstance(raw.get('output_hash'), str) else None,
        reason=raw.get('reason') if isinstance(raw.get('reason'), str) else None,
    )


class LeafOutcomeRecorder:
    """
    Thread-safe collector of per-leaf stage outcomes for one run.

    File scoping: call ``set_active_file`` before processing a target entry so
    leaf paths (JSON keys) do not collide across files. Sidecar merge loads
    prior translation statuses for ``--stage verify``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # relative_path -> leaf_path -> stage -> LeafStageRecord
        self._files: Dict[str, Dict[str, Dict[str, LeafStageRecord]]] = {}
        self._active_file: Optional[str] = None
        self._sidecar_missing_logged = False
        # Files whose translation records were loaded this run (translation_missing
        # applies only to those files — never leak across a mixed manifest).
        self._sidecar_merged_files: Set[str] = set()

    def set_active_file(self, relative_path: str) -> None:
        """Select the target-relative file path for subsequent record/lookup calls."""
        file_key = portable_file_identity(relative_path)
        with self._lock:
            self._active_file = file_key
            self._files.setdefault(file_key, {})

    def active_file(self) -> Optional[str]:
        """Return the portable target-relative path selected by ``set_active_file``, or None."""
        with self._lock:
            return self._active_file

    def note_sidecar_missing(self) -> bool:
        """
        Mark that a verify-only run lacked a leaves sidecar.

        Returns:
            True the first time this is noted (caller should log once).
        """
        with self._lock:
            if self._sidecar_missing_logged:
                return False
            self._sidecar_missing_logged = True
            return True

    def sidecar_was_merged(self) -> bool:
        """True when a translation record was loaded for the active file."""
        with self._lock:
            file_key = self._active_file or ''
            return file_key in self._sidecar_merged_files

    def record(self, leaf_path: str, stage: str, record: LeafStageRecord) -> None:
        """Store the outcome of one stage for the active file's leaf."""
        with self._lock:
            file_key = self._active_file or ''
            leaves = self._files.setdefault(file_key, {})
            stages = leaves.setdefault(leaf_path, {})
            stages[stage] = record

    def get_translation_record(self, leaf_path: str) -> Optional[LeafStageRecord]:
        """Return the active file's translation stage record, or None."""
        with self._lock:
            file_key = self._active_file or ''
            return self._files.get(file_key, {}).get(leaf_path, {}).get(OPERATION_TRANSLATION)

    def is_translation_degraded(self, leaf_path: str) -> bool:
        """True when the active file's translation stage recorded degraded for this leaf."""
        with self._lock:
            file_key = self._active_file or ''
            record = self._files.get(file_key, {}).get(leaf_path, {}).get(OPERATION_TRANSLATION)
            if record is None:
                return False
            return record.degraded or record.status == LEAF_STATUS_DEGRADED

    def merge_sidecar(self, relative_path: str, payload: Dict[str, Any]) -> None:
        """
        Merge a previously written leaves sidecar into the in-memory map.

        Existing in-memory stage records for the same leaf win over sidecar
        entries (same-run translate then verify must not be overwritten).
        A verification-only sidecar cannot establish translation provenance.
        Callers must reject schema v1 payloads before invoking this method;
        new recordings use v2 ``KeyPath.report_key()`` identities.
        File identity separators are normalized so a legacy native-path
        sidecar still merges under the portable key.
        """
        file_key = portable_file_identity(relative_path)
        leaves_raw = payload.get('leaves')
        if not isinstance(leaves_raw, dict):
            return
        with self._lock:
            file_leaves = self._files.setdefault(file_key, {})
            for leaf_path, stages_raw in leaves_raw.items():
                if not isinstance(leaf_path, str) or not isinstance(stages_raw, dict):
                    continue
                stages = file_leaves.setdefault(leaf_path, {})
                for stage, stage_raw in stages_raw.items():
                    if stage in stages:
                        continue
                    parsed = _record_from_dict(stage_raw)
                    if parsed is not None:
                        stages[stage] = parsed
                        if stage == OPERATION_TRANSLATION:
                            self._sidecar_merged_files.add(file_key)

    def file_sidecar_dict(self, relative_path: str) -> Dict[str, Any]:
        """Build the JSON document written under ``leaves/<relative>``."""
        file_key = portable_file_identity(relative_path)
        with self._lock:
            leaves = self._files.get(file_key, {})
            serialized: Dict[str, Dict[str, Any]] = {}
            for leaf_path, stages in leaves.items():
                serialized[leaf_path] = {
                    stage: record.to_dict() for stage, record in stages.items()
                }
        return {
            'schema_version': LEAF_REPORT_SCHEMA_VERSION,
            'relative_path': file_key,
            'leaves': serialized,
        }

    def aggregates(self) -> Dict[str, int]:
        """
        Run-wide counters for the human banner and fail_on_any_error.

        ``untranslated`` / ``degraded`` count translation-stage degradations
        (source kept) plus verification ``source_stale``. ``verify_skipped``
        counts leaves that skipped normal verification
        (``translation_degraded`` or ``translation_missing``).
        ``skipped`` counts content/blank stage records (a leaf may contribute
        once per stage). ``api_calls`` counts leaf stage outcomes with status
        ``ok`` or ``degraded`` — not HTTP attempts (see ``usage.json``).
        """
        with self._lock:
            return self._aggregates_unlocked()

    def _aggregates_unlocked(self) -> Dict[str, int]:
        totals = {
            'leaves': 0,
            'ok': 0,
            'skipped': 0,
            'cache_hit': 0,
            'dedup': 0,
            'degraded': 0,
            'untranslated': 0,
            'verify_skipped': 0,
            'api_calls': 0,
        }
        seen_leaves: set[Tuple[str, str]] = set()
        for file_key, leaves in self._files.items():
            for leaf_path, stages in leaves.items():
                leaf_id = (file_key, leaf_path)
                if leaf_id not in seen_leaves:
                    seen_leaves.add(leaf_id)
                    totals['leaves'] += 1
                translation = stages.get(OPERATION_TRANSLATION)
                verification = stages.get(OPERATION_VERIFICATION)
                for record in stages.values():
                    if record.status == LEAF_STATUS_OK:
                        totals['ok'] += 1
                        totals['api_calls'] += 1
                    elif record.status == LEAF_STATUS_SKIPPED:
                        totals['skipped'] += 1
                    elif record.status == LEAF_STATUS_CACHE_HIT:
                        totals['cache_hit'] += 1
                    elif record.status == LEAF_STATUS_DEDUP:
                        totals['dedup'] += 1
                    elif record.status == LEAF_STATUS_DEGRADED:
                        totals['degraded'] += 1
                        totals['api_calls'] += 1
                    elif record.status == LEAF_STATUS_VERIFY_SKIPPED:
                        totals['verify_skipped'] += 1
                if translation is not None and (
                    translation.degraded or translation.status == LEAF_STATUS_DEGRADED
                ):
                    totals['untranslated'] += 1
                elif (
                    verification is not None
                    and (verification.degraded or verification.status == LEAF_STATUS_DEGRADED)
                    and translation is None
                ):
                    # Verify-only degradation of a leaf with no translation record.
                    totals['untranslated'] += 1
                elif (
                    verification is not None
                    and verification.error_kind == LEAF_ERROR_KIND_SOURCE_STALE
                ):
                    totals['untranslated'] += 1
        return totals

    def has_untreated_error(self) -> bool:
        """True when any leaf stage ended degraded (untreated error)."""
        with self._lock:
            for leaves in self._files.values():
                for stages in leaves.values():
                    for record in stages.values():
                        if record.degraded or record.status == LEAF_STATUS_DEGRADED:
                            return True
        return False

    def build_run_report(self) -> Dict[str, Any]:
        """Machine-readable run report aggregating every recorded leaf."""
        with self._lock:
            files_out: Dict[str, Any] = {}
            for relative_path, leaves in self._files.items():
                file_key = portable_file_identity(relative_path)
                # Fold a legacy native-separator key onto the portable identity
                # so an older in-memory map does not emit a duplicate or drop
                # leaf stages when both forms somehow coexist.
                target = files_out.setdefault(file_key, {})
                for leaf_path, stages in leaves.items():
                    leaf_out = target.setdefault(leaf_path, {})
                    for stage, record in stages.items():
                        leaf_out.setdefault(stage, record.to_dict())
            aggregates = self._aggregates_unlocked()
        return {
            'schema_version': LEAF_REPORT_SCHEMA_VERSION,
            'aggregates': aggregates,
            'aggregates_api_calls': {
                'degraded': aggregates['degraded'],
                'ok': aggregates['ok'],
                'total_recorded_api_leaf_outcomes': aggregates['api_calls'],
            },
            'files': files_out,
        }

    def banner_lines(self) -> List[str]:
        """Human-readable leaf outcome summary lines for the final banner."""
        totals = self.aggregates()
        return [
            "LEAF OUTCOME SUMMARY",
            f"   - Leaves:            {totals['leaves']:,}",
            f"   - Untranslated:      {totals['untranslated']:,}",
            f"   - Degraded:          {totals['degraded']:,}",
            f"   - Verify skipped:    {totals['verify_skipped']:,}",
            f"   - Cache hits:        {totals['cache_hit']:,}",
            f"   - Deduplicated:      {totals['dedup']:,}",
            f"   - Content/blank skip: {totals['skipped']:,}",
        ]
