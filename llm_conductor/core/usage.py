# -*- coding: utf-8 -*-
"""
Per-attempt token usage ledger for cost estimation exports.

Separate from MetricsCollector: that collector keeps a flat scalar snapshot
per stage; this ledger keeps every HTTP attempt (including retries) plus
preflight and non-API leaf counts.

Attempt ids are stable handles so a later validation reject can reclassify
the matching HTTP success without depending on “last write wins” — required
once leaves may run concurrently (cloud async roadmap).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_conductor.constants import (
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    USAGE_NON_API_CACHE_HIT,
    USAGE_NON_API_DEDUP,
    USAGE_OUTCOME_FAILED,
    USAGE_OUTCOME_REJECTED,
    USAGE_OUTCOME_SUCCESS,
    USAGE_REPORT_SCHEMA_VERSION,
    USAGE_STAGE_PREFLIGHT,
)


@dataclass(frozen=True)
class UsageAttempt:
    """One HTTP attempt (or preflight probe) with optional provider usage."""

    id: int
    provider: str
    model: str
    stage: str
    attempt: int
    outcome: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    failure_kind: Optional[str] = None
    usage_unknown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form."""
        return {
            'id': self.id,
            'provider': self.provider,
            'model': self.model,
            'stage': self.stage,
            'attempt': self.attempt,
            'outcome': self.outcome,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'failure_kind': self.failure_kind,
            'usage_unknown': self.usage_unknown,
        }


class UsageLedger:
    """Thread-safe accumulator of API attempts and non-API leaf shortcuts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        # id -> attempt; insertion order kept in _order for stable exports.
        self._attempts: Dict[int, UsageAttempt] = {}
        self._order: List[int] = []
        self._non_api: Dict[str, Dict[str, int]] = {
            OPERATION_TRANSLATION: {USAGE_NON_API_CACHE_HIT: 0, USAGE_NON_API_DEDUP: 0},
            OPERATION_VERIFICATION: {USAGE_NON_API_CACHE_HIT: 0, USAGE_NON_API_DEDUP: 0},
        }

    def record_attempt(
        self,
        *,
        provider: str,
        model: str,
        stage: str,
        attempt: int,
        outcome: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        failure_kind: Optional[str] = None,
        usage_unknown: bool = False,
    ) -> int:
        """
        Append one attempt and return its stable id.

        Pass usage_unknown when the provider omitted usage. Callers that may
        later reclassify the attempt (content validation) must keep the id.
        """
        if prompt_tokens is None and completion_tokens is None and not usage_unknown:
            # Caller did not report tokens: treat as unknown unless both are explicit 0.
            usage_unknown = True
            prompt_tokens = None
            completion_tokens = None
        with self._lock:
            attempt_id = self._next_id
            self._next_id += 1
            self._attempts[attempt_id] = UsageAttempt(
                id=attempt_id,
                provider=provider,
                model=model,
                stage=stage,
                attempt=attempt,
                outcome=outcome,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                failure_kind=failure_kind,
                usage_unknown=usage_unknown,
            )
            self._order.append(attempt_id)
            return attempt_id

    def record_non_api(self, stage: str, kind: str, count: int = 1) -> None:
        """Count cache hits / dedup leaves that did not issue a new API request."""
        if stage not in self._non_api or kind not in self._non_api[stage]:
            return
        with self._lock:
            self._non_api[stage][kind] += count

    def mark_attempt_rejected(
        self,
        attempt_id: int,
        *,
        failure_kind: Optional[str] = None,
    ) -> bool:
        """
        Reclassify a prior successful HTTP attempt as rejected by validation.

        Looks up by attempt id (not “last success”), so concurrent leaves
        cannot clobber each other's outcomes. Tokens already recorded stay
        for cost; only ``outcome`` / ``failure_kind`` change.

        Returns:
            True when a successful attempt with that id was updated.
        """
        with self._lock:
            current = self._attempts.get(attempt_id)
            if current is None or current.outcome != USAGE_OUTCOME_SUCCESS:
                return False
            self._attempts[attempt_id] = UsageAttempt(
                id=current.id,
                provider=current.provider,
                model=current.model,
                stage=current.stage,
                attempt=current.attempt,
                outcome=USAGE_OUTCOME_REJECTED,
                prompt_tokens=current.prompt_tokens,
                completion_tokens=current.completion_tokens,
                failure_kind=failure_kind or current.failure_kind,
                usage_unknown=current.usage_unknown,
            )
            return True

    def stage_token_totals(self) -> Dict[str, Dict[str, int]]:
        """
        Aggregate known tokens per pipeline stage (excludes preflight).

        Unknown-usage attempts contribute 0 to the sums but are counted in
        the export's unknown_attempts field.
        """
        totals = {
            OPERATION_TRANSLATION: {
                'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            },
            OPERATION_VERIFICATION: {
                'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            },
        }
        with self._lock:
            for attempt_id in self._order:
                attempt = self._attempts[attempt_id]
                if attempt.stage not in totals or attempt.usage_unknown:
                    continue
                prompt = attempt.prompt_tokens or 0
                completion = attempt.completion_tokens or 0
                bucket = totals[attempt.stage]
                bucket['prompt_tokens'] += prompt
                bucket['completion_tokens'] += completion
                bucket['total_tokens'] += prompt + completion
        return totals

    def build_export(self) -> Dict[str, Any]:
        """Machine-readable usage report for cost re-pricing."""
        with self._lock:
            attempts = [self._attempts[attempt_id] for attempt_id in self._order]
            non_api = {
                stage: dict(counts) for stage, counts in self._non_api.items()
            }

        by_key: Dict[tuple, Dict[str, Any]] = {}
        unknown_attempts = 0
        for attempt in attempts:
            if attempt.usage_unknown:
                unknown_attempts += 1
            key = (attempt.provider, attempt.model, attempt.stage)
            bucket = by_key.setdefault(key, {
                'provider': attempt.provider,
                'model': attempt.model,
                'stage': attempt.stage,
                'attempts': 0,
                'successful_attempts': 0,
                'rejected_attempts': 0,
                'failed_attempts': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'unknown_usage_attempts': 0,
            })
            bucket['attempts'] += 1
            if attempt.outcome == USAGE_OUTCOME_SUCCESS:
                bucket['successful_attempts'] += 1
            elif attempt.outcome == USAGE_OUTCOME_REJECTED:
                bucket['rejected_attempts'] += 1
            elif attempt.outcome == USAGE_OUTCOME_FAILED:
                bucket['failed_attempts'] += 1
            if attempt.usage_unknown:
                bucket['unknown_usage_attempts'] += 1
            else:
                prompt = attempt.prompt_tokens or 0
                completion = attempt.completion_tokens or 0
                bucket['prompt_tokens'] += prompt
                bucket['completion_tokens'] += completion
                bucket['total_tokens'] += prompt + completion

        preflight = [row for row in by_key.values() if row['stage'] == USAGE_STAGE_PREFLIGHT]
        stages = [row for row in by_key.values() if row['stage'] != USAGE_STAGE_PREFLIGHT]

        return {
            'schema_version': USAGE_REPORT_SCHEMA_VERSION,
            'by_provider_model_stage': stages,
            'preflight': preflight,
            'unknown_usage_attempts': unknown_attempts,
            'non_api': non_api,
            'attempts': [attempt.to_dict() for attempt in attempts],
        }
