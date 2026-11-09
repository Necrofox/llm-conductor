# -*- coding: utf-8 -*-
"""Unit tests for UsageLedger."""
from llm_conductor.constants import (
    OPERATION_TRANSLATION,
    USAGE_NON_API_CACHE_HIT,
    USAGE_OUTCOME_FAILED,
    USAGE_OUTCOME_SUCCESS,
    USAGE_STAGE_PREFLIGHT,
)
from llm_conductor.core.usage import UsageLedger


def test_attempts_aggregate_by_provider_model_stage_and_mark_unknown():
    ledger = UsageLedger()
    ledger.record_attempt(
        provider="ollama", model="qwen", stage=OPERATION_TRANSLATION,
        attempt=1, outcome=USAGE_OUTCOME_FAILED, usage_unknown=True,
    )
    ledger.record_attempt(
        provider="ollama", model="qwen", stage=OPERATION_TRANSLATION,
        attempt=2, outcome=USAGE_OUTCOME_SUCCESS,
        prompt_tokens=10, completion_tokens=4,
    )
    ledger.record_attempt(
        provider="ollama", model="qwen", stage=USAGE_STAGE_PREFLIGHT,
        attempt=1, outcome=USAGE_OUTCOME_SUCCESS,
        prompt_tokens=1, completion_tokens=1,
    )
    ledger.record_non_api(OPERATION_TRANSLATION, USAGE_NON_API_CACHE_HIT, 3)

    export = ledger.build_export()
    assert export["unknown_usage_attempts"] == 1
    assert export["non_api"][OPERATION_TRANSLATION][USAGE_NON_API_CACHE_HIT] == 3
    assert len(export["preflight"]) == 1
    assert export["preflight"][0]["prompt_tokens"] == 1

    stage_rows = export["by_provider_model_stage"]
    assert len(stage_rows) == 1
    row = stage_rows[0]
    assert row["attempts"] == 2
    assert row["successful_attempts"] == 1
    assert row["failed_attempts"] == 1
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 4
    assert row["unknown_usage_attempts"] == 1

    totals = ledger.stage_token_totals()[OPERATION_TRANSLATION]
    assert totals["prompt_tokens"] == 10
    assert totals["completion_tokens"] == 4
    assert totals["total_tokens"] == 14
