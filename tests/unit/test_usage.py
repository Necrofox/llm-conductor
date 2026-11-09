# -*- coding: utf-8 -*-
"""Unit tests for UsageLedger."""
from llm_conductor.constants import (
    OPERATION_TRANSLATION,
    USAGE_NON_API_CACHE_HIT,
    USAGE_OUTCOME_FAILED,
    USAGE_OUTCOME_REJECTED,
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
    assert row["rejected_attempts"] == 0
    assert row["failed_attempts"] == 1
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 4
    assert row["unknown_usage_attempts"] == 1

    totals = ledger.stage_token_totals()[OPERATION_TRANSLATION]
    assert totals["prompt_tokens"] == 10
    assert totals["completion_tokens"] == 4
    assert totals["total_tokens"] == 14


def test_mark_attempt_rejected_by_id_is_concurrency_safe():
    """
    Two successful attempts; rejecting the earlier id must not touch the later
    one (would break under parallel leaves if we used “last success”).
    """
    ledger = UsageLedger()
    first = ledger.record_attempt(
        provider="ollama", model="a", stage=OPERATION_TRANSLATION,
        attempt=1, outcome=USAGE_OUTCOME_SUCCESS,
        prompt_tokens=3, completion_tokens=1,
    )
    second = ledger.record_attempt(
        provider="ollama", model="b", stage=OPERATION_TRANSLATION,
        attempt=1, outcome=USAGE_OUTCOME_SUCCESS,
        prompt_tokens=5, completion_tokens=2,
    )

    assert ledger.mark_attempt_rejected(first, failure_kind="newline_in_reply") is True
    assert ledger.mark_attempt_rejected(first, failure_kind="again") is False

    export = ledger.build_export()
    by_id = {row["id"]: row for row in export["attempts"]}
    assert by_id[first]["outcome"] == USAGE_OUTCOME_REJECTED
    assert by_id[first]["failure_kind"] == "newline_in_reply"
    assert by_id[first]["prompt_tokens"] == 3
    assert by_id[second]["outcome"] == USAGE_OUTCOME_SUCCESS

    row = export["by_provider_model_stage"]
    # Same stage key appears once per provider/model/stage — two models → two rows.
    assert sum(item["rejected_attempts"] for item in row) == 1
    assert sum(item["successful_attempts"] for item in row) == 1
    assert sum(item["prompt_tokens"] for item in row) == 8
