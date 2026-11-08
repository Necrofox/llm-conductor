# -*- coding: utf-8 -*-
"""
Live tier: assertions on run results against a real model.

Run only after the user confirms inference is ready (AGENTS.md (live tier)):
    $env:LLMC_LIVE_CONFIRMED = "1"; pytest -m live

Two classes of checks:
  * CODE guarantees — hard: structure and key order, non-strings and empty
    strings untouched, failed_requests == 0. A cached rerun makes no model
    requests only when the prior full run had no content-validation rejects
    (rejects are never cached). A failure here is a pipeline bug or
    infrastructure outage.
  * MODEL behaviour — non-strict xfail with a tracker ref: lost tags,
    chatter, foreign scripts, anomalous length. The content validator
    already rejects many of these; residual failures are non-deterministic
    model replies, so a strict-xfail would flap on a lucky answer.
Cyrillic in every accepted reply is hard: without it the run is meaningless.
Leaves degraded to the source after a validation reject stay in the source
language and are excluded from that check.
"""
import pytest

from llm_conductor.constants import EXIT_FAILURE_RATIO_EXCEEDED, EXIT_SUCCESS
from tests.live.runner import (
    CYRILLIC,
    foreign_latin_violations,
    foreign_script_violations,
    length_violations,
    paired_string_leaves,
    structure_violations,
    tag_violations,
)

PIPELINE_RUNS = ["full", "full_cached", "strict", "verify_only"]
FRESH_RUNS = ["full", "strict", "verify_only"]

MODEL_BEHAVIOUR = pytest.mark.xfail(
    strict=False,
    reason=(
        "residual model instability (tags/chatter/scripts/length) "
        "after content validation — replies are non-deterministic"
    ),
)


def _final(run):
    return run.outputs["sample_final.json"]


@pytest.fixture(autouse=True)
def _report(live_report):
    """The report is built once before the first assertion — even if assertions fail."""
    return live_report


# ============================================================================
# Code guarantees
# ============================================================================

@pytest.mark.parametrize("name", FRESH_RUNS)
def test_run_succeeds_and_writes_final(live_runs, name):
    run = live_runs[name]

    assert run.exit_code == EXIT_SUCCESS
    assert "sample_final.json" in run.outputs


def test_cached_rerun_writes_final(live_runs):
    """
    A cache rerun always writes final. Exit 0 and an identical tree hold only
    when the prior full run cached every leaf; residual validation rejects
    are re-queried and may push the failure ratio over the threshold (exit 5).
    """
    full = live_runs["full"]
    cached = live_runs["full_cached"]

    assert "sample_final.json" in cached.outputs
    if full.rejected_responses() == 0:
        assert cached.exit_code == EXIT_SUCCESS
        assert _final(cached) == _final(full)
    else:
        assert cached.exit_code in (EXIT_SUCCESS, EXIT_FAILURE_RATIO_EXCEEDED)


@pytest.mark.parametrize("name", PIPELINE_RUNS)
def test_structure_order_and_non_strings_preserved(live_runs, source_data, name):
    violations = structure_violations(source_data, _final(live_runs[name]))

    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize("name", FRESH_RUNS)
def test_no_failed_requests(live_runs, name):
    run = live_runs[name]

    assert run.model_requests() > 0, "прогон на чистом кэше не обратился к модели"
    assert run.failed_requests() == 0, run.metrics


def test_cached_rerun_model_requests_match_prior_rejects(live_runs):
    """
    Cache completeness is not guaranteed once the content validator rejects:
    only residual misses are re-queried. With a clean first run the rerun
    must hit the model zero times.
    """
    full = live_runs["full"]
    cached = live_runs["full_cached"]
    prior_rejects = full.rejected_responses()

    if prior_rejects == 0:
        assert cached.model_requests() == 0
        assert _final(cached) == _final(full)
    else:
        assert cached.model_requests() > 0


def test_verify_only_did_not_translate_again(live_runs):
    metrics = live_runs["verify_only"].metrics
    assert metrics["translation"]["total_requests"] == 0
    assert metrics["verification"]["total_requests"] > 0


def test_translate_only_does_not_verify(live_runs):
    run = live_runs["strict"]

    assert run.metrics["verification"]["total_requests"] == 0
    assert _final(run) == run.outputs["sample_translated.json"]


@pytest.mark.parametrize("name", FRESH_RUNS)
def test_every_answer_is_russian(live_runs, source_data, name):
    """
    Every leaf the pipeline kept as a successful reply must contain Cyrillic.
    Leaves degraded to the source after a validation reject stay English —
    that is intentional (§4.4а), not a missing translation.
    """
    run = live_runs[name]
    missing = [
        f"{path}: {out!r}"
        for path, src, out in paired_string_leaves(source_data, _final(run))
        if not CYRILLIC.search(out) and out != src
    ]

    assert missing == [], "\n".join(missing)


# ============================================================================
# Model behaviour — known residual defects
# ============================================================================

@MODEL_BEHAVIOUR
@pytest.mark.parametrize("name", FRESH_RUNS)
def test_model_preserves_tags(live_runs, source_data, name):
    violations = tag_violations(source_data, _final(live_runs[name]))

    assert violations == [], "\n".join(violations)


@MODEL_BEHAVIOUR
@pytest.mark.parametrize("name", FRESH_RUNS)
def test_model_adds_no_foreign_latin_text(live_runs, source_data, name):
    violations = foreign_latin_violations(source_data, _final(live_runs[name]))

    assert violations == [], "\n".join(violations)


@MODEL_BEHAVIOUR
@pytest.mark.parametrize("name", FRESH_RUNS)
def test_model_answers_without_third_scripts(live_runs, source_data, name):
    """CJK and similar — a 'has Cyrillic' check misses them when Cyrillic is also present."""
    violations = foreign_script_violations(source_data, _final(live_runs[name]))

    assert violations == [], "\n".join(violations)


@MODEL_BEHAVIOUR
@pytest.mark.parametrize("name", FRESH_RUNS)
def test_model_answer_length_is_proportional(live_runs, source_data, name):
    """'Hard' -> a paragraph of reasoning: a non-empty str still counts as a successful reply."""
    violations = length_violations(source_data, _final(live_runs[name]))

    assert violations == [], "\n".join(violations)
