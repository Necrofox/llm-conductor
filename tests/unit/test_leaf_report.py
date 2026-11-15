# -*- coding: utf-8 -*-
"""Unit tests for LeafOutcomeRecorder and helpers."""
from llm_conductor.constants import (
    LEAF_ERROR_KIND_SOURCE_STALE,
    LEAF_REPORT_SCHEMA_VERSION,
    LEAF_SKIP_REASON_TRANSLATION_DEGRADED,
    LEAF_SKIP_REASON_TRANSLATION_MISSING,
    LEAF_STATUS_DEGRADED,
    LEAF_STATUS_OK,
    LEAF_STATUS_VERIFY_SKIPPED,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
)
from llm_conductor.core.leaf_report import (
    LeafOutcomeRecorder,
    hash_text,
    legacy_sidecar_v1_reason,
    make_degraded,
    make_ok,
    make_source_stale,
    make_verify_skipped,
    portable_file_identity,
)


def test_active_file_round_trips_until_cleared_by_next_set():
    recorder = LeafOutcomeRecorder()
    assert recorder.active_file() is None
    recorder.set_active_file("one.json")
    assert recorder.active_file() == "one.json"
    recorder.set_active_file("two.json")
    assert recorder.active_file() == "two.json"


def test_translation_degraded_lookup_is_outcome_based_not_string_equality():
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file("a.json")
    # Legal identical translation must not count as degraded.
    recorder.record("name", OPERATION_TRANSLATION, make_ok("Necron", "Necron"))
    assert recorder.is_translation_degraded("name") is False

    recorder.record("desc", OPERATION_TRANSLATION, make_degraded("Hello", "rejected"))
    assert recorder.is_translation_degraded("desc") is True


def test_sidecar_round_trip_preserves_translation_degraded():
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file("ui.json")
    recorder.record("t", OPERATION_TRANSLATION, make_degraded("Hello", "parse"))

    payload = recorder.file_sidecar_dict("ui.json")
    loaded = LeafOutcomeRecorder()
    loaded.merge_sidecar("ui.json", payload)
    loaded.set_active_file("ui.json")

    assert loaded.is_translation_degraded("t") is True
    assert payload["leaves"]["t"][OPERATION_TRANSLATION]["status"] == LEAF_STATUS_DEGRADED


def test_in_memory_stage_wins_over_sidecar_merge():
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file("ui.json")
    recorder.record("t", OPERATION_TRANSLATION, make_ok("a", "b"))

    recorder.merge_sidecar("ui.json", {
        "leaves": {
            "t": {
                OPERATION_TRANSLATION: make_degraded("a", "parse").to_dict(),
            }
        }
    })
    assert recorder.is_translation_degraded("t") is False


def test_aggregates_and_fail_on_any():
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file("f.json")
    recorder.record("ok", OPERATION_TRANSLATION, make_ok("a", "b"))
    recorder.record("bad", OPERATION_TRANSLATION, make_degraded("c", "network"))
    recorder.record("bad", OPERATION_VERIFICATION, make_verify_skipped("c"))

    totals = recorder.aggregates()
    assert totals["untranslated"] == 1
    assert totals["degraded"] == 1
    assert totals["verify_skipped"] == 1
    assert recorder.has_untreated_error() is True

    report = recorder.build_run_report()
    assert report["files"]["f.json"]["bad"][OPERATION_VERIFICATION]["status"] == LEAF_STATUS_VERIFY_SKIPPED
    assert report["files"]["f.json"]["bad"][OPERATION_VERIFICATION]["reason"] == (
        LEAF_SKIP_REASON_TRANSLATION_DEGRADED
    )
    assert report["files"]["f.json"]["ok"][OPERATION_TRANSLATION]["status"] == LEAF_STATUS_OK


def test_note_sidecar_missing_only_once():
    recorder = LeafOutcomeRecorder()
    assert recorder.note_sidecar_missing() is True
    assert recorder.note_sidecar_missing() is False


def test_sidecar_merge_sets_flag_and_exposes_translation_record():
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file("ui.json")
    assert recorder.sidecar_was_merged() is False
    recorder.merge_sidecar("ui.json", {
        "leaves": {
            '["t"]': {
                OPERATION_TRANSLATION: make_ok("Hello", "Привет").to_dict(),
            }
        }
    })
    assert recorder.sidecar_was_merged() is True
    record = recorder.get_translation_record('["t"]')
    assert record is not None
    assert record.source_hash == hash_text("Hello")
    assert recorder.get_translation_record('["missing"]') is None


def test_sidecar_merged_flag_is_scoped_to_active_file():
    """A sidecar for file A must not enable translation_missing on file B."""
    recorder = LeafOutcomeRecorder()
    recorder.merge_sidecar("a.json", {
        "leaves": {
            '["t"]': {
                OPERATION_TRANSLATION: make_ok("A", "А").to_dict(),
            }
        }
    })
    recorder.set_active_file("a.json")
    assert recorder.sidecar_was_merged() is True

    recorder.set_active_file("b.json")
    assert recorder.sidecar_was_merged() is False
    assert recorder.get_translation_record('["t"]') is None


def test_legacy_sidecar_v1_reason_detects_version_and_dotted_keys():
    assert legacy_sidecar_v1_reason({"schema_version": 1, "leaves": {}}) is not None
    assert legacy_sidecar_v1_reason({
        "schema_version": 2,
        "leaves": {"ui.title": {OPERATION_TRANSLATION: make_ok("a", "b").to_dict()}},
    }) is not None
    assert legacy_sidecar_v1_reason({
        "schema_version": 2,
        "leaves": {'["ui","title"]': {OPERATION_TRANSLATION: make_ok("a", "b").to_dict()}},
    }) is None


def test_make_verify_skipped_accepts_translation_missing_reason():
    record = make_verify_skipped("x", reason=LEAF_SKIP_REASON_TRANSLATION_MISSING)
    assert record.status == LEAF_STATUS_VERIFY_SKIPPED
    assert record.reason == LEAF_SKIP_REASON_TRANSLATION_MISSING


def test_make_source_stale_is_degraded_with_current_original_hashes():
    record = make_source_stale("Close {1}.")
    assert record.status == LEAF_STATUS_DEGRADED
    assert record.degraded is True
    assert record.error_kind == LEAF_ERROR_KIND_SOURCE_STALE
    assert record.source_hash == hash_text("Close {1}.")
    assert record.output_hash == hash_text("Close {1}.")


def test_portable_file_identity_maps_native_separators():
    assert portable_file_identity(r"input\route-separate.json") == (
        "input/route-separate.json"
    )
    assert portable_file_identity("input/route-separate.json") == (
        "input/route-separate.json"
    )


def test_nested_file_identity_agrees_between_report_and_sidecar():
    """Report keys and sidecar relative_path share the portable identity (#33)."""
    nested = "input/route-separate.json"
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file(nested)
    recorder.record(
        '["title"]', OPERATION_TRANSLATION, make_ok("Hello", "Привет"),
    )

    sidecar = recorder.file_sidecar_dict(nested)
    report = recorder.build_run_report()

    assert sidecar["schema_version"] == LEAF_REPORT_SCHEMA_VERSION
    assert sidecar["relative_path"] == nested
    assert nested in report["files"]
    assert report["files"][nested] == sidecar["leaves"]
    assert "\\" not in sidecar["relative_path"]
    assert all("\\" not in key for key in report["files"])


def test_native_separator_active_file_normalizes_for_lookup_and_sidecar():
    """Legacy Windows identities still merge and look up under the portable key."""
    native = r"menu\main.json"
    portable = "menu/main.json"
    recorder = LeafOutcomeRecorder()
    recorder.set_active_file(native)
    recorder.record(
        '["label"]', OPERATION_TRANSLATION, make_degraded("Hi", "parse"),
    )

    assert recorder.active_file() == portable
    sidecar = recorder.file_sidecar_dict(native)
    assert sidecar["relative_path"] == portable

    loaded = LeafOutcomeRecorder()
    # Sidecar payload may still advertise the legacy relative_path field.
    loaded.merge_sidecar(portable, {
        **sidecar,
        "relative_path": native,
    })
    loaded.set_active_file(portable)
    assert loaded.sidecar_was_merged() is True
    assert loaded.is_translation_degraded('["label"]') is True

    report = loaded.build_run_report()
    assert list(report["files"]) == [portable]
    assert portable in report["files"]
    assert native not in report["files"]

