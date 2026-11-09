# -*- coding: utf-8 -*-
"""
Integration tests for the recursive engine (llm_conductor/l10n/processor.py, doc/pipeline.md)
on a real Config + CacheManager. Instead of the API — a FakeModel stub that
records every call, so the test can see what would have gone to the model and
how many times.
"""
import copy
import json
import os

import pytest

from llm_conductor.constants import (
    FAILURE_KIND_PARSE,
    LEAF_STATUS_DEGRADED,
    LEAF_STATUS_VERIFY_SKIPPED,
    MODE_STYLE,
    MODE_VERIFICATION,
    OPERATION_TRANSLATION,
    OPERATION_VERIFICATION,
    ORIGINAL_TEXT_BLOCK_NOTE,
    ORIGINAL_TEXT_BLOCK_TEMPLATE,
    REJECTED_REPLY_LOG_LENGTH,
)
from llm_conductor.core import shutdown
from llm_conductor.core.api_client import CallOutcome
from llm_conductor.core.cache import build_sampling_identity
from llm_conductor.core.leaf_report import make_degraded, make_ok
from llm_conductor.l10n.context import KeyPath, LinguisticContext
from llm_conductor.l10n.processor import RecursiveProcessor, process_stage_recursive

SAMPLE = {
    "meta": {"version": 3, "enabled": True, "ratio": 0.75, "owner": None},
    "ui": {
        "title": "Welcome to the Guild",
        "empty": "",
        "whitespace": "   ",
        "gold_counter": "You have {0} gold coins.",
        "warning": "<b>Warning:</b> this action cannot be undone.",
    },
    "dialogues": [
        {"greeting_male": "Welcome back.", "greeting_female": "Welcome back.", "farewell": "Safe travels!"},
        {"farewell": "Safe travels!"},
    ],
    "settings": {"options": ["Easy", 1, False]},
}


class FakeModel:
    """
    Stub for translate_key/verify_key with the process_func signature.

    Returns ``RU(<text>)`` — ``[Tn]`` markers stay in place, as a well-behaved
    model would — wrapped in CallOutcome like the real translate_key/verify_key.
    ``valid=False`` simulates a rejection (validator/model refused the reply):
    success=False and text = the original (still-masked) value, as on a real
    network/parse failure.
    """

    def __init__(self, verification=False, valid=True, fail_on=None):
        self.calls = []
        self.verification = verification
        self.valid = valid
        self.fail_on = fail_on

    def __call__(self, value, config, runtime_config, glossary_string="", mode=None):
        self.calls.append({'value': value, 'prompt': runtime_config['prompt'], 'mode': mode})
        if self.fail_on is not None and self.fail_on in value:
            raise RuntimeError("model exploded")
        if not self.valid:
            return CallOutcome(value, False, FAILURE_KIND_PARSE)
        return CallOutcome(f"RU({value})", True, None)

    @property
    def values(self):
        return [call['value'] for call in self.calls]


class BadReplyModel:
    """
    process_func stub for a call the transport accepts but the content
    validator rejects - unlike FakeModel(valid=False), which simulates a
    transport/parsing failure (CallOutcome.success=False).
    """

    def __init__(self, reply: str):
        self.reply = reply

    def __call__(self, value, config, runtime_config, glossary_string="", mode=None):
        return CallOutcome(self.reply, True, None)


def _run(config, data, model, mode=MODE_STYLE, glossary=None, original=None, verification=False):
    processor = RecursiveProcessor(config, LinguisticContext(config))
    runtime = config.get_verification_config() if verification else config.get_translation_config_for_mode(mode)
    result = processor.traverse(
        data=data,
        current_path=KeyPath(()),
        process_func=model,
        glossary=glossary or {},
        runtime_config=runtime,
        mode=MODE_VERIFICATION if verification else mode,
        original_data=original,
        is_verification=verification,
    )
    return result, processor


# ============================================================================
# Tree traversal
# ============================================================================

def test_structure_order_and_non_strings_preserved_input_not_mutated(make_config):
    config = make_config()
    original = copy.deepcopy(SAMPLE)

    result, _ = _run(config, SAMPLE, FakeModel())

    assert SAMPLE == original                                           # input not mutated
    assert list(result) == list(SAMPLE)                                 # key order
    assert list(result["ui"]) == list(SAMPLE["ui"])
    assert result["meta"] == SAMPLE["meta"]                             # non-strings as-is
    assert result["settings"]["options"] == ["RU(Easy)", 1, False]
    assert (result["ui"]["empty"], result["ui"]["whitespace"]) == ("", "   ")
    assert result["ui"]["title"] == "RU(Welcome to the Guild)"


def test_every_string_leaf_comes_back_as_str(make_config):
    result, _ = _run(make_config(), SAMPLE, FakeModel())

    def leaves(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from leaves(v)
        elif isinstance(node, list):
            for v in node:
                yield from leaves(v)
        else:
            yield node

    assert all(not isinstance(leaf, tuple) for leaf in leaves(result))


def test_tags_are_masked_for_model_and_restored_in_result(make_config):
    model = FakeModel()

    result, _ = _run(make_config(), {"w": "<b>Warning:</b> {0}\nok"}, model)

    assert model.values == ["[T0]Warning:[T1] [T2][T3]ok"]
    assert result["w"] == "RU(<b>Warning:</b> {0}\nok)"


def test_format_only_leaves_skip_model_and_preserve_value(make_config):
    model = FakeModel()
    data = {
        "status": " ({0})",
        "placeholder": "{0}",
        "empty_tags": "<b></b>",
    }

    result, processor = _run(
        make_config(dedup__enabled=False, cache__enabled=False), data, model
    )

    assert result == data
    assert model.calls == []
    assert processor.get_stats_summary()['skipped_leaves'] == 3


def test_boundary_whitespace_peeled_for_model_and_restored(make_config):
    model = FakeModel()

    result, _ = _run(
        make_config(dedup__enabled=False, cache__enabled=False),
        {"label": "  Open.\t"},
        model,
    )

    assert model.values == ["Open."]
    assert result["label"] == "  RU(Open.)\t"


def test_mixed_text_with_placeholder_still_calls_model(make_config):
    model = FakeModel()

    result, _ = _run(
        make_config(dedup__enabled=False, cache__enabled=False),
        {"msg": "Hello {0}"},
        model,
    )

    assert model.values == ["Hello [T0]"]
    assert result["msg"] == "RU(Hello {0})"


def test_statistics(make_config):
    _, processor = _run(make_config(dedup__enabled=False, cache__enabled=False),
                        SAMPLE, FakeModel())

    stats = processor.get_stats_summary()
    assert stats['leaf_nodes'] == 10
    assert stats['skipped_leaves'] == 2
    assert stats['processed_leaves'] == 8
    assert stats['failed_leaves'] == 0
    assert stats['max_depth_reached'] == 3


def test_depth_limit_returns_subtree_unchanged(make_config):
    config = make_config(limits__max_recursion_depth=2)
    data = {"a": {"b": {"c": "too deep"}}, "top": "shallow"}

    result, _ = _run(config, data, FakeModel())

    assert result == {"a": {"b": {"c": "too deep"}}, "top": "RU(shallow)"}
    assert config.logger.contains("Maximum recursion depth (2) exceeded", level='ERROR')


def test_shutdown_request_stops_processing(make_config):
    shutdown._shutdown_manager.request_shutdown(force=True)
    model = FakeModel()

    result, _ = _run(make_config(), SAMPLE, model)

    assert result == SAMPLE
    assert model.calls == []


def test_tripped_circuit_breaker_stops_processing(make_config):
    """traverse() returns the tree unchanged once the breaker is tripped — same as shutdown."""
    config = make_config(resilience__max_consecutive_failures=1)
    assert config.circuit_breaker.record_failure(OPERATION_TRANSLATION) is True
    model = FakeModel()

    result, _ = _run(config, SAMPLE, model)

    assert result == SAMPLE
    assert model.calls == []


def test_model_exception_degrades_leaf_to_original(make_config):
    config = make_config(resilience__max_consecutive_failures=1)

    result, processor = _run(config, {"ok": "fine", "bad": "explode me"}, FakeModel(fail_on="explode"))

    assert result == {"ok": "RU(fine)", "bad": "explode me"}
    assert processor.get_stats_summary()['failed_leaves'] == 1
    assert config.logger.contains("Error processing leaf at bad", level='ERROR')
    assert config.circuit_breaker.is_tripped(OPERATION_TRANSLATION) is True


# ============================================================================
# Context hints and glossary reach the prompt
# ============================================================================

def test_linguistic_hint_and_glossary_reach_the_prompt(make_config):
    model = FakeModel()

    _run(make_config(), {"greeting_female": "Welcome to the Guild"}, model,
         glossary={"Guild": "Гильдия"})

    prompt = model.calls[0]['prompt']
    assert "[LINGUISTIC CONTEXT]: Use FEMININE" in prompt
    assert "Guild = Гильдия" in prompt


def test_verification_glossary_is_picked_by_original_text(make_config):
    """§4.4: on verification, terms are matched against the original, not the translation."""
    model = FakeModel(verification=True)

    _run(make_config(), {"t": "Добро пожаловать в Гильдию"}, model, glossary={"Guild": "Гильдия"},
         original={"t": "Welcome to the Guild"}, verification=True)

    assert "Guild = Гильдия" in model.calls[0]['prompt']


# ============================================================================
# Original text in the verification prompt
# ============================================================================

def test_verification_prompt_contains_masked_original(make_config):
    model = FakeModel(verification=True)

    _run(make_config(), {"t": "Добро пожаловать"}, model,
         original={"t": "Welcome, <b>traveler</b>!"}, verification=True)

    prompt = model.calls[0]['prompt']
    assert ORIGINAL_TEXT_BLOCK_TEMPLATE.format(
        masked_original="Welcome, [T0]traveler[T1]!"
    ) in prompt
    assert ORIGINAL_TEXT_BLOCK_NOTE in prompt


def test_original_block_note_omitted_when_original_has_no_markers(make_config):
    """The numbering note is only for originals that carry markers."""
    model = FakeModel(verification=True)

    _run(make_config(), {"t": "Привет"}, model,
         original={"t": "Hello"}, verification=True)

    prompt = model.calls[0]['prompt']
    assert ORIGINAL_TEXT_BLOCK_TEMPLATE.format(masked_original="Hello") in prompt
    assert ORIGINAL_TEXT_BLOCK_NOTE not in prompt


def test_translation_prompt_never_gets_an_original_block(make_config):
    model = FakeModel(verification=False)

    _run(make_config(), {"t": "Hello"}, model, original={"t": "Hello"}, verification=False)

    assert "[ORIGINAL]:" not in model.calls[0]['prompt']


def test_verify_only_without_original_data_has_no_original_block(make_config):
    """original_data may be unavailable (e.g. a node missing from the source
    tree); the block must then be silently absent, not a crash."""
    model = FakeModel(verification=True)

    _run(make_config(), {"t": "Привет"}, model, original=None, verification=True)

    assert "[ORIGINAL]:" not in model.calls[0]['prompt']


def test_different_originals_for_the_same_translation_do_not_share_a_cache_entry(make_config):
    """The original-text block is part of the enhanced prompt before the
    cache key is built (full-request key), so two different
    English sources for an identical translated string must not collapse
    onto the same cache entry."""
    config = make_config()
    model = FakeModel(verification=True)

    _run(config, {"t": "Привет"}, model, original={"t": "Hello"}, verification=True)
    _run(config, {"t": "Привет"}, model, original={"t": "Hi there"}, verification=True)

    assert len(model.calls) == 2
    assert config.cache_manager.get_stats(OPERATION_VERIFICATION)['cache_size'] == 2


# ============================================================================
# Cache and dedup on the full key
# ============================================================================

def test_same_text_with_different_gender_hints_is_translated_separately(make_config):
    """Regression: greeting_female used to reuse greeting_male's translation."""
    model = FakeModel()

    _run(make_config(), {"greeting_male": "Welcome back.", "greeting_female": "Welcome back."}, model)

    assert len(model.calls) == 2
    assert "MASCULINE" in model.calls[0]['prompt']
    assert "FEMININE" in model.calls[1]['prompt']


def test_identical_leaves_with_identical_context_are_deduplicated(make_config):
    model = FakeModel()

    result, processor = _run(make_config(cache__enabled=False), SAMPLE, model)

    assert model.values.count("Safe travels!") == 1
    assert result["dialogues"][1]["farewell"] == "RU(Safe travels!)"
    assert processor.get_stats_summary()['deduplicated_leaves'] == 1


def test_dedup_disabled_calls_model_for_each_duplicate(make_config):
    model = FakeModel()

    _run(make_config(cache__enabled=False, dedup__enabled=False), SAMPLE, model)

    assert model.values.count("Safe travels!") == 2


def test_second_run_is_served_from_cache(make_config):
    config = make_config()
    first_model, second_model = FakeModel(), FakeModel()
    first, _ = _run(config, SAMPLE, first_model)

    second, _ = _run(config, SAMPLE, second_model)

    assert second == first
    assert second_model.calls == []


def test_cache_survives_save_and_reload(make_config):
    first_config = make_config()
    first, _ = _run(first_config, SAMPLE, FakeModel())
    first_config.cache_manager.save(OPERATION_TRANSLATION)

    second_config = make_config()
    second_config.cache_manager.load(OPERATION_TRANSLATION)
    model = FakeModel()
    second, _ = _run(second_config, SAMPLE, model)

    assert second == first
    assert model.calls == []


@pytest.mark.parametrize("change", [
    {"translation__api__model": "another-model"},
    {"translation__prompts__system": "Be extremely formal."},
    {"translation__prompts__style": "Render in pirate speak: "},
])
def test_changing_model_or_prompt_invalidates_cache(make_config, change):
    first_config = make_config()
    _run(first_config, {"t": "Hello"}, FakeModel())
    first_config.cache_manager.save(OPERATION_TRANSLATION)

    changed = make_config(**change)
    changed.cache_manager.load(OPERATION_TRANSLATION)
    model = FakeModel()
    _run(changed, {"t": "Hello"}, model)

    assert model.values == ["Hello"]


def test_strict_and_style_have_separate_cache_entries(make_config):
    config = make_config(translation__prompts__strict="Translate the text: ",
                         translation__prompts__style="Translate the text: ")
    _run(config, {"t": "Hello"}, FakeModel(), mode="strict")
    model = FakeModel()

    _run(config, {"t": "Hello"}, model, mode="style")

    assert model.values == ["Hello"]


def test_cached_value_is_unmasked_original_text(make_config):
    """§4.4 invariant: the cache stores unmasked results."""
    config = make_config()
    _run(config, {"t": "Hi {0}"}, FakeModel())
    config.cache_manager.save(OPERATION_TRANSLATION)

    with open(config.cache_manager.get_cache_filename(OPERATION_TRANSLATION), encoding='utf-8') as f:
        (entry,) = json.load(f).values()
    assert entry['value'] == "RU(Hi {0})"
    assert entry['path'] == "t"


# ============================================================================
# Cache sanitization
# ============================================================================

def _key_for_leaf(config, value, path=("t",), mode=MODE_STYLE):
    """Replay the key computation from _process_leaf for a leaf with no hints/glossary."""
    processor = RecursiveProcessor(config, LinguisticContext(config))
    runtime = config.get_translation_config_for_mode(mode)
    prompt = processor._build_enhanced_prompt(runtime['prompt'], [], "", OPERATION_TRANSLATION)
    return config.cache_manager.create_cache_key(
        OPERATION_TRANSLATION, runtime['model'], runtime['system_prompt'], prompt, mode, value,
        build_sampling_identity(runtime),
    )


def test_legacy_verification_tuple_in_cache_is_unwrapped(make_config):
    config = make_config()
    config.cache_manager.set(OPERATION_TRANSLATION, _key_for_leaf(config, "Hello"), [True, "Привет"])
    model = FakeModel()

    result, _ = _run(config, {"t": "Hello"}, model)

    assert result == {"t": "Привет"}
    assert model.calls == []


def test_non_string_in_cache_is_deleted_and_reprocessed(make_config):
    config = make_config()
    key = _key_for_leaf(config, "Hello")
    config.cache_manager.set(OPERATION_TRANSLATION, key, {"corrupted": True})
    model = FakeModel()

    result, _ = _run(config, {"t": "Hello"}, model)

    assert result == {"t": "RU(Hello)"}
    assert model.values == ["Hello"]
    assert config.cache_manager.get(OPERATION_TRANSLATION, key) == "RU(Hello)"
    assert config.logger.contains("Cache returned non-string value", level='ERROR')


# ============================================================================
# Verification
# ============================================================================

def test_verification_succeeds_and_uses_own_cache(make_config):
    config = make_config()
    model = FakeModel(verification=True)

    result, _ = _run(config, {"t": "Привет"}, model, original={"t": "Hello"}, verification=True)

    assert result == {"t": "RU(Привет)"}
    assert config.cache_manager.get_stats(OPERATION_VERIFICATION)['cache_size'] == 1
    assert config.cache_manager.get_stats(OPERATION_TRANSLATION)['cache_size'] == 0


def test_rejected_verification_keeps_translation_and_is_not_cached(make_config):
    config = make_config()

    result, processor = _run(config, {"t": "Привет"}, FakeModel(verification=True, valid=False),
                             original={"t": "Hello"}, verification=True)

    assert result == {"t": "Привет"}
    assert processor.get_stats_summary()['failed_leaves'] == 1
    assert config.cache_manager.get_stats(OPERATION_VERIFICATION)['cache_size'] == 0


def test_rejected_translation_reply_is_not_cached(make_config):
    """Content-validator rejects on translation must not write the translation cache (§4.4а)."""
    config = make_config()
    bad_reply = "RU(Hello)\nunexpected chatter"

    result, processor = _run(config, {"t": "Hello"}, BadReplyModel(bad_reply))

    assert result == {"t": "Hello"}
    assert processor.get_stats_summary()['failed_leaves'] == 1
    assert config.cache_manager.get_stats(OPERATION_TRANSLATION)['cache_size'] == 0
    assert config.leaf_outcomes.has_untreated_error() is True


def test_verification_skips_translation_degraded_without_model_call(make_config):
    """Normal verify must not run on translation_degraded (outcome-based, not ==)."""
    config = make_config()
    config.leaf_outcomes.set_active_file("sample.json")
    config.leaf_outcomes.record(
        "t", OPERATION_TRANSLATION, make_degraded("Hello", FAILURE_KIND_PARSE)
    )
    model = FakeModel(verification=True)

    result, _ = _run(
        config, {"t": "Hello"}, model, original={"t": "Hello"}, verification=True
    )

    assert result == {"t": "Hello"}
    assert model.calls == []
    sidecar = config.leaf_outcomes.file_sidecar_dict("sample.json")
    assert sidecar["leaves"]["t"][OPERATION_VERIFICATION]["status"] == LEAF_STATUS_VERIFY_SKIPPED


def test_identical_translation_still_verified_when_not_degraded(make_config):
    """Names/ids that legally equal the source still get normal verification."""
    config = make_config()
    config.leaf_outcomes.set_active_file("sample.json")
    config.leaf_outcomes.record(
        "name", OPERATION_TRANSLATION, make_ok("Necron", "Necron")
    )
    model = FakeModel(verification=True)

    result, _ = _run(
        config, {"name": "Necron"}, model,
        original={"name": "Necron"}, verification=True,
    )

    assert model.values == ["Necron"]
    assert result == {"name": "RU(Necron)"}
    assert config.leaf_outcomes.is_translation_degraded("name") is False


def test_rejected_translation_records_degraded_status(make_config):
    config = make_config()
    config.leaf_outcomes.set_active_file("sample.json")

    _run(config, {"t": "Hello"}, FakeModel(valid=False))

    sidecar = config.leaf_outcomes.file_sidecar_dict("sample.json")
    assert sidecar["leaves"]["t"][OPERATION_TRANSLATION]["status"] == LEAF_STATUS_DEGRADED
    assert sidecar["leaves"]["t"][OPERATION_TRANSLATION]["error_kind"] == FAILURE_KIND_PARSE


def test_rejected_reply_is_logged_with_a_preview(make_config):
    """A reply the content validator rejects is still logged, not just its rejection reason."""
    config = make_config()
    bad_reply = "RU(Hello)\nunexpected chatter"  # a literal \n always rejects (newline_in_reply)

    result, _ = _run(config, {"t": "Hello"}, BadReplyModel(bad_reply))

    assert result == {"t": "Hello"}
    assert config.logger.contains("Rejected model reply for t: newline_in_reply", level='WARNING')
    assert config.logger.contains(repr(bad_reply), level='WARNING')


def test_rejected_reply_log_is_truncated_for_long_replies(make_config):
    config = make_config()
    bad_reply = ("x" * (REJECTED_REPLY_LOG_LENGTH + 50)) + "\n"

    _run(config, {"t": "Hello"}, BadReplyModel(bad_reply))

    logged = next(m for m in config.logger.messages('WARNING') if "Rejected model reply" in m)
    assert repr(bad_reply) not in logged
    assert repr(bad_reply[:REJECTED_REPLY_LOG_LENGTH]) in logged
    assert logged.rstrip().endswith('...')


def test_tag_marker_mismatch_log_includes_missing_detail(make_config):
    config = make_config()
    # Source masks {0} -> [T0]; reply drops the marker.
    bad_reply = "Нажмите для прыжка"

    result, _ = _run(config, {"t": "Press {0} to jump"}, BadReplyModel(bad_reply))

    assert result == {"t": "Press {0} to jump"}
    logged = next(m for m in config.logger.messages('WARNING') if "Rejected model reply" in m)
    assert "tag_marker_mismatch[missing=[T0]]" in logged


def test_save_rejected_replies_writes_full_body_under_log_dir(make_config, tmp_path):
    config = make_config(validation__save_rejected_replies=True)
    bad_reply = "full rejected body that must be persisted\n"

    _run(config, {"t": "Hello"}, BadReplyModel(bad_reply))

    saved = tmp_path / "logs" / "rejected" / OPERATION_TRANSLATION / "t.txt"
    assert saved.is_file()
    assert saved.read_text(encoding='utf-8') == bad_reply


def test_save_rejected_replies_off_writes_no_file(make_config, tmp_path):
    config = make_config(validation__save_rejected_replies=False)
    bad_reply = "RU(Hello)\nunexpected chatter"

    _run(config, {"t": "Hello"}, BadReplyModel(bad_reply))

    rejected_root = tmp_path / "logs" / "rejected"
    assert not rejected_root.exists()


# ============================================================================
# Autosave and process_stage_recursive
# ============================================================================

def test_auto_save_flushes_cache_every_n_new_entries(make_config):
    config = make_config(cache__auto_save_interval=2)
    cache_file = config.cache_manager.get_cache_filename(OPERATION_TRANSLATION)
    flushed_sizes = []
    original_save = config.cache_manager.save

    def spy_save(cache_type):
        result = original_save(cache_type)
        with open(cache_file, encoding='utf-8') as f:
            flushed_sizes.append(len(json.load(f)))
        return result

    config.cache_manager.save = spy_save

    _run(config, {"a": "One", "b": "Two", "c": "Three", "d": "Four", "e": "Five"}, FakeModel())

    assert flushed_sizes == [2, 4]       # every 2 new entries; remainder waits for final save()


@pytest.mark.characterization(
    reason="stage remainder < interval new entries waits for final save() in main() — lost on kill")
def test_auto_save_counter_is_per_stage_processor(make_config):
    """The counter lives on the RecursiveProcessor instance and does not accumulate across stages."""
    config = make_config(cache__auto_save_interval=2)

    _run(config, {"a": "One"}, FakeModel())
    _run(config, {"b": "Two"}, FakeModel())

    assert not os.path.exists(config.cache_manager.get_cache_filename(OPERATION_TRANSLATION))


def test_auto_save_ignores_cache_hits(make_config):
    """The counter counts only NEW entries — a cache hit does not touch the disk."""
    config = make_config(cache__auto_save_interval=2)
    _run(config, {"a": "One"}, FakeModel())

    _run(config, {"a": "One"}, FakeModel())     # hit
    assert not os.path.exists(config.cache_manager.get_cache_filename(OPERATION_TRANSLATION))


def test_process_stage_recursive_writes_stage_file_and_reports(make_config, tmp_path):
    config = make_config()
    output_path = str(tmp_path / "translated.json")
    runtime = config.get_translation_config_for_mode(MODE_STYLE)
    runtime['mode'] = MODE_STYLE

    result = process_stage_recursive(output_path, "translated", SAMPLE, SAMPLE, FakeModel(), False,
                                     config, {}, OPERATION_TRANSLATION, runtime)

    with open(output_path, encoding='utf-8') as f:
        assert json.load(f) == result
    assert config.logger.contains("Translated stage completed in", level='INFO')
    assert config.logger.contains("Translation cache:", level='INFO')
