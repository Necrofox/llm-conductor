# -*- coding: utf-8 -*-
"""Unit tests for rejected-reply dump naming and attribution helpers."""
from llm_conductor.constants import (
    OPERATION_TRANSLATION,
    REJECTED_REPLY_LEAF_DIGEST_LENGTH,
    REJECTED_REPLY_NO_FILE_LABEL,
)
from llm_conductor.core.leaf_report import hash_text
from llm_conductor.l10n.context import KeyPath
from llm_conductor.l10n.processor import RecursiveProcessor


def test_sanitize_rejected_reply_label_replaces_unsafe_and_empty():
    assert RecursiveProcessor._sanitize_rejected_reply_label("") == REJECTED_REPLY_NO_FILE_LABEL
    assert RecursiveProcessor._sanitize_rejected_reply_label("a/b:c") == "a_b_c"
    assert RecursiveProcessor._sanitize_rejected_reply_label("ui[0].title") == "ui_0_.title"


def test_build_rejected_reply_stem_separates_sanitize_collisions():
    """Paths that sanitize to the same display label keep distinct digests."""
    slash = KeyPath(("a/b",))
    underscore = KeyPath(("a_b",))
    dotted = KeyPath(("a.b",))
    nested = KeyPath(("a", "b"))
    shared = {"relative_file": "one.json", "attempt_id": 1, "unique_suffix": "42"}

    stems = {
        RecursiveProcessor.build_rejected_reply_stem(leaf=slash, **shared),
        RecursiveProcessor.build_rejected_reply_stem(leaf=underscore, **shared),
        RecursiveProcessor.build_rejected_reply_stem(leaf=dotted, **shared),
        RecursiveProcessor.build_rejected_reply_stem(leaf=nested, **shared),
    }
    assert len(stems) == 4
    for leaf in (slash, underscore, dotted, nested):
        digest = hash_text(leaf.report_key())[:REJECTED_REPLY_LEAF_DIGEST_LENGTH]
        assert any(digest in stem for stem in stems)


def test_build_rejected_reply_stem_separates_files_and_attempts():
    leaf = KeyPath(("same",))
    one = RecursiveProcessor.build_rejected_reply_stem(
        relative_file="one.json", leaf=leaf, attempt_id=1, unique_suffix="9",
    )
    two = RecursiveProcessor.build_rejected_reply_stem(
        relative_file="two.json", leaf=leaf, attempt_id=1, unique_suffix="9",
    )
    retry = RecursiveProcessor.build_rejected_reply_stem(
        relative_file="one.json", leaf=leaf, attempt_id=2, unique_suffix="9",
    )
    later = RecursiveProcessor.build_rejected_reply_stem(
        relative_file="one.json", leaf=leaf, attempt_id=1, unique_suffix="10",
    )
    assert len({one, two, retry, later}) == 4


def test_build_rejected_reply_meta_links_attempt_and_report_key():
    leaf = KeyPath(("a.b",))
    meta = RecursiveProcessor.build_rejected_reply_meta(
        relative_file="pack/ui.json",
        leaf=leaf,
        operation_type=OPERATION_TRANSLATION,
        attempt_id=7,
        dump_name="dump.txt",
    )
    assert meta == {
        "file": "pack/ui.json",
        "leaf": leaf.report_key(),
        "leaf_display": "a.b",
        "operation": OPERATION_TRANSLATION,
        "attempt_id": 7,
        "dump": "dump.txt",
    }
    assert meta["leaf"] != KeyPath(("a", "b")).report_key()
