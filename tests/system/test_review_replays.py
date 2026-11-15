# -*- coding: utf-8 -*-
"""Replay reduced v1.1.1 pilot cases through the CLI without live inference."""
import json
from pathlib import Path

import pytest
import responses

from llm_conductor.constants import (
    EXIT_FAILURE_RATIO_EXCEEDED,
    EXIT_SUCCESS,
    REJECTED_REPLY_META_SUFFIX,
)
from llm_conductor.l10n.context import KeyPath
from tests.support import OLLAMA_URL, dump_toml, ollama_reply
from tests.system.harness import VERIFY_PROMPT, base_docs


@pytest.fixture
def http():
    """Intercept every scripted model request; unregistered requests fail locally."""
    with responses.RequestsMock(assert_all_requests_are_fired=True) as mock:
        yield mock


@pytest.mark.parametrize("source,reply,missing", [
    pytest.param("Tell {0} that {10} can leave.", "Скажи [T0], что они могут уйти.",
                 "[T1]", id="second-placeholder-replaced-by-pronoun"),
    pytest.param("The <b>UNA</b> is here.", "ОСА здесь.",
                 "[T0], [T1]", id="acronym-tags-dropped"),
    pytest.param("Ask <b>Macha</b> to call {1}.", "Попроси Macha позвонить.",
                 "[T0], [T1], [T2]", id="name-tags-and-placeholder-dropped"),
])
def test_marker_loss_falls_back_skips_verify_and_is_not_cached(
        workspace, http, run_main, source, reply, missing):
    """A valid HTTP reply can fail content validation without reaching the editor."""
    workspace.write_configs(
        preflight__check_availability=False,
        validation__allowed_scripts=["cyrillic"],
        resilience__fail_on_any_error=True,
        resilience__max_failure_ratio=1.0,
        resilience__max_consecutive_failures=0,
    )
    workspace.write_json("sample.json", {"line": source})
    http.add(responses.POST, OLLAMA_URL, json=ollama_reply(reply))
    command = ("run", "--target", "sample.json")
    leaf = KeyPath(("line",)).report_key()

    assert run_main(*command, "--stage", "translate") == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(http.calls) == 1
    assert workspace.read_json(workspace.result("final")) == {"line": source}
    sidecar = workspace.read_json(workspace.result("leaves"))
    translation = sidecar['leaves'][leaf]['translation']
    assert translation['status'] == 'degraded'
    assert translation['error_kind'] == 'tag_marker_mismatch'
    assert f"tag_marker_mismatch[missing={missing}]" in workspace.log_text()
    usage = workspace.read_json("result/sample.json/usage.json")
    assert len(usage['attempts']) == 1
    attempt = usage['attempts'][0]
    assert attempt['outcome'] == 'rejected'
    assert attempt['failure_kind'] == 'tag_marker_mismatch'
    assert attempt['prompt_tokens'] == 7
    assert attempt['completion_tokens'] == 3

    assert run_main(*command, "--stage", "verify") == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(http.calls) == 1
    assert workspace.read_json(workspace.result("final")) == {"line": source}
    sidecar = workspace.read_json(workspace.result("leaves"))
    verification = sidecar['leaves'][leaf]['verification']
    assert verification['status'] == 'verify_skipped'
    assert verification['reason'] == 'translation_degraded'

    assert run_main(*command, "--stage", "translate") == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(http.calls) == 2
    assert workspace.read_json(workspace.result("final")) == {"line": source}


def test_draft_manifest_cache_uses_original_and_profile(workspace, http, run_main):
    """Matching draft/source pairs share cache; a different source or model does not."""
    workspace.write_configs(preflight__check_availability=False,
                            validation__allowed_scripts=["latin"])
    other, _ = base_docs(preflight__check_availability=False,
                         validation__allowed_scripts=["latin"],
                         verification__api__model="other-editor")
    workspace.write_text("profiles/other.toml", dump_toml(other))
    jobs = []
    originals = ["Request A", "Request A", "Request B", "Request A"]
    for index, original in enumerate(originals):
        workspace.write_json(f"input/{index}.json", {"label": original})
        workspace.write_json(f"draft/{index}.json", {"label": "LOW"})
        job = f'[[targets]]\noriginal = "input/{index}.json"\ndraft = "draft/{index}.json"\n'
        if index == 3:
            job += 'llm_config = "profiles/other.toml"\n'
        jobs.append(job)
    workspace.write_text("job.toml", "\n".join(jobs))
    for reply in ("LOW", "HIGH", "MEDIUM"):
        http.add(responses.POST, OLLAMA_URL, json=ollama_reply(reply))

    assert run_main("run", "--target", "job.toml") == EXIT_SUCCESS

    payloads = [json.loads(call.request.body) for call in http.calls]
    assert len(payloads) == 3
    assert [p['model'] for p in payloads] == ['test-model', 'test-model', 'other-editor']
    assert all(VERIFY_PROMPT in p['messages'][-1]['content'] for p in payloads)
    assert 'Request A' in payloads[0]['messages'][-1]['content']
    assert 'Request B' in payloads[1]['messages'][-1]['content']
    for index, expected in enumerate(("LOW", "LOW", "HIGH", "MEDIUM")):
        assert workspace.read_json(workspace.result("final", f"input/{index}.json", "job.toml")) == {
            "label": expected,
        }
        assert workspace.read_json(f"draft/{index}.json") == {"label": "LOW"}
        assert workspace.read_json(f"input/{index}.json") == {"label": originals[index]}
    report = workspace.read_json("result/job.toml/run_report.json")
    files = {Path(name).as_posix(): leaves for name, leaves in report['files'].items()}
    duplicate = files['input/1.json']['["label"]']['verification']
    assert duplicate['status'] == 'cache_hit'
    usage = workspace.read_json("result/job.toml/usage.json")
    assert len(usage['attempts']) == 3
    assert usage['non_api']['verification']['cache_hit'] == 1
    assert not workspace.exists("result/job.toml/translated")


def test_manifest_routes_packed_context_and_redacts_with_separate_profiles(
        workspace, http, run_main):
    """Profiles can process labels and redaction tokens through string leaves."""
    workspace.write_configs(preflight__check_availability=False)
    inputs = {
        "routes": {
            "annex": "ANNEX: The shared lobby lamp is dark.",
            "main": "MAIN: The shared lobby lamp is dark.",
        },
        "contacts": {
            "name": "Contact Nera Quindle about Room 12.",
            "phone": "Call 202.555.0119. Equipment code 555-0141.",
            "public": "Quindle Tower, Room 23: a hinge squeaks.",
        },
    }
    outputs = {
        "routes": {"annex": "ANNEX_LIGHTING", "main": "MAIN_LIGHTING"},
        "contacts": {
            "name": "Contact [NAME] about Room 12.",
            "phone": "Call [PHONE]. Equipment code 555-0141.",
            "public": inputs['contacts']['public'],
        },
    }
    jobs = []
    for name, source in inputs.items():
        profile, _ = base_docs(
            preflight__check_availability=False,
            validation__allowed_scripts=["latin"],
            translation__api__model=f"{name}-model",
            translation__prompts__system=f"Process {name} only.",
        )
        workspace.write_text(f"profiles/{name}.toml", dump_toml(profile))
        workspace.write_json(f"input/{name}.json", source)
        jobs.append(f'[[targets]]\noriginal = "input/{name}.json"\n'
                    f'llm_config = "profiles/{name}.toml"\n')
        for reply in outputs[name].values():
            http.add(responses.POST, OLLAMA_URL, json=ollama_reply(reply))
    workspace.write_text("job.toml", 'stage = "translate"\n' + '\n'.join(jobs))

    assert run_main("run", "--target", "job.toml") == EXIT_SUCCESS

    payloads = [json.loads(call.request.body) for call in http.calls]
    assert len(payloads) == 5
    assert [p['model'] for p in payloads] == ['routes-model'] * 2 + ['contacts-model'] * 3
    assert [p['messages'][0]['content'] for p in payloads] == (
        ['Process routes only.'] * 2 + ['Process contacts only.'] * 3
    )
    assert 'ANNEX: The shared lobby lamp is dark.' in payloads[0]['messages'][-1]['content']
    assert 'MAIN: The shared lobby lamp is dark.' in payloads[1]['messages'][-1]['content']
    for name, expected in outputs.items():
        assert workspace.read_json(workspace.result("final", f"input/{name}.json", "job.toml")) == expected
        assert workspace.read_json(f"input/{name}.json") == inputs[name]
    report = workspace.read_json("result/job.toml/run_report.json")
    files = {Path(name).as_posix(): leaves for name, leaves in report['files'].items()}
    unchanged = files['input/contacts.json']['["public"]']['translation']
    assert unchanged['status'] == 'ok'
    assert unchanged['degraded'] is False


@pytest.fixture(params=[
    pytest.param({"one.json": {"same": "Open {0}."},
                  "two.json": {"same": "Close {1}."}}, id="different-files"),
    pytest.param({"one.json": {"a.b": "Open {0}.", "a": {"b": "Close {1}."}}},
                 id="literal-dot-and-nested-path"),
    pytest.param({"one.json": {"a/b": "Open {0}.", "a_b": "Close {1}."}},
                 id="sanitized-path-collision"),
])
def rejected_pair(request, workspace, http, run_main):
    """Run two distinct rejections whose diagnostic filenames can collide."""
    workspace.write_configs(
        preflight__check_availability=False,
        validation__allowed_scripts=["cyrillic"],
        validation__save_rejected_replies=True,
        resilience__fail_on_any_error=True,
        resilience__max_consecutive_failures=0,
    )
    jobs = []
    for name, source in request.param.items():
        workspace.write_json(name, source)
        jobs.append(f'[[targets]]\noriginal = "{name}"\n')
    workspace.write_text("job.toml", 'stage = "translate"\n' + '\n'.join(jobs))
    for reply in ("Открой.", "Закрой."):
        http.add(responses.POST, OLLAMA_URL, json=ollama_reply(reply))
    exit_code = run_main("run", "--target", "job.toml")
    return exit_code, request.param


def test_colliding_rejections_keep_distinct_leaf_and_usage_records(
        rejected_pair, workspace, http):
    exit_code, inputs = rejected_pair
    assert exit_code == EXIT_FAILURE_RATIO_EXCEEDED
    assert len(http.calls) == 2
    for name, source in inputs.items():
        assert workspace.read_json(workspace.result("final", name, "job.toml")) == source
    usage = workspace.read_json("result/job.toml/usage.json")
    assert len(usage['attempts']) == 2
    assert len({attempt['id'] for attempt in usage['attempts']}) == 2
    assert all(attempt['outcome'] == 'rejected' for attempt in usage['attempts'])
    report = workspace.read_json("result/job.toml/run_report.json")
    records = [stages['translation'] for leaves in report['files'].values()
               for stages in leaves.values()]
    assert len(records) == 2
    assert all(record['status'] == 'degraded' for record in records)


def test_rejected_reply_dumps_preserve_both_bodies(rejected_pair, workspace):
    """Every rejected attempt must retain its own full diagnostic reply."""
    _exit_code, inputs = rejected_pair
    rejected_dir = workspace.root / "logs/rejected/translation"
    dumps = sorted(rejected_dir.glob("*.txt"))
    assert sorted(path.read_text(encoding="utf-8") for path in dumps) == ["Закрой.", "Открой."]
    metas = [
        json.loads((path.with_name(path.stem + REJECTED_REPLY_META_SUFFIX)).read_text(
            encoding="utf-8"))
        for path in dumps
    ]
    assert len(metas) == 2
    assert {meta["file"] for meta in metas} == set(inputs)
    # Same leaf key across files shares report_key; distinct paths do not.
    assert len({(meta["file"], meta["leaf"]) for meta in metas}) == 2
    attempt_ids = {meta["attempt_id"] for meta in metas}
    assert None not in attempt_ids
    assert len(attempt_ids) == 2
    usage = workspace.read_json("result/job.toml/usage.json")
    assert attempt_ids == {attempt["id"] for attempt in usage["attempts"]}
