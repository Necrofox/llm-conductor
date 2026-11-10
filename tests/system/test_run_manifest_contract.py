# -*- coding: utf-8 -*-
"""
System tests for conflict / ambiguous cases of the TOML run-manifest contract
(GH-13 precedence B, GH-14 draft). Happy-path batch runs live in test_pipeline.
"""
import json

import pytest
import responses

from llm_conductor.constants import EXIT_SUCCESS, EXIT_USAGE_ERROR
from tests.support import OLLAMA_URL, ollama_reply, write_configs
from tests.system.harness import TRANSLATE_PROMPT, VERIFY_PROMPT, base_docs

MENU = {"title": "Main menu"}
DRAFT = {"title": "Главное меню"}


@pytest.fixture
def project(workspace):
    workspace.write_json("project/menu/main.json", MENU)
    workspace.write_json("project/dialogs/npc/guard.json", {"line": "Halt!"})
    workspace.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"

[[targets]]
original = "dialogs/npc/guard.json"
""")
    return workspace


# ============================================================================
# Precedence B — job CLI flags with a .toml target
# ============================================================================

@pytest.mark.parametrize("flag_args", [
    ("--stage", "translate"),
    ("--strict",),
    ("--llm-config", "llm.toml"),
    ("--glossary", "glossary.json"),
    ("--result2target",),
    ("--cache2target",),
], ids=["stage", "strict", "llm-config", "glossary", "result2target", "cache2target"])
def test_job_cli_flag_with_manifest_exits_2(project, model, run_main, flag_args, capsys):
    code = run_main("run", "--target", "project/ui.toml", *flag_args)
    assert code == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "not allowed with a .toml" in combined or "Job flags" in combined
    assert model.requests == []


def test_session_flags_verb_and_no_cache_still_work_with_manifest(
        project, model, run_main, preserve_logs):
    assert run_main(
        "run", "--target", "project/ui.toml", "--verb", "DEBUG", "--no-cache",
    ) == EXIT_SUCCESS
    assert "File 2/2" in project.log_text()


def test_legacy_list_is_rejected_with_explicit_migration_message(
        project, model, run_main, capsys):
    project.write_text("project/ui.list", "menu/main.json\n")
    assert run_main("run", "--target", "project/ui.list") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert ".list manifests are no longer supported" in combined
    assert "[[targets]]" in combined
    assert 'original = "path/from/line.json"' in combined
    assert "doc/cli.md" in combined
    assert model.requests == []


def test_non_manifest_toml_exits_2(project, model, run_main, capsys):
    project.write_text("project/conductor-lookalike.toml", """
[logging]
level = "INFO"
""")
    assert run_main("run", "--target", "project/conductor-lookalike.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "not a run manifest" in combined
    assert model.requests == []


def test_job_flag_rejected_even_when_toml_is_not_a_manifest(project, model, run_main, capsys):
    """Precedence B keys off the .toml suffix before schema validation."""
    project.write_text("project/not-manifest.toml", 'verb = "INFO"\n')
    assert run_main(
        "run", "--target", "project/not-manifest.toml", "--stage", "translate",
    ) == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "Job flags" in combined or "not allowed" in combined
    assert model.requests == []


# ============================================================================
# Per-target overrides
# ============================================================================

def test_per_target_stage_override(project, model, run_main):
    """First target translate-only; second verify would need translated/ — use translate for both shapes."""
    project.write_text("project/ui.toml", """
stage = "all"

[[targets]]
original = "menu/main.json"
stage = "translate"

[[targets]]
original = "dialogs/npc/guard.json"
stage = "translate"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS
    assert project.exists(project.result("translated", "menu/main.json", target="ui.toml"))
    # translate-only does not write redacted/
    assert not project.exists(project.result("redacted", "menu/main.json", target="ui.toml"))


# ============================================================================
# Draft (#14) conflicts and verify-from-draft
# ============================================================================

def test_draft_plus_target_stage_exits_2_before_api(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
stage = "verify"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "draft implies verify" in combined
    assert model.requests == []


def test_draft_structure_mismatch_exits_2_before_api(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", {"title": "x", "extra": "nope"})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "extra" in combined or "structure" in combined
    assert model.requests == []


def test_draft_marker_mismatch_exits_2_before_api(project, model, run_main, capsys):
    """D2: draft that already lost placeholders must fail preflight."""
    project.write_json("project/menu/main.json", {"line": "Open {0}."})
    project.write_json("project/drafts/main.ru.json", {"line": "Открой."})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "marker" in combined
    assert model.requests == []


def test_draft_verify_calls_only_verification_and_keeps_inputs(
        project, model, run_main):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--no-cache") == EXIT_SUCCESS

    assert project.read_json("project/menu/main.json") == MENU
    assert project.read_json("project/drafts/main.ru.json") == DRAFT
    assert project.exists(project.result("final", "menu/main.json", target="ui.toml"))
    assert project.exists(project.result("redacted", "menu/main.json", target="ui.toml"))
    assert not project.exists(project.result("translated", "menu/main.json", target="ui.toml"))
    assert model.calls_with(TRANSLATE_PROMPT) == []
    assert model.calls_with(VERIFY_PROMPT)


def test_draft_missing_file_exits_2_before_api(project, model, run_main, capsys):
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/absent.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    assert model.requests == []


def test_draft_plus_stage_with_bad_path_collects_both_errors(
        project, model, run_main, capsys):
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "../outside.ru.json"
stage = "verify"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "target 1:" in combined
    assert "outside the manifest directory" in combined
    assert "draft implies verify" in combined
    assert model.requests == []


def test_run_level_stage_does_not_conflict_with_draft_target(project, model, run_main):
    """Other non-draft targets still use run-level stage; draft ignores it."""
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
stage = "translate"

[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"

[[targets]]
original = "dialogs/npc/guard.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS
    assert not project.exists(project.result("translated", "menu/main.json", target="ui.toml"))
    assert project.exists(project.result("translated", "dialogs/npc/guard.json", target="ui.toml"))


def test_per_target_llm_config_and_glossary_reach_the_api(project, model, run_main):
    """Different originals in one manifest can use different profiles and glossaries."""
    # Source must mention the glossary term or it is not injected into the prompt.
    project.write_json("project/menu/main.json", {"title": "Welcome to the Guild"})
    project.write_json("project/dialogs/npc/guard.json", {"line": "Guild ahead"})
    for name, model_name in (("dialogs", "dlg-model"), ("mechanics", "mech-model")):
        llm_doc, _ = base_docs(
            translation__api__model=model_name,
            verification__api__model=model_name,
            meta__name=f"{name} profile",
        )
        write_configs(project.root / "project" / "profiles" / name, llm_doc, {})
        (project.root / "project" / "profiles" / name / "llm.toml").rename(
            project.root / "project" / "profiles" / f"{name}.toml"
        )
    project.write_json("project/glossaries/dialogs.json", {"Guild": "Гильдия"})
    project.write_json("project/glossaries/mechanics.json", {"Guild": "Орден"})
    project.write_text("project/ui.toml", """
stage = "translate"

[[targets]]
original = "menu/main.json"
llm_config = "profiles/dialogs.toml"
glossary = "glossaries/dialogs.json"

[[targets]]
original = "dialogs/npc/guard.json"
llm_config = "profiles/mechanics.toml"
glossary = "glossaries/mechanics.json"
""")
    assert run_main("run", "--target", "project/ui.toml", "--no-cache") == EXIT_SUCCESS

    dlg = model.calls_with("Guild = Гильдия")
    mech = model.calls_with("Guild = Орден")
    assert dlg and {p["model"] for p in dlg} == {"dlg-model"}
    assert mech and {p["model"] for p in mech} == {"mech-model"}
    assert model.calls_with(VERIFY_PROMPT) == []


def test_dual_editor_profiles_use_two_manifests(project, model, run_main):
    """Same original+draft compared via two manifests (#14); one manifest forbids dup originals."""
    project.write_json("project/drafts/main.ru.json", DRAFT)
    for name, model_name in (("editor-a", "editor-a-model"), ("editor-b", "editor-b-model")):
        llm_doc, _ = base_docs(
            translation__api__model="unused-translate",
            verification__api__model=model_name,
            meta__name=name,
        )
        write_configs(project.root / "project" / "profiles" / name, llm_doc, {})
        (project.root / "project" / "profiles" / name / "llm.toml").rename(
            project.root / "project" / "profiles" / f"{name}.toml"
        )
        project.write_text(f"project/{name}.toml", f"""
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
llm_config = "profiles/{name}.toml"
""")

    assert run_main("run", "--target", "project/editor-a.toml", "--no-cache") == EXIT_SUCCESS
    assert run_main("run", "--target", "project/editor-b.toml", "--no-cache") == EXIT_SUCCESS
    assert model.calls_with(TRANSLATE_PROMPT) == []
    assert {p["model"] for p in model.calls_with(VERIFY_PROMPT)} == {
        "editor-a-model", "editor-b-model",
    }
    assert project.read_json("project/menu/main.json") == MENU
    assert project.read_json("project/drafts/main.ru.json") == DRAFT


def test_manifest_validates_each_stage_with_its_target_profile(project, run_main):
    project.write_configs(validation__allowed_scripts=["cyrillic"])
    llm_doc, _ = base_docs(
        translation__api__model="cjk-model",
        verification__api__model="cjk-model",
        validation__allowed_scripts=["cjk"],
    )
    write_configs(project.root / "project" / "profiles", llm_doc, {})
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"

[[targets]]
original = "dialogs/npc/guard.json"
llm_config = "profiles/llm.toml"
""")

    def reply(request):
        payload = json.loads(request.body)
        content = payload["messages"][-1]["content"]
        if content == "test":
            return 200, {}, json.dumps(ollama_reply("ok"))
        translated, edited = ("你好", "你好世界") if payload["model"] == "cjk-model" else (
            "Привет", "Привет мир"
        )
        text = edited if VERIFY_PROMPT in content else translated
        return 200, {}, json.dumps(ollama_reply(text))

    with responses.RequestsMock() as http:
        http.add_callback(responses.POST, OLLAMA_URL, callback=reply)
        assert run_main("run", "--target", "project/ui.toml") == EXIT_SUCCESS

    for name, key, expected in (
        ("menu/main.json", "title", "Привет мир"),
        ("dialogs/npc/guard.json", "line", "你好世界"),
    ):
        assert project.read_json(project.result("final", name, target="ui.toml")) == {key: expected}


def test_editor_cache_isolated_by_endpoint(project, run_main):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    endpoints = {
        "a": (OLLAMA_URL, "Первый ответ"),
        "b": ("http://second.llm.test/api/chat", "Второй ответ"),
    }

    def reply(text):
        def callback(request):
            content = json.loads(request.body)["messages"][-1]["content"]
            return 200, {}, json.dumps(ollama_reply("ok" if content == "test" else text))
        return callback

    with responses.RequestsMock() as http:
        for name, (url, text) in endpoints.items():
            llm_doc, _ = base_docs(verification__api__url=url,
                                   validation__allowed_scripts=["cyrillic"])
            write_configs(project.root / "project" / "profiles" / name, llm_doc, {})
            project.write_text(f"project/{name}.toml", f"""
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
llm_config = "profiles/{name}/llm.toml"
""")
            http.add_callback(responses.POST, url, callback=reply(text))

        for name in ("a", "b", "b"):
            assert run_main("run", "--target", f"project/{name}.toml") == EXIT_SUCCESS
            final_path = project.result("final", "menu/main.json", target=f"{name}.toml")
            assert project.read_json(final_path) == {"title": endpoints[name][1]}

        model_urls = [
            call.request.url for call in http.calls
            if json.loads(call.request.body)["messages"][-1]["content"] != "test"
        ]
        assert model_urls == [url for url, _ in endpoints.values()]


def test_draft_verify_preserves_translation_sidecar_in_shared_result_root(project, model, run_main):
    assert run_main("run", "--target", "project/menu/main.json",
                    "--stage", "translate", "--result2target") == EXIT_SUCCESS
    sidecar_path = "project/menu/result/leaves/main.json"
    original_sidecar = project.read_json(sidecar_path)

    project.write_text("project/menu/edit.toml", """
result2target = true
[[targets]]
original = "main.json"
draft = "draft.json"
""")
    project.write_json("project/menu/draft.json", DRAFT)
    assert run_main("run", "--target", "project/menu/edit.toml") == EXIT_SUCCESS
    assert model.calls_with(VERIFY_PROMPT)
    assert project.read_json(sidecar_path) == original_sidecar


def test_same_original_twice_in_one_manifest_is_rejected(project, model, run_main, capsys):
    project.write_json("project/drafts/main.ru.json", DRAFT)
    project.write_text("project/ui.toml", """
[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"

[[targets]]
original = "menu/main.json"
draft = "drafts/main.ru.json"
""")
    assert run_main("run", "--target", "project/ui.toml") == EXIT_USAGE_ERROR
    combined = "".join(capsys.readouterr())
    assert "duplicate original" in combined
    assert "result path collides" in combined
    assert model.requests == []
