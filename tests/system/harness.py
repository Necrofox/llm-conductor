# -*- coding: utf-8 -*-
"""
Helpers of the system tier: the fake model, the run directory, expected results.

Kept out of conftest.py as an importable module (see tests/support.py).
"""
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

from tests.support import (
    OLLAMA_URL,
    deep_merge,
    ollama_reply,
    split_overrides,
    write_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = REPO_ROOT / "llm-conductor.py"

TRANSLATE_PROMPT = "Translate the text: "
STRICT_PROMPT = "Translate strictly the text: "
VERIFY_PROMPT = "Verify the text: "
PROMPT_TAIL = "the text: "

SAMPLE = {
    "meta": {"version": 3, "enabled": True, "ratio": 0.75, "owner": None},
    "ui": {
        "title": "Welcome to the Guild",
        "empty": "",
        "gold_counter": "You have {0} gold coins.",
        "warning": "<b>Warning:</b> this action cannot be undone.",
        "multiline": "First line.\nSecond line.",
    },
    "dialogues": [
        {"greeting_male": "Welcome back.", "greeting_female": "Welcome back."},
    ],
    "options": ["Easy", 1, False],
}


def base_docs(**overrides) -> Tuple[dict, dict]:
    """
    (llm.toml, conductor.toml) documents of a run: Ollama on the fake host,
    INFO, ./cache and ./logs next to conductor.toml.

    Args:
        overrides: Double-underscore TOML keys, see tests.support.split_overrides.
    """
    llm = {
        "translation": {
            "api": {"url": OLLAMA_URL, "type": "ollama", "model": "test-model", "max_retries": 2},
            "prompts": {"system": "", "strict": STRICT_PROMPT, "style": TRANSLATE_PROMPT},
            "sampling": {"stop": []},
        },
        "verification": {
            "api": {"url": OLLAMA_URL, "type": "ollama", "model": "test-model", "max_retries": 2},
            "prompts": {"system": "", "edit": VERIFY_PROMPT},
            "sampling": {"stop": []},
        },
    }
    conductor = {"logging": {"level": "INFO", "dir": "./logs"}, "cache": {"dir": "./cache"}}
    llm_overrides, conductor_overrides = split_overrides(**overrides)
    return deep_merge(llm, llm_overrides), deep_merge(conductor, conductor_overrides)


class EchoModel:
    """
    Fake Ollama server for responses: "translates" into RU(...), "proofreads"
    into OK(...), answers "ok" to the preflight probe ("test").

    The leaf value is taken from the end of the user message: call_model_api
    glues prompt and value without a separator, and every prompt of a run
    ends with PROMPT_TAIL. [Tn] markers come back unchanged - an obedient model.

    on_model_call, when set, is called with the 1-based number of every model
    call before it is answered - e.g. to deliver a signal mid-run.
    """

    def __init__(self):
        self.requests = []
        self.fail_model_calls = False
        self.on_model_call: Optional[Callable[[int], None]] = None
        # Substrings of the user message that trigger a content-validation
        # reject (literal newline in the reply). Used to leave residual
        # uncached leaves for cache-rerun scenarios.
        self.reject_leaf_substrings: Tuple[str, ...] = ()

    def __call__(self, request):
        payload = json.loads(request.body)
        content = payload["messages"][-1]["content"]
        self.requests.append(payload)

        if content == "test":
            return 200, {}, json.dumps(ollama_reply("ok"))
        if self.on_model_call is not None:
            self.on_model_call(len(self.model_calls))
        if self.fail_model_calls:
            return 500, {}, json.dumps({"error": "model crashed"})

        value = content.rsplit(PROMPT_TAIL, 1)[1]
        wrapper = "OK" if VERIFY_PROMPT in content else "RU"
        reply = f"{wrapper}({value})"
        if any(fragment in content for fragment in self.reject_leaf_substrings):
            # Literal \n always fails validate_model_reply (newline_in_reply).
            reply = f"{reply}\nchatter"
        return 200, {}, json.dumps(ollama_reply(reply))

    @property
    def model_calls(self):
        """Model requests excluding preflight probes."""
        return [p for p in self.requests if p["messages"][-1]["content"] != "test"]

    def calls_with(self, fragment):
        return [p for p in self.model_calls if fragment in p["messages"][-1]["content"]]


class Workspace:
    """
    Working directory of a run: configuration files, inputs, artifacts.

    conductor.toml lives in root, so the shared result directory of a target
    is root/result/<target name>/ (ResultLayout without --result2target).
    """

    def __init__(self, root: Path):
        self.root = root

    def write_configs(self, **overrides) -> Path:
        """Write conductor.toml and llm.toml from base_docs(**overrides); returns conductor.toml."""
        return write_configs(self.root, *base_docs(**overrides))

    def write_json(self, name: str, data) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read_json(self, name: str):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def result(self, stage: str, relative: str = "sample.json", target: str = "sample.json") -> str:
        """Path (relative to root) of a result in the shared layout: result/<target>/<stage>/<relative>."""
        return f"result/{target}/{stage}/{relative}"

    def logs(self) -> list:
        return sorted((self.root / "logs").glob("conductor_*.log"))

    def log_text(self) -> str:
        """Content of the only log file of the workspace."""
        (log,) = self.logs()
        return log.read_text(encoding="utf-8")


def transform_strings(node, func):
    """Apply func to every non-blank string leaf of a tree (the expected result)."""
    if isinstance(node, dict):
        return {k: transform_strings(v, func) for k, v in node.items()}
    if isinstance(node, list):
        return [transform_strings(v, func) for v in node]
    if isinstance(node, str) and node.strip():
        return func(node)
    return node
