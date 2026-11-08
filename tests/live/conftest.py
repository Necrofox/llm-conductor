# -*- coding: utf-8 -*-
"""
Fixtures for the live tier: a run against real inference (AGENTS.md (live tier)).

Three guards, all before the first request that loads the model:
  1. the live marker is excluded by default (pytest.ini, -m "not live");
  2. LLMC_LIVE_CONFIRMED=1 — set ONLY after the user confirms that inference
     is ready (the agent must ask);
  3. a no-load preflight: e2e/*.e2e.toml profiles exist, GET /api/tags
     answers and the model is listed. Otherwise — skip, not fail.

Scenarios run once per session (model time is expensive); tests only
assert against their results.
"""
import json
import os
import shutil
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests

from llm_conductor.constants import RESULT_DIR_NAME, STAGE_TRANSLATED
from tests.live.runner import (
    CONDUCTOR_PROFILE,
    PROFILE,
    REPORTS_DIR,
    SAMPLE_FILE,
    load_entry_module,
    prepare_workdir,
    run_scenario,
    write_report,
)

CONFIRM_ENV = "LLMC_LIVE_CONFIRMED"
PREFLIGHT_TIMEOUT = 5
WARMUP_TIMEOUT = 180
KEEP_ALIVE = "1m"  # short on purpose: tests don't need the model warm after they finish


@pytest.fixture(scope="session")
def live_profile():
    """e2e profile and host address; skip if the run is unconfirmed or the profile is missing."""
    if os.environ.get(CONFIRM_ENV) != "1":
        pytest.skip(f"{CONFIRM_ENV}=1 не задан: живой прогон требует подтверждения, "
                    f"что инференс подготовлен (AGENTS.md (live tier))")
    if not all(path.exists() for path in (PROFILE, CONDUCTOR_PROFILE, SAMPLE_FILE)):
        pytest.skip(f"нет {PROFILE}, {CONDUCTOR_PROFILE} или {SAMPLE_FILE} - "
                    f"профиль привязан к локальной сети, см. §9.1")

    profile = tomllib.loads(PROFILE.read_text(encoding="utf-8"))
    parts = urlsplit(profile["translation"]["api"]["url"])
    return {"profile": profile, "base_url": f"{parts.scheme}://{parts.netloc}"}


@pytest.fixture(scope="session")
def live_host(live_profile):
    """
    No-load preflight, then warmup.

    GET /api/tags and /api/ps do not load the model (allowed by §9.1 without
    confirmation). Warmup via /api/generate loads the model into VRAM — it is
    allowed here because LLMC_LIVE_CONFIRMED was already checked in live_profile.
    """
    base_url = live_profile["base_url"]
    profile = live_profile["profile"]
    models = {profile["translation"]["api"]["model"], profile["verification"]["api"]["model"]}
    try:
        tags = requests.get(f"{base_url}/api/tags", timeout=PREFLIGHT_TIMEOUT).json()
    except requests.RequestException as exc:
        pytest.skip(f"хост {base_url} недоступен: {exc}")
    available = {m["name"] for m in tags.get("models", [])}
    missing = models - available
    if missing:
        pytest.skip(f"на {base_url} нет моделей {sorted(missing)}; есть: {sorted(available)}")

    for model in models:
        requests.post(f"{base_url}/api/generate", json={"model": model, "keep_alive": KEEP_ALIVE},
                      timeout=WARMUP_TIMEOUT).raise_for_status()
    return {"base_url": base_url, "models": sorted(models)}


@pytest.fixture(scope="session")
def source_data():
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def live_runs(live_host, tmp_path_factory):
    """
    §11 checklist scenarios on clean caches, once per session:

      full        — full pipeline, style
      full_cached — repeat in the same directory: 0 model requests when the
                    prior full run had no validation rejects (rejects are
                    never cached, so residual misses are re-queried)
      strict      — --strict --stage translate, separate directory
      verify_only — --stage verify on the translation from full, without the verification cache
    """
    module = load_entry_module()
    root = tmp_path_factory.mktemp("live")
    runs = {}

    full_dir = prepare_workdir(root, "full")
    runs["full"] = run_scenario(module, "full", full_dir)
    runs["full_cached"] = run_scenario(module, "full_cached", full_dir)

    strict_dir = prepare_workdir(root, "strict")
    runs["strict"] = run_scenario(module, "strict", strict_dir, "--strict", "--stage", "translate")

    verify_dir = prepare_workdir(root, "verify_only")
    translated = Path(RESULT_DIR_NAME, STAGE_TRANSLATED, "sample.json")
    (verify_dir / translated).parent.mkdir(parents=True)
    shutil.copy2(full_dir / translated, verify_dir / translated)
    runs["verify_only"] = run_scenario(module, "verify_only", verify_dir, "--stage", "verify")

    return runs


@pytest.fixture(scope="session")
def live_report(live_runs, live_host, source_data):
    """Human-readable report — e2e/reports/<time>/report.md (e2e/ is outside git)."""
    report_dir = REPORTS_DIR / time.strftime("%Y%m%d_%H%M%S")
    return write_report(list(live_runs.values()), source_data,
                        f"{', '.join(live_host['models'])} @ {live_host['base_url']}", report_dir)
