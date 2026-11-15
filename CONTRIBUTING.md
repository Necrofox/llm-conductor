# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Python **≥ 3.11** (development uses 3.13).

## Versions and builds

Git release tags (`vMAJOR.MINOR.PATCH`) are the version source of truth.
setuptools-scm generates `llm_conductor/_version.py` during package builds
and installs; this file is ignored by Git. Do not edit it or bump a version
literal in `__init__.py`.

A clean checkout at `v1.1.1` builds as `1.1.1`. Five commits after that tag
normally build as `1.1.2.dev5+g<commit>`; uncommitted tracked changes add a
date suffix. This guesses the next patch, not the intended next release.
Versions follow reachable tags, not the highest tag elsewhere in the repo.
When releases tag merge commits on `main`, merge `main` back into `dev` after
each release so subsequent dev builds use that release as their baseline.
Build jobs need full Git history and tags (`fetch-depth: 0` in Actions).

Build a local test package or Windows executable with:

```bash
pip install -e ".[dev]"
python -m build
pyinstaller --clean --noconfirm llm-conductor.spec
python llm-conductor.py --version
python tools/smoke_frozen.py dist/llm-conductor.exe --expected-version <version>
```

The PyInstaller spec refreshes the generated version on every build. An
editable install captures the version at installation time: reinstall after
switching commits if you need the source CLI to report the updated version.
Without generated metadata, a checkout can still run and reports
`0.0.0+unknown`. Package/exe builds require SCM metadata (or the metadata in
an sdist) and fail if it is unavailable; there is no guessed release fallback.
The installed package and exe do not need Git or setuptools-scm at runtime.

Dev versions with a `+g<commit>` local segment are for local testing or CI
artifacts. Publish releases to PyPI only from clean release tags.

Text files use LF through `.gitattributes`; Python files also use Git's
built-in Python diff driver. Configure your editor to save text with LF.

## Checks

```bash
ruff check .
mypy llm_conductor
pytest                 # unit + integration + system; no network
pytest -m unit         # or integration | system
```

Live inference (`pytest -m live`) is opt-in — only with a prepared endpoint.
Mutation shield (when touching critical paths): `python tools/mutation_check.py`.
`KILLED` means a test detected the changed behavior; `SURVIVED` means it passed.
`HARNESS_ERROR` means pytest could not provide a valid verdict (collection,
setup/teardown, import, timeout, or runner failure). Fix the harness before
interpreting mutation coverage. Harness errors fail the command even for an
equivalent mutant and never count as kills.

## Conventions

Summarized from [AGENTS.md](AGENTS.md); full operator docs are in `doc/`.

1. Magic numbers and defaults → `llm_conductor/constants.py`
2. Absolute imports only
3. Depend on narrow `Protocol`s from `interfaces.py`, not concrete `Config`
4. File IO via `safe_read_file` / `safe_write_file` / `FileOperations`
5. English Google-style docstrings; no backlog task numbers in package code
6. Shared mutable state under locks (keep them even while the pipeline is
   single-threaded)
7. LLM calls only from `_process_leaf`
8. `sys.exit` only in `cli/main.py`
9. Do not import `llm_conductor.config` from `core` / `l10n` / most of `cli`
10. Schema changes go through `llm_conductor/schema.py` (and `init` templates)

Do not commit secrets (`conductor.toml`, `llm.toml` with literal keys, caches,
logs). Example profiles use `*.example.toml` / `key_env`.

## Pull requests

- Keep changes focused; update `doc/` when behaviour users see changes
- Add or adjust tests with the change
- Ensure ruff / mypy / pytest are clean

Active work after the first public push is tracked in **GitHub Issues**.
Each issue gets exactly one kind label (`bug` / `enhancement` / `feature` /
`concept` / `chore`) — see AGENTS.md and
`.cursor/rules/github-issue-labels.mdc`. Local maintainer journals live under
gitignored `history/` (see AGENTS.md).
