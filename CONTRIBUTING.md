# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Python **≥ 3.11** (development uses 3.13).

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
