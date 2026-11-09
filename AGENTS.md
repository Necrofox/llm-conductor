# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

**LLM-Conductor** — a CLI **pipeline** that localizes nested JSON string
leaves via Ollama or OpenAI-compatible APIs (optional second-stage
verification). It is not an embedded MT engine or end-user translator.
Language pair = prompts in `llm.toml` only.

Human documentation (English):

- [README.md](README.md) — quickstart and positioning
- [doc/concepts.md](doc/concepts.md) — isolation, language=prompt, run modes
- [doc/configuration.md](doc/configuration.md) — TOML split, paths, secrets
- [doc/cli.md](doc/cli.md) — commands, results, exit codes
- [doc/prompts.md](doc/prompts.md) — prompt authoring
- [doc/pipeline.md](doc/pipeline.md) — leaf flow, cache, validation, resilience
- [CONTRIBUTING.md](CONTRIBUTING.md) — conventions and checks

## Local journals (`history/`)

`history/` is the standard path for **local-only** maintainer material
(backlog, closed-task archive, lint debt, audit notes, publish checklists).
It is gitignored (`/history/`). Never put that content under `doc/`.
Project rule: [`.cursor/rules/history-local.mdc`](.cursor/rules/history-local.mdc).

## GitHub Issues (kind labels)

Active work is tracked in GitHub Issues. Each issue gets **exactly one** kind
label: `bug` | `enhancement` | `feature` | `concept` | `chore`.
Criteria and decision rules:
[`.cursor/rules/github-issue-labels.mdc`](.cursor/rules/github-issue-labels.mdc).
Priority stays in the issue body (or a milestone), not as labels.

## Conventions (short)

1. No magic literals — use `llm_conductor/constants.py`.
2. Absolute imports only (`from llm_conductor.…`).
3. New components take a narrow `Protocol` from `interfaces.py`, not concrete `Config`.
4. All file writes go through `FileOperations` / `safe_write_file`; reads through `safe_read_file`.
5. Public APIs: English Google-style docstrings. No task-number history in code comments.
6. Shared mutable state under `threading.Lock`. Do not remove locks “as unused.”
7. String leaves are processed only in `_process_leaf` — no extra LLM call sites.
8. `sys.exit` only in `cli/main.py`. Pipeline code must not catch bare `BaseException` / bare `except:` (would swallow `RunInterrupted`).
9. Do not import `llm_conductor.config` from `core` / `l10n` / `cli` (except `cli/main.py`). Enforced by `tests/meta/test_import_layers.py`.
10. User-facing CLI and log messages are **English**.

Schema source of truth: `llm_conductor/schema.py`. Prefer linking `doc/` over restating large design essays in this file.

## Checks before finishing a change

```text
ruff check .
mypy llm_conductor
pytest
```

Live inference (`pytest -m live`) is opt-in and needs a prepared endpoint.
Mutation shield: `python tools/mutation_check.py` when touching critical paths.

## Packaging note

Version is `llm_conductor.__version__`. Release exe is built by GitHub
Actions on tag `v*` (see `.github/workflows/`). Do not treat a local
`dist/` as the release source.
