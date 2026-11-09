# Configuration

Two TOML files, split by what changes when you switch models:

| File | Role |
|---|---|
| **`llm.toml`** | Model profile: endpoints, keys, prompts, sampling, `[validation].allowed_scripts` |
| **`conductor.toml`** | Script behaviour: logging, cache, dedup, heartbeat, limits, resilience, reply-validator thresholds |

The full key list, types, defaults, and descriptions live in
`llm_conductor/schema.py`. `llm-conductor init` writes commented templates
from that schema.

## Where files live

- `conductor.toml` — next to the entry point (frozen exe → next to
  `sys.executable`), **not** cwd. `LLMC_CONFIG` overrides the path (tests /
  advanced use).
- `llm.toml` — same directory as `conductor.toml`. `--llm-config PATH`
  (relative to cwd) replaces the whole profile.
- Relative `[logging].dir` / `[cache].dir` resolve from the
  `conductor.toml` directory. The shared `result/` tree lives there too
  unless `--result2target`.
- CLI paths (`--target`, `--glossary`, `--llm-config`) are from cwd.
  Outputs derived from the target (`glossary.json` beside the target,
  `--result2target`, `--cache2target`) are from the target’s directory.

Loading is strict: unknown sections/keys and wrong types fail with a full
problem list (and a “did you mean …?” hint). Missing keys take schema
defaults.

## Secrets

Prefer `[*.api].key_env` (environment variable name). `[*.api].key` is a
literal fallback. An API key is required for `type = "openai"`; ignored for
`ollama`. Version example profiles without literals as `*.example.toml`
(see `.gitignore`).

## Validation scripts (profile)

`llm.toml` → `[validation].allowed_scripts` lists script names the reply
validator treats as always allowed for letters (`latin`, `cyrillic`,
`greek`, `arabic`, `hebrew`, `cjk`, `thai`, …). Empty default means **no**
script is unconditionally allowed — set this for your target writing
system. Length limits, `latin_text_policy`, and `save_rejected_replies` stay
in `conductor.toml` `[validation]` (script behaviour, not language pair).

Characters present in the source string or glossary values are always
allowed; so are punctuation / digits / symbols / separators.

`[validation].save_rejected_replies` (default `false`) writes each
content-validator rejection’s full masked reply under
`[logging].dir/rejected/<operation>/` for offline diagnosis.

## Overrides

Priority: **CLI > files > defaults**. There are no CLI flags to override
model URL or name — use another `--llm-config`. Run flags that affect
`conductor.toml` (`--verb`, `--no-cache`, `--cache2target`) apply via
immutable `RunOverrides` on a copy of the loaded data.

## Related

- [cli.md](cli.md) — commands and result layout
- [prompts.md](prompts.md) — what goes into the model
- [pipeline.md](pipeline.md) — cache, validator, resilience
