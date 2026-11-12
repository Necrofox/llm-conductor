# LLM-Conductor

**CLI pipeline** for localizing nested JSON string resources with language
models (Ollama or any OpenAI-compatible API). Optional second stage
proofreads with a separate endpoint.

This is **not** a local machine-translation product, offline dictionary, or
GUI translator. There is no built-in MT engine — you supply the model and the
prompts; quality follows them. Think “batch orchestration for LLM
localization,” not “install and translate offline.”

**Language pair = prompt only.** The engine does not hard-code source or
target languages. Name them in `llm.toml`. The default `init` profile is one
working example (often English→another language), not a product limit.

## Features

- Recursive JSON of any depth; non-strings left untouched
- Per-string isolation (no batching unrelated keys into one prompt)
- Key-name linguistic hints (`_male` / `_female` / `_plural` / …)
- Glossary injection, formatting-tag masking, disk cache + in-run dedup
- Reply validation, consecutive-failure breaker, OS locks, progress heartbeat
- Independent transports for translation vs verification

## Quick start

### 1. Install or grab a binary

From PyPI:

```bash
pip install llm-conductor
```

From a clone (editable install + dev tools):

```bash
git clone https://github.com/Necrofox/llm-conductor.git
cd llm-conductor
pip install -e ".[dev]"
```

Or download `llm-conductor.exe` from
[GitHub Releases](https://github.com/Necrofox/llm-conductor/releases).

### 2. Configs next to the program

```bash
llm-conductor init               # conductor.toml + llm.toml
# portable checkout:
python llm-conductor.py init
```

Configs resolve **beside the entry point** (exe / wrapper / installed shim),
not cwd. Edit `llm.toml` for your endpoint and prompts; set
`[validation].allowed_scripts` for the target writing system (e.g.
`["cyrillic"]`, `["latin"]`, `["cjk"]`). Prefer `key_env` over literal keys.

### 3. Run

```bash
llm-conductor run --target path/to/strings.json
llm-conductor run --target path/to/strings.json --stage translate --strict
llm-conductor run --target project/ui.toml
llm-conductor run --target project/ui.toml --dry-run   # same resolver; no API
```

Batch runs use a `.toml` run manifest (`[[targets]]`); `.list` is removed.
Results land under `result/<target-name>/` (`translated/`, `redacted/`,
`final/`) unless `result2target` (CLI for a lone JSON, or a field in the
manifest). See
[doc/cli.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/cli.md).

### Ollama vs OpenAI-compatible

| | Ollama | OpenAI-compatible |
|---|---|---|
| `type` | `ollama` | `openai` |
| URL | `…/api/chat` | `…/v1/chat/completions` |
| Key | ignored | `key_env` / `key` required |

Point translation and verification at different hosts if you want.

## Example

Input (see
[examples/sample.json](https://github.com/Necrofox/llm-conductor/blob/main/examples/sample.json)):

```json
{
  "npc": {
    "greeting_male": "Welcome, traveler.",
    "greeting_female": "Welcome, traveler."
  }
}
```

Same English text, different key suffixes → separate requests and (usually)
different grammatical forms in the target language. Full prompt rules:
[doc/prompts.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/prompts.md).

## Key-name conventions

| Pattern | Hint |
|---|---|
| `*_male`, `*_female`, `*_neutral` | gender |
| `*_plural`, `*_singular`, `*_few`, `*_zero`, `count_*` | number |
| `*_formal`, `*_informal` | formality |

## Documentation

| Doc | Topic |
|---|---|
| [doc/concepts.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/concepts.md) | What it is / is not; isolation; language=prompt |
| [doc/configuration.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/configuration.md) | TOML split, paths, secrets |
| [doc/cli.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/cli.md) | Commands, results, exit codes |
| [doc/prompts.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/prompts.md) | Authoring prompts |
| [doc/pipeline.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/pipeline.md) | Cache, validation, resilience |
| [doc/content.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/content.md) | String content contract and copyable regex recipes |
| [CONTRIBUTING.md](https://github.com/Necrofox/llm-conductor/blob/main/CONTRIBUTING.md) | Dev setup and conventions |
| [AGENTS.md](https://github.com/Necrofox/llm-conductor/blob/main/AGENTS.md) | Notes for AI coding agents |

## Cache note

Rejected model replies are not cached. A warm-cache rerun that only hits
residual misses can exit with code `5` even if an earlier full run was `0`.
Fix prompts/model or remeasure with `--no-cache`
([doc/cli.md](https://github.com/Necrofox/llm-conductor/blob/main/doc/cli.md)).

## Roadmap (high level)

- Richer input formats (e.g. subtitle adapters)
- Optional in-stage parallelism with single-flight on the full cache key
- Stage-local resume orthogonal to the LLM cache (concept)

## License

MIT — see [LICENSE](https://github.com/Necrofox/llm-conductor/blob/main/LICENSE).

---

## Кратко (RU)

**LLM-Conductor** — CLI-конвейер: гоняет строковые листья JSON через LLM
(Ollama / OpenAI-совместимые API), опционально второй моделью вычитывает
перевод. Это **не** локальный переводчик и не движок MT — модель и промпты
ваши. Пара языков задаётся только текстом промпта в `llm.toml`, не схемой
движка. Быстрый старт: `init` → правки профиля → `run --target …`. Документация
на английском в `doc/`.
