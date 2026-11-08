# Concepts

LLM-Conductor is a **CLI pipeline** that sends string leaves from nested JSON
through language-model APIs (Ollama or any OpenAI-compatible endpoint), then
optionally through a second model for proofreading. You bring the model and
the prompts; this tool orchestrates files, caching, validation, and exit codes.

It is **not** a local machine-translation engine, offline dictionary, or
end-user translator app. There is no built-in MT model and no GUI. Output
quality is entirely a function of the model and the prompts in your profile.

## Two stages

1. **Translation** — each string leaf is sent to the translation endpoint
   (`strict` or `style` mode).
2. **Verification** — the translated leaf is proofread by a (possibly
   different) endpoint. Stages can run together (`run`) or separately
   (`--stage translate` / `--stage verify`).

Translation and verification transports are configured independently
(typical pattern: cloud for translation, local Ollama for verification).

## Per-string isolation

Each JSON string is a **separate** model request. Neighbouring keys are never
batched into one prompt. Game and app resource trees often place unrelated
lines next to each other; packing them into one context pollutes the model.
Identical text under different linguistic hints (for example `greeting_male`
vs `greeting_female`) is translated separately — hints participate in the
cache key (see [pipeline.md](pipeline.md)).

## Language pair = prompt only

The engine does **not** know a source or target language. There is no
`target_lang` (or similar) in the schema. Name the languages in the system
and mode prompts inside `llm.toml`. Changing the pair means editing the
profile or passing another `--llm-config`.

Linguistic hints derived from key names describe grammatical properties
(“if the target language has gender…”) — not rules of a specific language.
The default `init` template and some examples may illustrate one working
pair; that is a convenience sample, not a product limit.

For non-Latin (or mixed) targets, set `[validation].allowed_scripts` in the
**profile** (`llm.toml`) so the reply validator accepts the right scripts
(see [configuration.md](configuration.md)).

## Key-name conventions

Hints come from the **last segment** of the JSON path (suffix match, not
substring):

| Key pattern | Hint |
|---|---|
| `*_male`, `*_female`, `*_neutral` | grammatical gender |
| `*_plural`, `*_singular`, `*_few`, `*_zero`, `count_*` | number / count form |
| `*_formal`, `*_informal` | formality |
| parent segment with a suffix (`npc_female.greeting`) | hint may inherit downward |

## Two ways to run

Configs (`conductor.toml`, `llm.toml`) resolve next to the **program**, not
the process cwd (scheduler-friendly):

1. **Portable** — frozen `.exe`, or `python llm-conductor.py …` from a
   checkout; configs sit beside that entry point.
2. **Installed** — `pip install` then `llm-conductor …`; configs sit beside
   the console-script shim.

Prefer those two over `python -m llm_conductor` when you want configs in the
repo root (`-m` resolves next to the package). Details:
[configuration.md](configuration.md), [cli.md](cli.md).
