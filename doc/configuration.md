# Configuration

Configuration files, split by what changes when you switch models:

| File | Role |
|---|---|
| **`llm.toml`** | Model profile: endpoints, keys, prompts, sampling, content rules (`[processing]`), `[validation].allowed_scripts` |
| **`conductor.toml`** | Script behaviour: logging, cache, dedup, heartbeat, limits, resilience, reply-validator thresholds |
| **`context.toml`** | Key-context catalog: `[[rules]]`, optional exact `[[leaves]]` map and `[templates.<name>]`. `init` writes the suffix catalog. A selected file is the whole catalog |

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
- `context.toml` written by `init` sits beside `conductor.toml`. A run uses
  `--context` (JSON targets), `context.toml` beside the target, then that
  program file. A missing catalog means no key context. See [cli.md](cli.md).
- CLI paths (`--target`, `--glossary`, `--context`, `--llm-config`) are from cwd.
  Outputs derived from the target (`glossary.json` beside the target,
  `--result2target`, `--cache2target`) are from the target’s directory.
- **Run manifest** (`.toml` with `[[targets]]`): `original` / `draft` /
  manifest-relative `llm_config` / `glossary` resolve from the **manifest
  directory** and must stay inside it. Job fields are set in the manifest
  only (precedence B); session flags `--verb` / `--no-cache` still apply.
  See [cli.md](cli.md#run-manifest-toml).

Loading is strict: unknown sections/keys and wrong types fail with a full
problem list (and a “did you mean …?” hint). Missing keys take schema
defaults.

## String content rules

In `llm.toml`, `[processing].required_content_pattern` decides whether a nonblank string is
eligible; `[processing].mask_pattern` selects protected fragments. Defaults
retain Unicode-letter eligibility and protection of `{0}`, newlines, and
XML-like tags. Empty strings disable the corresponding rule, not the separate
blank-leaf skip. Custom masking replaces the complete default expression.
Each manifest target uses the rules of its effective `llm_config` profile;
omitted keys take schema defaults, independent of other targets.

See [content.md](content.md) for the contract and copyable recipes: letters
only, numeric normalization, HTML/XML, named/positional placeholders,
double/dollar braces, printf, and combined protection.

## Key-context catalog

`context.toml` is authored configuration, validated by `CONTEXT_SCHEMA` in
`schema.py`. `[[rules]]` rows have `name`, `match` (`last` or `path`),
`pattern` (Python regex), `slot`, `priority`, and exactly one of `text` or
`template` (plus optional `value`, which replaces `{value}` in
`[templates.<name>].text`). The first matching rule in file order fills a
slot. `priority` only orders different slots in the prompt.

`init` writes the current suffix catalog. Loading a file does not put
deleted rows back. `check` validates `context.toml` beside the program when
the file exists; a missing file is fine. An invalid catalog on a run exits 2.
An invalid program file on `check` exits 3.

### Existing installations

To retain the former suffix hints, copy
[`examples/context.example.toml`](../examples/context.example.toml) to
`context.toml` beside `conductor.toml`. `init` refuses to overwrite existing
configuration files; no catalog is silently created or merged during a run.

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
`[logging].dir/rejected/<operation>/` for offline diagnosis. Names are unique
across input files, leaf paths, and repeated attempts; a `.meta.json` sidecar
records the source file, leaf `report_key`, operation, and usage attempt id.

## Overrides

**JSON `--target`:** priority **CLI > files > defaults**. There are no CLI
flags to override model URL or name — use another `--llm-config`. Run flags
that affect `conductor.toml` (`--verb`, `--no-cache`, `--cache2target`) apply
via immutable `RunOverrides` on a copy of the loaded data.

**Run manifest `.toml`:** job fields (`stage`, `strict`, `llm_config`,
`glossary`, `context`, `result2target`, `cache2target`) come only from the manifest;
passed job CLI flags → exit 2. Session flags `--verb` / `--no-cache` /
`--dry-run` / `--preview-prompts` still apply (`--verb` overrides manifest `verb`). Per-target
fields win over run-level defaults (`target ?? run ?? default`).
Each entry's effective `llm_config` supplies its allowed writing systems for
both translation and verification.

## Related

- [cli.md](cli.md) — commands and result layout
- [prompts.md](prompts.md) — what goes into the model
- [pipeline.md](pipeline.md) — cache, validator, resilience


### Exact sidecar context map

The selected `context.toml` is also the optional sidecar map. Add one
`[[leaves]]` row per structural path and slot. `path` is a JSON-array string:
string elements are literal object keys and non-negative integers are array
indices. `[]` addresses a root string. Thus `'["a.b"]'` and `'["a","b"]'`
address different leaves. Duplicate path/slot pairs and invalid paths fail
before inference. Entries need not exist in every input that shares a catalog;
unmatched entries have no effect.

```toml
[templates.referent]
text = "The third-person referent is {value}. Preserve the roles of other participants."

[[leaves]]
path = '["Dialog.Interpolation.Pronoun.Female.ObjectCase"]'
slot = "gender"
template = "referent"
value = "feminine"

[[leaves]]
path = '["dialogues",0,"text"]'
slot = "speaker"
text = "The speaker is masculine; do not reassign the gender of other participants."
```

Map rows accept `path`, `slot`, `priority`, and exactly one of `text` or
`template`, with optional `value`. A mapped slot wins over all regex rules for
that leaf. Regex rules still fill other slots. Exact mappings never inherit
to descendants. For regex rules, local matches win first; then the nearest
ancestor fills remaining slots before more distant ancestors. Inherited hints
receive a parent annotation and reduced priority/confidence. File order breaks
conflicts inside a regex slot; priority orders different slots only.

### Dotted-key and role recipes

The following replaces the default catalog with referent hints, rather than
assuming that a gender marker always describes the speaker. Patterns use
Python `re.search`; use anchors or boundaries when matching whole segments.
`last` sees the actual last key (an entire dotted key for flat JSON); `path`
sees the human-readable dotted/indexed path and cannot distinguish literal dots
from structural separators. Use `[[leaves]]` when that distinction matters.

```toml
[[rules]]
name = "female-referent"
match = "path"
pattern = '(?i)(?:^|[._])(?:noun_)?female(?:[._]|$)'
slot = "gender"
text = "The third-person referent is feminine; preserve the roles of other participants."

[[rules]]
name = "male-referent"
match = "path"
pattern = '(?i)(?:^|[._])(?:noun_)?male(?:[._]|$)'
slot = "gender"
text = "The third-person referent is masculine; preserve the roles of other participants."
```

These patterns cover `.Male`, `.Noun_Male.134`, and
`Dialog.Interpolation.Pronoun.Female.ObjectCase`. A different project may use
speaker hints instead; role labels are authored text, not inferred semantics.
The final context texts affect cache/dedup identity; exact map paths do not.
