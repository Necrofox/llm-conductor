# Prompt authoring

Prompts live in `[translation.prompts]` / `[verification.prompts]` of
`llm.toml` (multiline `'''…'''` literals are stored byte-for-byte). The
**language pair is named only in those texts** — see [concepts.md](concepts.md).

## What the model receives

Each JSON string is its own request:

| Role | Source |
|---|---|
| `system` | `[translation.prompts].system` or `[verification.prompts].system` |
| `user` | Mode prompt (`strict` / `style` / `edit`) with injected blocks **plus the leaf string concatenated with no separator** |

End the mode prompt so the leaf follows cleanly, usually with a labelled
newline:

```text
...\nText to translate:\n
```

Without that boundary the instruction and the payload blur together.

## Formatting markers

Before the call, placeholders `{0}`, newlines, and HTML/XML-like tags become
`[T0]`, `[T1]`, … and are restored afterward.

- **Do not list sample markers** (`[T0]`, `{0}`, `%s`, …) in the system
  prompt — samples encourage fabricated or mismatched markers. Say only that
  every formatting marker already in the text must be preserved in place.
  The model never sees raw `{0}` / tags — only `[Tn]` (or another prefix if
  the source already contains literal `[T…]`).
- Markers may move with word order; lost or invented markers fail validation
  when the reply validator is enabled ([pipeline.md](pipeline.md)).

## `{LINGUISTIC_CONTEXT}` and `{GLOSSARY}`

The pipeline injects titled blocks. Prefer **both markers** explicitly,
immediately before the text label. Empty blocks are omitted.

| Markers in the prompt | Placement |
|---|---|
| both | each at its marker |
| only `{LINGUISTIC_CONTEXT}` | context, then glossary after it |
| only `{GLOSSARY}` | context then glossary at the marker |
| none | before the last sentence containing `translate` / `verify`, else appended |

Hints come from key-name conventions ([concepts.md](concepts.md)). Same text
with different hints does not share a cache entry.

## Glossary

`glossary.json` is `{ "term": "translation" }`. Only terms found in the
**current** leaf (whole-word, case-insensitive) are injected. Terms with
edge punctuation (`C++`, `.NET`) still match. Identity entries pin proper
names: `{"Arasaka": "Arasaka"}`. On verification, terms are matched against
the **original** leaf, not the translation.

## System prompt: output only the result

Common failure mode: chatter, alternatives, or apologies. Useful rules:

- Output **only** the final text — no explanations or alternatives.
- Even a single-word UI label is text to process — output only its result.
- Do not mix source-language words into the target when that would be wrong
  for your pair (spell that out for your languages).

`stop` sequences: real newlines in the source are already masked, so `\n\n`
in the reply is almost always chatter. Note that request `stop` **replaces**
the model’s defaults — include your model’s end tokens as well. OpenAI-style
APIs allow at most four stop strings.

## Verification prompts

The second stage receives **already translated** text. Say so explicitly
(forbid translating back). Allow “output unchanged if already correct.”

### `{ORIGINAL_TEXT}`

Injects a masked original as a reference block (separate mask map — `[Tn]`
numbers may not align with the leaf). Tell the system prompt not to copy
`[ORIGINAL]` into the answer. Without the marker, the block is inserted
before the `verify` sentence (or at the end). The assembled prompt is part of
the cache key.

## Prompts and cache

The cache key hashes the full request (model, system prompt, assembled mode
prompt with hints/glossary/original, mode, value, and generation-affecting
sampling — see [pipeline.md](pipeline.md)). Editing a prompt or sampling
parameter misses the cache automatically — no manual cache wipe for those
changes. Large files mean a full re-run after a prompt edit.

## Checklist

- [ ] Languages named explicitly in the prompts
- [ ] System prompt preserves markers **without** listing samples
- [ ] No-chatter rule; short-string rule if needed
- [ ] `{LINGUISTIC_CONTEXT}{GLOSSARY}` before the text label
- [ ] Mode prompt ends with a text label + newline
- [ ] `stop` includes `\n\n` and your model’s end tokens
- [ ] Verification forbids reverse translation; includes `{ORIGINAL_TEXT}`
- [ ] `[validation].allowed_scripts` matches the target writing system
- [ ] Spot-check with `examples/sample.json` on real inference
