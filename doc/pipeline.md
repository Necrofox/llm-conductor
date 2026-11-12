# Pipeline internals

Operator-facing view of what happens to each string leaf. For architecture
rationale kept locally, see `history/` on a maintainer checkout.

## Leaf processing (order)

1. Blank / whitespace-only → return as-is.
2. Content rule: peel boundaries for an eligibility view, replace each
   protected match with one space, and search `required_content_pattern`.
   No match → return as-is; no cache lookup or LLM round-trip. Defaults
   require a Unicode letter outside formatting. See [content.md](content.md)
   for the contract and regex recipes, including numeric/date processing.
3. Peel leading/trailing whitespace from the leaf (restore after accept).
4. Build glossary snippet (on verify: match terms in the **original**).
5. Linguistic hints from the key path.
6. Assemble enhanced prompt ([prompts.md](prompts.md)).
7. Full cache key + component fingerprints (on the **full** leaf, including
   outer whitespace).
8. Persistent cache lookup; in-run dedup (translation only) on the same key.
9. Mask formatting tags on the peeled core → call translate/verify → unmask.
10. Transport/parse/truncation/refusal failures → degrade to original; not
    cached; circuit breaker counts a failure.
11. If `[validation].enabled`, content validator (vs peeled core); reject →
    degrade, no cache.
12. Accept → restore outer whitespace; record success; write cache / update
    dedup.

Invariant: glossary, hints, and cache key use the **unmasked** full source;
the model sees the masked peeled core; outer whitespace is owned by the
client (not trusted to the model or reply `strip()`); the cache stores the
**unmasked** result with whitespace restored.

## Cache and dedup

| Mechanism | Persistence | Key |
|---|---|---|
| Disk cache | `{cache_dir}/translation_cache.json` / `verification_cache.json` | Hash of full request |
| Glossary regex cache | Memory | Term + boundary mode |
| Dedup index | Memory, this run | Same as disk cache key |

Full-request identity (see `CACHE_KEY_VERSION` / `CACHE_SAMPLING_IDENTITY_FIELDS`
in `llm_conductor/constants.py`) includes the operation type, API URL, model name,
system prompt, assembled user prompt (hints/glossary/original), mode, leaf value,
the active profile's allowed writing systems, both content-rule expressions,
and generation-affecting sampling
(`temperature`, `top_p`, `top_k`,
`repeat_penalty`, `frequency_penalty`, `presence_penalty`, `num_predict`,
`num_ctx`, `stop`). Changing any of these components
misses the cache. Transport-only fields (`keep_alive`, `max_tokens_param`) are
not part of identity. A key-schema bump drops stale disk entries on load.

`--no-cache` / `[cache].enabled = false`: no disk read/write; in-memory dedup
still runs. Cache dir may still be created for `progress_state.json`.

Rejected validator replies are **never** cached. A warm-cache rerun that only
pays residual misses (new strings or past rejects) can trip exit code `5`
even after a previous full run exited `0`. Fix prompts/model or use
`--no-cache` to remeasure.

## Reply validation

When enabled, checks (first failure wins) include: literal protected newlines
in the masked reply; marker mismatches; fabricated protected fragments;
abnormal length; foreign scripts
outside `allowed_scripts` / source / glossary (Latin may warn or reject per
`latin_text_policy`). Soft Latin warnings do not reject.

A `tag_marker_mismatch` rejection logs structured `missing` / `duplicated` /
`extra` marker lists. Set `[validation].save_rejected_replies = true` to also
write each full masked reply under `[logging].dir/rejected/<operation>/` with
a unique filename (source file, leaf `report_key` digest, usage attempt id,
and a nanosecond stamp) plus a matching `.meta.json` attribution sidecar.
Diagnostic write failures are logged and never interrupt leaf fallback.

## Resilience

- **Per-leaf degradation:** network/parse/truncation/refusal never abort the
  whole tree; the original string is kept. The outcome is recorded as
  `degraded` from the request/validation result — never inferred from
  `result == source` (names and ids may legally stay identical).
- **Verify gates** (standalone `--stage verify` with a loaded leaves sidecar,
  and same-run verify after translate). Decisions come from stage outcomes /
  hashes — never from `value == original`:
  - `translation_degraded` → `verify_skipped` (no editor call on the English
    fallback).
  - no translation stage record for a leaf after loading translation history
    for its file →
    `verify_skipped` with reason `translation_missing` (e.g. leaf never
    reached after a circuit-breaker abort). Keeps the current tree value.
    A sidecar loaded for one file must not gate later draft entries in the
    same manifest.
  - translation `source_hash` present and unequal to the hash of the current
    original leaf → `degraded` with `error_kind=source_stale`; restore the
    current original (no editor call). With `fail_on_any_error` enabled,
    this yields exit `5`. The request and file failure ratios exclude it.
    The check runs before the blank / format-only skip, so `final/` gets
    the current original even when the old draft is blank or placeholder-only.
  - Verify-only from `translated/`: JSON structure (keys, types, array lengths)
    must match the current original before any editor call or `final/` write.
    Text-only leaf changes at the same path stay on the `source_stale` path.
  - Schema v1 sidecars (dotted leaf keys) are rejected; re-run `translate` to
    regenerate schema v2. A missing sidecar or a legacy verification-only
    sidecar still allows verify without translation gates.
  - Frozen `draft` verify uses the draft marker/structure preflight and leaves
    any translation sidecar in place.
  Status is persisted under `result/<target>/leaves/<relative>` so a later
  verify still sees prior translation outcomes. Verify without translation
  history records outcomes in `run_report.json`.
- **Circuit breaker:** `[resilience].max_consecutive_failures` stops further
  leaves for that stage (same path as shutdown); exit `6`.
- **Failure ratio:** after the run, request and file failure ratios vs
  `max_failure_ratio` → exit `5` if exceeded.
- **`fail_on_any_error`:** when `[resilience].fail_on_any_error = true`, any
  degraded leaf forces exit `5` even if the ratio stays under threshold
  (default `false`).
- **Shutdown:** first SIGINT/SIGTERM saves caches and exits `130`; results may
  be incomplete.

## Reports

After a run, the result root also gets:

| File | Contents |
|---|---|
| `run_report.json` | Per-leaf stage status (`ok` / `cache_hit` / `dedup` / `degraded` / `verify_skipped` / …), hashes, aggregates |
| `usage.json` | Token attempts by provider/model/stage (retries included); outcomes `success` / `rejected` / `failed`; preflight separate; cache/dedup non-API counts; unknown usage marked |
| `leaves/<relative>` | Per-file sidecar used for verify gates across stage splits (`<relative>` is the same path as under `translated/` / `final/`) |

Leaf identities in `run_report.json` / sidecars (schema version **2**) are
compact JSON arrays of `KeyPath` segments (e.g. `["a.b"]` vs `["a","b"]`),
so a literal dotted key cannot collide with a nested path. Logs still show
the human dotted form.

**File identities** (the `files` map keys in `run_report.json`, and
`relative_path` in each leaves sidecar) are target-relative paths with
**POSIX separators** (`/`), even on Windows. Example: a manifest entry
`original = "input/route-separate.json"` always serializes as
`input/route-separate.json`, never `input\\route-separate.json`. Physical
result paths under `translated/` / `redacted/` / `final/` / `leaves/` still
use the host OS layout. Schema version stays **2**; older native-separator
file keys are accepted on read (normalized to `/`) so records are not
dropped — consumers of archived Windows reports should apply the same
`\\` → `/` mapping when matching keys.

Aggregate units in `run_report.json`:

| Field | Counts |
|---|---|
| `skipped` | Content/blank stage records (a leaf can contribute once per stage) |
| `verify_skipped` | Verification skipped (`translation_degraded` or `translation_missing`) |
| `api_calls` | Leaf stage outcomes with status `ok` or `degraded` — **not** HTTP attempts |
| `usage.json` attempts | Every HTTP try including retries, plus a separate preflight bucket |

The final log banner includes explicit `untranslated` / `degraded` /
`verify_skipped` counts.

## Metrics

API metrics count every leaf call; failures break down by kind; validator
rejections are counted separately. `usage.json` additionally counts every
HTTP attempt (including retries) and preflight probes; a reply rejected by
content validation keeps its token totals but is marked `rejected` via the
attempt id (safe if leaves later run concurrently). Token banners and
stage summaries are English. Clean log-file retention depends on
`[logging].level` (see schema / `init` comments).
