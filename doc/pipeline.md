# Pipeline internals

Operator-facing view of what happens to each string leaf. For architecture
rationale kept locally, see `history/` on a maintainer checkout.

## Leaf processing (order)

1. Blank / whitespace-only → return as-is.
2. Format-only (no letters outside markup / placeholders / residual
   punctuation) → return as-is; no cache lookup or LLM round-trip.
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
in `llm_conductor/constants.py`) includes: operation type, model name, system
prompt, assembled user prompt (hints/glossary/original), mode, leaf value, and
generation-affecting sampling — `temperature`, `top_p`, `top_k`,
`repeat_penalty`, `frequency_penalty`, `presence_penalty`, `num_predict`,
`num_ctx`, `stop`. There is **no** cross-parameter reuse: changing any of these
misses the cache. Transport-only fields (`keep_alive`, `max_tokens_param`) are
not part of identity. A key-schema bump drops stale disk entries on load.

`--no-cache` / `[cache].enabled = false`: no disk read/write; in-memory dedup
still runs. Cache dir may still be created for `progress_state.json`.

Rejected validator replies are **never** cached. A warm-cache rerun that only
pays residual misses (new strings or past rejects) can trip exit code `5`
even after a previous full run exited `0`. Fix prompts/model or use
`--no-cache` to remeasure.

## Reply validation

When enabled, checks (first failure wins) include: literal `\n` in the masked
reply; marker mismatches; fabricated tags; abnormal length; foreign scripts
outside `allowed_scripts` / source / glossary (Latin may warn or reject per
`latin_text_policy`). Soft Latin warnings do not reject.

A `tag_marker_mismatch` rejection logs structured `missing` / `duplicated` /
`extra` marker lists. Set `[validation].save_rejected_replies = true` to also
write the full masked reply under `[logging].dir/rejected/<operation>/`.

## Resilience

- **Per-leaf degradation:** network/parse/truncation/refusal never abort the
  whole tree; the original string is kept. The outcome is recorded as
  `degraded` from the request/validation result — never inferred from
  `result == source` (names and ids may legally stay identical).
- **Verify skip:** by default, leaves with `translation_degraded` do **not**
  enter normal verification (no editor call on the English fallback). Status
  is persisted under `result/<target>/leaves/<relative>.json` so a standalone
  `--stage verify` still sees the flag.
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
| `usage.json` | Token attempts by provider/model/stage (retries included); preflight separate; cache/dedup non-API counts; unknown usage marked |
| `leaves/<relative>.json` | Per-file sidecar used for verify-skip across stage splits |

The final log banner includes explicit `untranslated` / `degraded` /
`verify_skipped` counts.

## Metrics

API metrics count every leaf call; failures break down by kind; validator
rejections are counted separately. `usage.json` additionally counts every
HTTP attempt (including retries) and preflight probes. Token banners and
stage summaries are English. Clean log-file retention depends on
`[logging].level` (see schema / `init` comments).
