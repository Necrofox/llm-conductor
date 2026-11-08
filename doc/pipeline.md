# Pipeline internals

Operator-facing view of what happens to each string leaf. For architecture
rationale kept locally, see `history/` on a maintainer checkout.

## Leaf processing (order)

1. Blank / whitespace-only → return as-is.
2. Build glossary snippet (on verify: match terms in the **original**).
3. Linguistic hints from the key path.
4. Assemble enhanced prompt ([prompts.md](prompts.md)).
5. Full cache key + component fingerprints.
6. Persistent cache lookup; in-run dedup (translation only) on the same key.
7. Mask formatting tags → call translate/verify → unmask.
8. Transport/parse/truncation/refusal failures → degrade to original; not
   cached; circuit breaker counts a failure.
9. If `[validation].enabled`, content validator; reject → degrade, no cache.
10. Accept → record success; write cache / update dedup.

Invariant: glossary, hints, and cache key use the **unmasked** source; the
model sees masked text; the cache stores the **unmasked** result.

## Cache and dedup

| Mechanism | Persistence | Key |
|---|---|---|
| Disk cache | `{cache_dir}/translation_cache.json` / `verification_cache.json` | Hash of full request |
| Glossary regex cache | Memory | Term + boundary mode |
| Dedup index | Memory, this run | Same as disk cache key |

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

## Resilience

- **Per-leaf degradation:** network/parse/truncation/refusal never abort the
  whole tree; the original string is kept.
- **Circuit breaker:** `[resilience].max_consecutive_failures` stops further
  leaves for that stage (same path as shutdown); exit `6`.
- **Failure ratio:** after the run, request and file failure ratios vs
  `max_failure_ratio` → exit `5` if exceeded.
- **Shutdown:** first SIGINT/SIGTERM saves caches and exits `130`; results may
  be incomplete.

## Metrics

API metrics count every attempt; failures break down by kind; validator
rejections are counted separately. Token banners and stage summaries are
English. Clean log-file retention depends on `[logging].level` (see schema /
`init` comments).
