# CLI and results

## Commands

```text
llm-conductor init                          # write conductor.toml + llm.toml (no overwrite)
llm-conductor check                         # strict validate both TOML files
llm-conductor run --target PATH [options]   # pipeline
llm-conductor run --target PATH --dry-run   # preflight plan (no API / result writes)
llm-conductor --version
```

From a checkout without install: `python llm-conductor.py …` (same configs
layout as the frozen exe). After `pip install`: `llm-conductor …`.

### `run` options

| Flag | Meaning |
|---|---|
| `--target PATH` | Required. `.json` file or `.toml` run manifest |
| `--stage {all,translate,verify}` | Default `all` (JSON targets only) |
| `--strict` | Translation in strict mode (JSON targets only) |
| `--no-cache` | Disable persistent cache for this run |
| `--dry-run` | Validate inputs and print effective settings; no API, locks, or result writes (session flag) |
| `--llm-config PATH` | Alternate profile, cwd-relative (JSON targets only) |
| `--glossary PATH` | Glossary JSON (else `glossary.json` beside target, else empty; JSON targets only) |
| `--verb LEVEL` | Log level override (session flag; also overrides manifest `verb`) |
| `--result2target` | Write `result/` next to the target (JSON targets only) |
| `--cache2target` | Put cache dir next to the target (JSON targets only) |

`check` validates TOML only (not manifests, locks, or live API).
`run --dry-run` is the target/manifest audit: same settings resolver as a real
`run`, exit `0` on a clean plan, exit `2` on contract/path/JSON/glossary/draft
errors, exit `3` when a resolved profile/config fails to load (same as a real
`run`). On success it prints an effective-settings table (index, original,
draft, stage, strict, llm_config, glossary, result paths). Duplicate object
keys in input JSON emit one WARNING per file (last value wins) and do **not**
fail the plan. Dry-run does not acquire locks and does not report lock
contention. There is no `plan` / `inspect` command and no `dry_run` field in
the manifest (unknown key → exit 2).

### Dry-run vs real-run failure modes

Shared input preflight runs for both `run` and `run --dry-run` **before**
locks and API probes. Cheap structural / IO faults that a real run would
reject as unusable inputs fail here with exit `2` (same diagnostics either
way). Dry-run still makes no model calls, takes no locks, and writes no
results.

| Fault | Stages | Dry-run | Real run |
|---|---|---|---|
| Manifest / path / glossary / result-ownership contract | any | exit `2` | exit `2` |
| Unreadable or invalid original JSON | any | exit `2` | exit `2` |
| Root JSON `null` (original) | any | exit `2` | exit `2` |
| Draft missing / unreadable / structure or marker mismatch | verify via `draft` | exit `2` | exit `2` |
| Verify-only without `draft`: missing / unreadable / null / structure-mismatched `translated/` | `verify` | exit `2` | exit `2` |
| Profile / conductor load failure | any | exit `3` | exit `3` |
| Duplicate object keys in input JSON | any | WARNING; exit `0` | WARNING; continues |
| API unavailable | stages that call the model | not checked | exit `4` |
| Target / cache / result lock held | any | not checked | exit `7` |
| Leaf degradation / request or file failure ratio | processing | not applicable | exit `5` (or `6` on circuit breaker) |
| Stale / missing leaves sidecar gates during verify | `verify` without `draft` | not checked (sidecar optional) | warning or per-file fail as documented in [pipeline.md](pipeline.md) |

`translate` and `all` do not require a prior `translated/` tree (the run
writes it). `verify` without `draft` does.

## Run manifest (`.toml`)

Batch input is a **run manifest**: a `.toml` file with a non-empty
`[[targets]]` array. Detection: `--target` ending in `.toml` must parse as
that shape; any other TOML (e.g. `conductor.toml` / `llm.toml`) → exit 2
(«not a run manifest»).

```toml
verb = "INFO"
result2target = true
cache2target = false
stage = "all"          # all | translate | verify
strict = false
llm_config = "llm.toml"
glossary = "glossary.json"

[[targets]]
original = "dialogs/ui.json"
# draft = "drafts/ui.ru.json"   # implies verify; do not set stage here
# llm_config = "profiles/editor-a.toml"
# glossary = "glossaries/matrix.json"
# strict = true

[[targets]]
original = "mechanics.json"
llm_config = "profiles/mechanics.toml"
```

### Precedence B (manifest target)

Job fields come **only** from the manifest: `stage`, `strict`, `llm_config`,
`glossary`, `result2target`, `cache2target`, plus per-target `original` /
`draft` / overrides. Any **passed** job CLI flag with a `.toml` target →
**exit 2** (not quiet ignore). Session flags still apply: `--verb`,
`--no-cache`, `--dry-run`.

Effective settings per target: `target ?? run ?? default`. Paths
(`original`, `draft`, manifest-relative `llm_config` / `glossary`) are
relative to the manifest directory, must not use `..` to escape it, and are
validated for the whole manifest before inference (1-based `[[targets]]`
index in errors). Duplicate `original` after normpath/normcase and
result-path collisions are rejected.

### `draft` (frozen verify input)

Optional `[[targets]].draft` is the only way to feed a frozen translation
into verification. It implies stage `verify` (editor model only). Setting
`stage` on the **same** target → exit 2. Run-level `stage` is unused for that
target (not a conflict). Before any API call, original and draft are matched
by full JSON key path **and** by tag/placeholder multiset per string leaf
(`{N}`, newlines, XML-like tags — same pattern as reply masking). Structure
or marker mismatch → exit 2. Neither file is mutated. There is **no** CLI
`--draft` / `--original`. A lone `--target file.json` cannot take a draft —
use a one-entry `.toml`.

Comparing two editor profiles on the same `original` + `draft` needs **two
run manifests** (or two copies of the input under different relative paths):
duplicate `original` after normpath/normcase is rejected inside one
manifest, so two `[[targets]]` rows cannot share the same source file.

### Migration from `.list`

`.list` manifests are **removed**. Passing `--target something.list` exits 2
with an explicit migration message. Each former non-comment line becomes:

```toml
[[targets]]
original = "path/from/line.json"
```

## Glossary resolution

**JSON target:** `--glossary PATH` (missing → exit 2) → else `glossary.json`
beside the target → else no terms.

**Run manifest:** `glossary` on the target or run (manifest-relative); if
unset everywhere → `glossary.json` beside the manifest, else empty. CLI
`--glossary` is forbidden (precedence B).

A found but invalid file is an error. Source is always logged at INFO.

## Result layout

Without `result2target`: `result/<target-name>/` next to `conductor.toml`.
With `result2target` (CLI for JSON, or manifest field): `result/` next to the
target / manifest.

| Path | When |
|---|---|
| `{result}/translated/{rel}` | After translation |
| `{result}/redacted/{rel}` | After verification |
| `{result}/final/{rel}` | Always (also partial on circuit-breaker abort) |
| `{result}/run_report.json` | Per-leaf stage status + aggregates |
| `{result}/usage.json` | Token attempts by provider / model / stage |
| `{result}/leaves/{rel}` | Translation sidecar used by later verify runs |

Semantics of statuses and usage outcomes:
[pipeline.md](pipeline.md#reports).

`{rel}` is the path relative to the manifest directory (or the file name for
a single target), including its `.json` suffix when present — so a target
`job.json` writes `leaves/job.json` (not `leaves/job.json.json`). Inputs are
never modified. An input that already lies under `{result}` is rejected.
Shared `result/<name>/` is claimed with `.llmc.target` so a same-named
target from another path cannot silently overwrite.

## Exit codes

Only `main()` calls `sys.exit`. Priority when several problems apply:
`1 > 3 > 2 > 4 > 7 > 6 > 5 > 130 > 0`.

| Code | When |
|---|---|
| `0` | Success; failure ratios within threshold |
| `1` | Unexpected exception |
| `2` | Usage / target / manifest / glossary / result ownership |
| `3` | Config invalid or `init` refused overwrite |
| `4` | Preflight API unavailable |
| `5` | Request or file failure ratio above `[resilience].max_failure_ratio`, or any degraded leaf when `[resilience].fail_on_any_error` is true |
| `6` | Consecutive-failure circuit breaker tripped |
| `7` | Target, cache dir, or result root locked by another run |
| `130` | Interrupted (SIGINT/SIGTERM); caches saved, results incomplete |

Unreadable or invalid input JSON, root `null`, draft faults, and verify-only
missing/`null`/structure-mismatched `translated/` fail shared preflight with
usage error `2` before inference. A file that passes preflight but then fails
during processing still counts as a failed file (ratio may yield `5`).

**Warm cache + exit 5:** rejected replies are not cached. A rerun that only
hits residual misses can exceed the failure ratio even if an earlier full
run exited `0`. Fix prompts/model or remeasure with `--no-cache`. See
[pipeline.md](pipeline.md).

## Locks and progress

OS-level non-blocking locks on the target, cache dir (when cache enabled),
and shared result root. Heartbeat JSON: `{cache_dir}/progress_state.json`
(written even with `--no-cache`) for external monitors — report only, not
the process exit authority.
