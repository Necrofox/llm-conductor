# CLI and results

## Commands

```text
llm-conductor init                          # create any missing conductor.toml, llm.toml, context.toml
llm-conductor check                         # strict validate the TOML files that exist
llm-conductor run --target PATH [options]   # pipeline
llm-conductor run --target PATH --dry-run   # preflight plan (no API / result writes)
llm-conductor --version
```

`init` writes a template only when that path is absent. It prints
`Created <path>` or `Skipped <path> (already exists)` and exits 0 when every
file is already present. Existing files are not opened for writing and are
not repaired. The OpenAI key hint is printed only when `llm.toml` is created
in that run. A write error exits 3. `check` validates the files that exist.

From a checkout without install: `python llm-conductor.py …` (same configs
layout as the frozen exe). After `pip install`: `llm-conductor …`.

`--version` reports the version generated at build/install time. A fresh
checkout without generated version metadata reports `0.0.0+unknown`.
See [CONTRIBUTING.md](../CONTRIBUTING.md#versions-and-builds) for dev builds.

### `run` options

| Flag | Meaning |
|---|---|
| `--target PATH` | Required. `.json` file or `.toml` run manifest |
| `--stage {all,translate,verify}` | Default `all` (JSON targets only) |
| `--strict` | Translation in strict mode (JSON targets only) |
| `--no-cache` | Disable persistent cache for this run |
| `--dry-run` | Validate inputs and print effective settings; no API, locks, or result writes (session flag) |
| `--preview-prompts` | Print each shown leaf: source file, structural path, operation, final hints with origin, and the provider JSON payload. With `--dry-run`, print only. Without it, the run still calls the API (session flag) |
| `--preview-path JSON` | Exact structural path as a JSON array (`["flat.key"]`, `["flat","key"]`, `["items",0,"name"]`). Repeatable. Filters the preview only (session flag) |
| `--preview-limit N` | Non-negative maximum number of preview requests across all files and stages. Default: unlimited; `0` emits none. Does not limit processing or API calls (session flag) |
| `--preview-output PATH` | Write the same preview as JSONL. Does not by itself print to stdout (session flag) |
| `--llm-config PATH` | Alternate profile, cwd-relative (JSON targets only) |
| `--context PATH` | Key-context TOML (else `context.toml` beside the target, else beside the program, else none; JSON targets only) |
| `--glossary PATH` | Glossary JSON (else `glossary.json` beside target, else empty; JSON targets only) |
| `--verb LEVEL` | Log level override (session flag; also overrides manifest `verb`) |
| `--result2target` | Write `result/` next to the target (JSON targets only) |
| `--cache2target` | Put cache dir next to the target (JSON targets only) |

`check` validates TOML only (not manifests, locks, or live API).
`run --dry-run` is the target/manifest audit: same settings resolver as a real
`run`, exit `0` on a clean plan, exit `2` on contract/path/JSON/glossary/draft
errors, exit `3` when a resolved profile/config fails to load (same as a real
`run`). On success it prints an effective-settings table (index, original,
draft, stage, strict, llm_config, glossary, context, result paths).
`--preview-prompts` with `--dry-run` then prints the provider JSON payload of
every shown leaf and still makes no API call and writes no results.
`--preview-path` and `--preview-limit` choose which of those requests are
shown; they do not change the set of processed resources, the number of real
calls, or input preflight. Using either without `--preview-prompts` or
`--preview-output` is exit 2. `--preview-output` writes one JSON object per
shown request and does not print that preview to stdout unless
`--preview-prompts` is also set.
During dry-run, duplicate object keys in originals and drafts emit one WARNING per file
(last value wins) and do **not** fail the plan. Dry-run does not acquire locks
or report lock contention. There is no `plan` / `inspect` command and no
`dry_run` field in the manifest (unknown key → exit 2).

### Dry-run vs real-run failure modes

Both modes check any existing result owner marker and run the same input
preflight. Dry-run reads the marker without claiming ownership. A real run
takes its locks first, then validates the inputs before claiming an unowned
result directory or probing the API. A real run can create lock files even
when input preflight fails. Dry-run makes no model calls, takes no locks, and
writes no results.

| Fault | Stages | Dry-run | Real run |
|---|---|---|---|
| Manifest / path / glossary / context / result-ownership contract | any | exit `2` | exit `2` |
| Unreadable or invalid original JSON | any | exit `2` | exit `2` |
| Root JSON `null` (original) | any | exit `2` | exit `2` |
| Draft missing / unreadable / structure or marker mismatch | verify via `draft` | exit `2` | exit `2` |
| Verify-only without `draft`: missing / unreadable / null / structure-mismatched `translated/` | `verify` | exit `2` | exit `2` |
| Profile / conductor load failure | any | exit `3` | exit `3` |
| Duplicate object keys in an original or draft | any | WARNING; exit `0` | Last value wins; continues |
| API unavailable | stages that call the model | not checked | exit `4` |
| Target / cache / result lock held | any | not checked | exit `7` |
| Leaf degradation / request or file failure ratio | processing | not applicable | exit `5` when a failure ratio exceeds its threshold or `fail_on_any_error` is enabled and a leaf is degraded (exit `6` on circuit breaker) |
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
context = "context.toml"

[[targets]]
original = "dialogs/ui.json"
# draft = "drafts/ui.ru.json"   # implies verify; do not set stage here
# llm_config = "profiles/editor-a.toml"
# glossary = "glossaries/matrix.json"
# context = "contexts/dialog.toml"
# strict = true

[[targets]]
original = "mechanics.json"
llm_config = "profiles/mechanics.toml"
```

### Precedence B (manifest target)

Job fields come **only** from the manifest: `stage`, `strict`, `llm_config`,
`glossary`, `context`, `result2target`, `cache2target`, plus per-target `original` /
`draft` / overrides. Any **passed** job CLI flag with a `.toml` target →
**exit 2** (not quiet ignore). Session flags still apply: `--verb`,
`--no-cache`, `--dry-run`, `--preview-prompts`, `--preview-path`,
`--preview-limit`, `--preview-output`.

Effective settings per target: `target ?? run ?? default`. Paths
(`original`, `draft`, manifest-relative `llm_config` / `glossary` / `context`) are
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
(selected by `[processing].mask_pattern` in the target's effective `llm_config`
profile, the same rule as reply masking;
defaults protect `{N}`, newlines, XML-like tags). See [content.md](content.md)
for the contract and recipes. Structure
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

## Key-context resolution

**JSON target:** `--context PATH` (missing → exit 2) → else `context.toml`
beside the target → else `context.toml` beside the program (`init`) → else
no key context.

**Run manifest:** `context` on the target or run (manifest-relative); if
unset everywhere → `context.toml` beside the manifest when that file exists,
else the program file. CLI `--context` is forbidden (precedence B).

A found but invalid file exits 2. The selected file is the whole catalog:
it is not merged with the program file, and omitted `[[rules]]` are not
filled back in. Source is logged at INFO. `check` validates the program
file when it exists (invalid → exit 3); absence is success.

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
`job.json` writes `leaves/job.json` (not `leaves/job.json.json`). In
machine-readable reports the same `{rel}` is always written with POSIX
`/` separators (see [pipeline.md](pipeline.md#reports)); the on-disk layout
still follows the host OS. Inputs are never modified. An input that already
lies under `{result}` is rejected. Shared `result/<name>/` is claimed with
`.llmc.target` so a same-named target from another path cannot silently
overwrite.

Result-relative paths must be non-empty and contain no root, drive prefix,
or `.` / `..` components. Backslashes are treated as separators on every
platform, including in POSIX single-file names. A target name that violates
these rules exits `2` before processing. Manifest paths may still use inner
`..` segments when they resolve inside the manifest directory; their result
paths are derived after normalization.

## Exit codes

Only `main()` calls `sys.exit`. When a run records multiple exit codes, it
uses this priority order.

`1 > 3 > 2 > 4 > 7 > 6 > 5 > 130 > 0`

| Code | When |
|---|---|
| `0` | Success; failure ratios within threshold |
| `1` | Unexpected exception |
| `2` | Usage / target / manifest / glossary / result ownership |
| `3` | Config invalid, or `init` could not write a template |
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

Read `last_progress_at`, `progress.leaves_done`, and `activity` for work
done. `updated_at` and `elapsed_sec` also move on
`[heartbeat].interval_sec` while a request or retry is waiting
(`activity` is `waiting_api`) and do not mean another leaf finished.
`last_progress_at` stays `null` until the first completed leaf.
`schema_version` is still 1. `status` ends as `completed` (including a
failed run, with `exit_code`) or `interrupted`, and the file is not
rewritten after that.


## Prompt preview accuracy

```bash
llm-conductor run --target strings.json --stage translate --dry-run --preview-prompts
llm-conductor run --target review.toml --dry-run --preview-prompts \
  --preview-path '["flat.key"]' --preview-limit 20 --preview-output preview.jsonl
```

Preview and HTTP submission use the same payload builder, including masked
leaf text, system message, context, glossary and provider-specific sampling.
Each shown request names the source file, the operation, and the structural
path as a JSON array, so a flat key `["flat.key"]` is distinct from a nested
path `["flat","key"]` and from an array index. Final hints are listed per
slot with where that text won: `exact` (a `[[leaves]]` row, named `leaf-<index>`),
`rule` (a local `[[rules]]` entry or registered detector, by name), or
`inherited` (the same rule on an ancestor, plus that parent's structural path).
The listing explains the hint that was already chosen; it does not change
which text wins a slot. Authentication headers are never printed. Preview
does not contact or probe the endpoint. File-processing errors return a
nonzero status (currently exit 1).

`--preview-output` writes the same records as JSONL. Each line has `source`,
`path` (segment array), `operation`, `hints` (`slot`, `text`, `origin`,
`rule`, and `parent` when inherited), `payload`, `preliminary`, and `note`.
A manifest run puts the source file's portable relative path in `source`.
The destination is relative to the working directory. An existing export is
replaced, not appended to; no matches or `--preview-limit 0` produces an empty
file. Parent directories are created when the export is written. Export paths
that alias an original, draft, manifest, or loaded configuration/glossary/context
file, or lie inside this run's result, cache, or log directories, are rejected
with exit 2 before any API calls. This also applies to dry-run. Write failures
return a nonzero status and a real run's heartbeat records its final status.

With `stage = "all"`, offline preview cannot know the future translation:
the verification payload uses the original string as a placeholder, the log
emits a warning, and each verification record sets `preliminary` with a note
that the future translation is not yet known and the editor request is
preliminary. Use verify with an existing translated result or a manifest
`draft` for an exact verification payload. On a real run, verification
previews use the actual translation and are not marked preliminary. Previews
are printed before cache/dedup lookup, so a printed candidate payload does
not imply that an HTTP call will be needed. A path filter or limit stops
only this output.
