# CLI and results

## Commands

```text
llm-conductor init                          # write conductor.toml + llm.toml (no overwrite)
llm-conductor check                         # strict validate both TOML files
llm-conductor run --target PATH [options]   # pipeline
llm-conductor --version
```

From a checkout without install: `python llm-conductor.py …` (same configs
layout as the frozen exe). After `pip install`: `llm-conductor …`.

### `run` options

| Flag | Meaning |
|---|---|
| `--target PATH` | Required. `.json` file or `.list` manifest |
| `--stage {all,translate,verify}` | Default `all` |
| `--strict` | Translation in strict mode |
| `--no-cache` | Disable persistent cache for this run |
| `--llm-config PATH` | Alternate profile (cwd-relative) |
| `--glossary PATH` | Glossary JSON (else `glossary.json` beside target, else empty) |
| `--verb LEVEL` | Log level override |
| `--result2target` | Write `result/` next to the target |
| `--cache2target` | Put cache dir next to the target |

`check` validates TOML only (not manifests, locks, or live API).

## Manifest (`.list`)

One path per line, relative to the manifest directory, must stay inside it.
Blank lines and `#` comments skipped. Absolute paths, `..`, drive prefixes,
the directory itself, and duplicates after normalization are errors **before**
processing (exit 2). A missing file listed in the manifest fails that file
only; others continue.

## Glossary resolution

`--glossary PATH` (missing → exit 2) → else `glossary.json` beside the
target → else no terms. Not read from cwd. A found but invalid file is an
error. Source is always logged at INFO.

## Result layout

Without `--result2target`: `result/<target-name>/` next to `conductor.toml`.
With `--result2target`: `result/` next to the target.

| Path | When |
|---|---|
| `{result}/translated/{rel}` | After translation |
| `{result}/redacted/{rel}` | After verification |
| `{result}/final/{rel}` | Always (also partial on circuit-breaker abort) |

`{rel}` is the path relative to the manifest directory (or the file name for
a single target). Inputs are never modified. An input that already lies under
`{result}` is rejected. Shared `result/<name>/` is claimed with
`.llmc.target` so a same-named target from another path cannot silently
overwrite.

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
| `5` | Request or file failure ratio above `[resilience].max_failure_ratio` |
| `6` | Consecutive-failure circuit breaker tripped |
| `7` | Target, cache dir, or result root locked by another run |
| `130` | Interrupted (SIGINT/SIGTERM); caches saved, results incomplete |

A single corrupt JSON file counts as a failed file (ratio may yield `5`),
not usage error `2`.

**Warm cache + exit 5:** rejected replies are not cached. A rerun that only
hits residual misses can exceed the failure ratio even if an earlier full
run exited `0`. Fix prompts/model or remeasure with `--no-cache`. See
[pipeline.md](pipeline.md).

## Locks and progress

OS-level non-blocking locks on the target, cache dir (when cache enabled),
and shared result root. Heartbeat JSON: `{cache_dir}/progress_state.json`
(written even with `--no-cache`) for external monitors — report only, not
the process exit authority.
