# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-11-09

### Added

- Per-leaf run report (`run_report.json`), result-root `leaves/` sidecars,
  and banner counts for untranslated / degraded / verify_skipped (#6)
- Skip normal verification for `translation_degraded` leaves; status
  survives standalone `--stage verify` via sidecars (#11)
- Token usage export (`usage.json`) by provider / model / stage, including
  retries, preflight, cache/dedup non-API counts, and `success` /
  `rejected` / `failed` outcomes (#15)
- Opt-in `[resilience].fail_on_any_error` — exit `5` on any degraded leaf

### Changed

- README doc links use absolute GitHub URLs so the PyPI long description
  resolves correctly; `doc/cli.md` lists report artifacts in the result
  layout

## [1.0.1] - 2026-11-09

### Fixed

- Preserve leaf boundary whitespace and skip format-only leaves (#10)
- Include generation-affecting sampling parameters in the cache key;
  bump `CACHE_KEY_VERSION` to 2 (#12)

### Changed

- Enrich `tag_marker_mismatch` logs with missing/extra/duplicated detail;
  opt-in `[validation].save_rejected_replies` (#7)

## [1.0.0] - 2026-11-08

First public release.

### Added

- Installable package `llm-conductor` with console script and frozen Windows
  `.exe` release workflow
- Two-stage JSON localization pipeline (translate + verify) over Ollama or
  OpenAI-compatible APIs
- Strict TOML configuration (`conductor.toml` + `llm.toml`), `init` / `check` /
  `run` CLI
- Linguistic key hints, glossary, tag masking, disk cache, in-run dedup
- Reply validation, consecutive-failure circuit breaker, OS file locks,
  progress heartbeat
- English human documentation under `doc/`, `AGENTS.md`, examples, MIT license

### Notes

- Language pair is defined only by prompts in the model profile — not by an
  engine setting
- User-facing CLI and log messages are English
