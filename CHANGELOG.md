# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
