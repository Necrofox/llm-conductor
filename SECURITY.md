# Security Policy

## Supported versions

The `1.x` line on the default branch and GitHub Releases tagged `v1.*` receive
security fixes.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Email or privately contact the repository owner via GitHub
([Necrofox/llm-conductor](https://github.com/Necrofox/llm-conductor)) using a
private vulnerability advisory if available. Include:

- Affected version / commit
- Description and impact
- Reproduction steps or a proof of concept if possible

You should receive an acknowledgement within a reasonable time. We will
coordinate a fix and disclosure schedule.

## Application notes

- Profiles may store API keys (`[*.api].key`). Prefer `key_env` and keep
  `llm.toml` / `.env` out of version control (see `.gitignore`).
- The tool sends request payloads to configured LLM endpoints (local or
  third-party). Treat those hosts as part of your trust boundary; do not
  point production secrets at untrusted proxies.
- Cache files under `[cache].dir` may contain previously accepted model
  outputs derived from your source strings.
