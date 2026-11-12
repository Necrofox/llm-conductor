# String content rules

`llm.toml` controls which string leaves are eligible for processing and
which fragments the model must preserve. Each manifest target uses its own
effective `llm_config` profile, including defaults for omitted settings; rules
are never inherited from the first target. These rules apply to translation,
verification, and external-draft marker checks. They do not change JSON keys,
non-string values, or the prompt-defined language pair. Outputs remain strings,
including numeric-looking replies such as `"42"`.

## Contract

- Missing settings use the defaults below. An empty `required_content_pattern`
  admits every nonblank string; an empty `mask_pattern` disables masking.
  A custom mask **replaces** the default; include every format you need.
- Empty and whitespace-only leaves are always preserved exactly, without a
  cache lookup or model call. Leading/trailing whitespace is owned by the
  client: it is peeled before a call and restored on acceptance.
- For eligibility only, replace each non-overlapping protected match in the
  peeled string with **one ASCII space**, then search for the required pattern.
  No match means preserve the entire original value, recorded as `skipped`.
  A match admits the normal cache/dedup/model path; it does not force an API call.
- This eligibility view never changes the actual input. `Op{0}en` becomes
  `Op en`, not `Open`; `Open {0} door` still matches `Open.*door`.
- Masking uses each complete, nonempty regex match, regardless of capture
  groups. Matches do not overlap; zero-length matches protect nothing. Each
  occurrence gets its own collision-free marker, even for identical fragments.
  Input matching uses the peeled core, including in the verification reference
  and external-draft comparison: `^` and `$` see the core's boundaries.
- With reply validation enabled, every generated marker must occur exactly
  once. Protected fragments must have the same values and counts after
  restoration; their order may change. Literal protected syntax invented by
  the model is rejected. This is fragment preservation, not XML validation.
- Newlines inside the string follow the mask rule too. The default protects
  LF (`\n`); a literal protected newline in the raw reply is rejected. If a
  custom mask leaves newlines unprotected, they can be changed or introduced.
  Boundary whitespace is still preserved. Real LF and the two characters
  backslash + `n` are different; CRLF can be protected together with `\r?\n`.
- Eligibility is evaluated on the current stage input (the draft on verify).
  Existing stale-source and translation-failure/missing gates still apply;
  changing a regex cannot bypass them. External drafts must preserve fragments
  selected by the current mask, even when a leaf would skip inference.
- Both pattern strings participate in cache/dedup identity. Changing either
  cannot reuse a result accepted under another content policy. Older cache-key
  versions are discarded. Sidecar leaf identity and schema are unchanged.

Patterns use the Python `regex` package with VERSION0 semantics. Inline flags
such as `(?i)` and `(?x)` are supported; switching to VERSION1 is not.
`\p{L}` means Unicode letters, rather than just
Latin/Cyrillic; `\p{Nd}` means decimal digits. Invalid syntax is a configuration
error detected before processing. Use TOML **literal strings** (single quotes),
so regex backslashes do not need doubling. In multiline literal strings,
line breaks are part of the expression unless `(?x)` is used.

## Copyable recipes

Each block below is a valid, independent `llm.toml` fragment. Omitted
settings retain their defaults. Combine alternatives with `|`; put longer,
more specific alternatives first when they overlap.

### Letters only (default)

```toml
[processing]
required_content_pattern = '\p{L}'
mask_pattern = '(?:\{\d+\})|(?:\n)|(?:<[^>]+>)'
```

`Hello`, `Привет`, `你好`, and `<item>Hello {0}</item>` are eligible.
` +0042 `, `-007`, `2026-11-09`, `<item id="abc">42</item>`, `{0}`, and
`<item/>` are preserved without inference. Letters inside tags do not count.

The default tag matcher keeps the earlier, simple behavior: it stops at the
first `>`, including one inside a quoted attribute. Named placeholders are
not included, and the newline alternative protects LF rather than the whole
CRLF pair. Use the HTML/XML or combined recipe below when you need these
additional formats protected.

### Letters or decimal digits

```toml
[processing]
required_content_pattern = '[\p{L}\p{Nd}]'
```

`+0042`, `-007`, and `2026-11-09` become eligible; `{0}` and `...` still skip.
An input `" +0042 "` and accepted reply `"42"` produce the string `" 42 "`.

### Every nonblank string

```toml
[processing]
required_content_pattern = ''
```

Numbers, punctuation, emoji, and placeholder-only leaves are eligible.
`""`, spaces, tabs, and newline-only leaves still skip. Default masking remains
active: `{0}` is sent as a marker that must be returned.

### No masking

```toml
[processing]
mask_pattern = ''
```

`<item/>` now contains eligible letters and is sent literally. The model may
change tags/placeholders; they no longer receive structural protection.
Literal text such as `[T0]` is ordinary content when masking is disabled.

### HTML and XML tags

```toml
[processing]
mask_pattern = '''</?[\p{L}_:][\w:.-]*(?:\s+[^<>"']+|"[^"]*"|'[^']*')*\s*/?>'''
```

Protects `<item>`, `</item>`, `<item/>`, `<item id="42">`, `<ns:item>`,
`<br />`, Unicode names such as `<элемент>`, and tags with `>` inside quoted
attribute values. Text between tags
remains available for processing. This recipe does not include placeholders,
newlines, comments, CDATA, declarations, or processing instructions. It is a
tag matcher, not a parser for complete XML/HTML documents.

### Positional and named brace placeholders

```toml
[processing]
mask_pattern = '\{(?:\d+|[A-Za-z_][A-Za-z0-9_]*)\}'
```

Protects `{0}`, `{42}`, `{name}`, and `{player_name}`. `Hello {name}` processes
the greeting; `{player_name}` alone skips under the default letters rule.
Nested ICU messages and format specifications are not covered.

### Double-brace placeholders

```toml
[processing]
mask_pattern = '\{\{\s*[A-Za-z_][A-Za-z0-9_.]*\s*\}\}'
```

Protects `{{name}}`, `{{ user.name }}`. Full template expressions are not covered.

### Dollar-brace placeholders

```toml
[processing]
mask_pattern = '\$\{[A-Za-z_][A-Za-z0-9_.]*\}'
```

Protects `${name}` and `${user.name}`.

### Printf placeholders

```toml
[processing]
mask_pattern = '%(?:%|(?:\([A-Za-z_][A-Za-z0-9_]*\)|\d+\$)?[-+ #0]*\d*(?:\.\d+)?[diouxXeEfFgGcs])'
```

Protects `%s`, `%d`, `%02d`, `%.2f`, `%1$s`, `%(name)s`, and `%%`.
Star width/precision, length modifiers, and other dialect-specific extensions
are not covered.

### Combined localization recipe

```toml
[processing]
required_content_pattern = '\p{L}'
mask_pattern = '''(?x)
    </?[\p{L}_:][\w:.-]*(?:\s+[^<>"']+|"[^"]*"|'[^']*')*\s*/?>
  | \{\{\s*[A-Za-z_][A-Za-z0-9_.]*\s*\}\}
  | \$\{[A-Za-z_][A-Za-z0-9_.]*\}
  | \{(?:\d+|[A-Za-z_][A-Za-z0-9_]*)\}
  | %(?:%|(?:\([A-Za-z_][A-Za-z0-9_]*\)|\d+\$)?[-+\x20\#0]*\d*(?:\.\d+)?[diouxXeEfFgGcs])
  | \r?\n
'''
```

Protects tags, all placeholder families above, and LF/CRLF in a single rule.
To also normalize numeric leaves, change only `required_content_pattern` to
`'[\p{L}\p{Nd}]'`, or `''` for all nonblank leaves.

See [configuration.md](configuration.md) for configuration placement and
[pipeline.md](pipeline.md) for validation, caching, and verification gates.
