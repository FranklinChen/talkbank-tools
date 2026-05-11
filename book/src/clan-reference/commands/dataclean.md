# DATACLEAN -- Fix Common CHAT Formatting Errors

**Status:** Current
## Purpose

Reimplements CLAN's DataCleanUp command, which fixes spacing and formatting issues in CHAT files. Because these are text-level formatting concerns that operate below the AST level, the AST transform is a no-op; the actual logic operates on serialized CHAT text via `clean_chat_text()` and the end-to-end `run_dataclean()` function.

## Usage

```bash
chatter clan dataclean file.cha
```

## Options

DATACLEAN has no command-specific flags. It accepts a single positional
input path plus the optional shared `-o`/`--output` from
`CommonAnalysisArgs` (default: stdout). The full set of fixes always
runs together; there is no `--spacing-only` switch — to apply only the
text-spacing repairs you would have to pre-filter the input or
post-edit the output.

## Behavior

The following fixes are applied to non-header lines:

- Missing space before `[` brackets
- Missing space after `]` brackets
- Tab characters inside lines (converted to spaces)
- Bare `...` without `+` prefix (converted to `+...`)
- `#long` converted to `##`
- Header lines (`@`-prefixed) are left untouched

## Differences from CLAN

- Operates on serialized text (post-parse) rather than raw input, since these are formatting concerns below the AST level.
- Uses the framework transform pipeline (parse -> transform -> serialize -> text fixups -> write).
