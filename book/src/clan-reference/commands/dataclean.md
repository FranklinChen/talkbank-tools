# DATACLEAN -- Fix Common CHAT Formatting Errors

**Status:** Current
**Last updated:** 2026-05-22 13:01 EDT

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

## CLAN `+`-flag coverage audit

DATACLEAN is a **transform**. Sources:
`OSX-CLAN/src/clan/DataCleanUp.cpp::usage`,
`crates/talkbank-clan/src/transforms/dataclean.rs`.

### DATACLEAN-specific `+`-flags (from `DataCleanUp.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b` | Work on the BOLD problem only | — | Missing | CLAN isolates one fix-class at a time; chatter always runs the full set. |
| `+c` | Work on the `++`/`+,` problem only | — | Missing | |
| `+o` | Work on all other problems only | — | Missing | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default full-set) |
| Missing | 3 |

DATACLEAN's `+b`/`+c`/`+o` partition the fix set into three
classes that CLAN lets the user run in isolation. chatter
collapses them into "run everything"; partitioning would require
splitting the fix passes into named groups with per-flag gating.
Filed as a Phase 1.7 follow-up; not load-bearing for typical use.

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
