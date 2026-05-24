# DATES -- Age Computation from Birth and Date Headers

**Status:** Current
**Last updated:** 2026-05-22 13:02 EDT

## Purpose

Reimplements CLAN's `dates` command, which computes the age of each participant at the time of transcription by subtracting `@Birth` dates from the file-level `@Date` header. Computed ages are inserted as `@Comment: Age of CHI is Y;M.D` headers after the `@ID`/`@Birth` block.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409311) for the original command documentation.

The computed age uses CHAT format: `years;months.days` (e.g., `2;6.15` for two years, six months, and fifteen days).

## Usage

```bash
chatter clan dates file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

DATES is a **transform**. Sources:
`OSX-CLAN/src/clan/dates.cpp::usage`,
`crates/talkbank-clan/src/transforms/dates.rs`.

### DATES-specific `+`-flags (from `dates.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+aC S` | Specify age of participant `C` (default CHI) as `Y;M.D` | — | Missing | Overrides the computed age with a literal value. |
| `+bC S` | Specify birth of participant `C` as `12-JAN-1962` or `01/12/62` | — | Missing | Overrides the `@Birth` header for the participant. |
| `+d S` | Specify date of transcript as `12-JAN-1962` or `01/12/62` | — | Missing | Overrides the `@Date` header. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (compute-from-headers) |
| Missing | 3 |

DATES's three `+`-flags are command-line *overrides* for the
file's `@Birth`/`@Date` headers — useful when the file is missing
those headers or when the user wants to override them for a one-
off computation. chatter operates strictly on the parsed
headers; adding a `--age` / `--birth` / `--date` override family
would close the gap. Filed as a Phase 1.7 follow-up.

## Behavior

1. Collects the `@Date` header value and all `@Birth` headers with their participant codes.
2. Computes each participant's age by subtracting the birth date from the file date.
3. Inserts `@Comment: Age of <PARTICIPANT> is <age>` headers into the file.
4. If no `@Date` header is present, the file is left unchanged.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
