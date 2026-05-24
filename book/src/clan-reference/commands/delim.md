# DELIM -- Add Missing Utterance Terminators

**Status:** Current
**Last updated:** 2026-05-22 12:51 EDT

## Purpose

Reimplements CLAN's `delim` command, which ensures every main tier has a terminator. Utterances missing a terminator (`.`, `?`, `!`) receive a default period (`.`). This is typically used as a repair step for files imported from external formats that lack CHAT punctuation conventions.

## Usage

```bash
chatter clan delim file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

DELIM is a **transform**. Sources:
`OSX-CLAN/src/clan/delim.cpp::usage`,
`crates/talkbank-clan/src/transforms/delim.rs`.

### DELIM-specific `+`-flags (from `delim.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Change default terminator from `.` to `++.` | — | Missing | Inserts an "interrupted" marker. chatter only inserts `.` today. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default terminator insertion) |
| Missing | 1 |

DELIM's surface is genuinely minimal. The `+a` interrupted-
terminator option would be a one-line config addition: a
`Terminator` choice on `DelimConfig`. Filed as a Phase 1.7
follow-up.

## Behavior

For each utterance in the file, if the main tier lacks a terminator, a period (`.`) is inserted as the default terminator.

Utterances that already have a terminator (`.`, `?`, or `!`) are left unchanged.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
- **4 accepted divergences**: CLAN writes an empty file when no changes are needed; we always write the full file. This is intentional -- the output is always a valid CHAT file.
- **Golden test parity**: 4 accepted divergences (empty-file behavior).
