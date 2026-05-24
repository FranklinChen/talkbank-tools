# RETRACE -- Add %ret Dependent Tier with Verbatim Main-Tier Copy

**Status:** Current
**Last updated:** 2026-05-22 12:57 EDT

## Purpose

Reimplements CLAN's `retrace` command, which adds a `%ret:` dependent tier to each utterance containing a verbatim serialized copy of the main-tier content (including retrace markers, pauses, events, etc.). This serves as a reference tier preserving the original utterance text before other transforms modify it.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409318) for the original command documentation.

## Usage

```bash
chatter clan retrace file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

RETRACE is a **transform**. Sources:
`OSX-CLAN/src/clan/retrace.cpp::usage`,
`crates/talkbank-clan/src/transforms/retrace.rs`.

### RETRACE-specific `+`-flags (from `retrace.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+c` | Substitute the `%ret:` line for the main line in output | — | Missing | chatter always emits both. A `--substitute-main` boolean would close this. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default emit-both) |
| Missing | 1 |

RETRACE's `+c` is a one-line config addition (substitute-main
boolean) plus a small render-time choice. Filed as a Phase 1.7
follow-up.

## Behavior

For each utterance, the transform:

1. Serializes the main tier to its full CHAT text representation.
2. Extracts the content portion (after `*SPEAKER:\t`).
3. Creates a `%ret:` user-defined dependent tier containing the verbatim content.
4. Inserts the `%ret:` tier at position 0 (before other dependent tiers).

All headers are preserved. Existing dependent tiers are kept.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
