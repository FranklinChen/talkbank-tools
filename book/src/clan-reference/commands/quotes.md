# QUOTES -- Extract Quoted Text to Separate Utterances

**Status:** Current
**Last updated:** 2026-05-22 12:52 EDT

## Purpose

Reimplements CLAN's QUOTES command.

This is a relatively uncommon command used for discourse analysis of reported speech.

## Usage

```bash
chatter clan quotes file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

QUOTES is a **transform**. Sources:
`OSX-CLAN/src/clan/quotes.cpp::usage`,
`crates/talkbank-clan/src/transforms/quotes.rs`.

CLAN's `quotes.cpp::usage` lists **no command-specific `+`-flags**
— the surface is the inherited general flag set only (none of
which apply to a pure transform). chatter's QUOTES surface is
likewise minimal: input CHAT file in, transformed CHAT out.

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default extract-quoted behaviour) |
| Missing | 0 |

QUOTES is one of the cleanest parity stories in the catalog —
both CLAN and chatter expose the identical "do the obvious
thing" surface and produce comparable output. No follow-ups
filed.

## Behavior

The Rust port now inspects the parsed CHAT AST directly.

- If no quote-extraction postcode (`[+ "]`) is present, the command is a semantic no-op and emits normalized CHAT.
- If `[+ "]` is present, the command fails with an explicit error. `talkbank-clan` does not silently fall back to post-serialization string manipulation for this transform.

## Differences from CLAN

- Does not currently implement CLAN's text-level extraction rewrite for `[+ "]`.
- Fails explicitly on unsupported quote-extraction postcodes instead of attempting a lossy raw-text rewrite.
