# RETRACE -- Add %ret Dependent Tier with Verbatim Main-Tier Copy

**Status:** Current
## Purpose

Reimplements CLAN's `retrace` command, which adds a `%ret:` dependent tier to each utterance containing a verbatim serialized copy of the main-tier content (including retrace markers, pauses, events, etc.). This serves as a reference tier preserving the original utterance text before other transforms modify it.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409318) for the original command documentation.

## Usage

```bash
chatter clan retrace file.cha
```

## Options

This command has no configurable options.

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
