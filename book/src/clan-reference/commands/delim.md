# DELIM -- Add Missing Utterance Terminators

**Status:** Current
## Purpose

Reimplements CLAN's `delim` command, which ensures every main tier has a terminator. Utterances missing a terminator (`.`, `?`, `!`) receive a default period (`.`). This is typically used as a repair step for files imported from external formats that lack CHAT punctuation conventions.

## Usage

```bash
chatter clan delim file.cha
```

## Options

This command has no configurable options.

## Behavior

For each utterance in the file, if the main tier lacks a terminator, a period (`.`) is inserted as the default terminator.

Utterances that already have a terminator (`.`, `?`, or `!`) are left unchanged.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
- **4 accepted divergences**: CLAN writes an empty file when no changes are needed; we always write the full file. This is intentional -- the output is always a valid CHAT file.
- **Golden test parity**: 4 accepted divergences (empty-file behavior).
