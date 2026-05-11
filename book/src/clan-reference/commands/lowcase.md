# LOWCASE -- Lowercase All Words on Main Tiers

**Status:** Current
## Purpose

Reimplements CLAN's `lowcase` command, which converts all words on main tiers to lowercase. Speaker codes, headers, and dependent tiers are preserved unchanged. The transformation recurses into annotated words, replaced words, groups, and annotated groups.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409329) for the original command documentation.

## Usage

```bash
chatter clan lowcase file.cha
```

## Options

This command has no configurable options.

## Behavior

For each utterance, the transform lowercases all word surfaces on the main tier. The transformation recurses into:

- Plain words
- Annotated words (inner word)
- Replaced words (both original and replacement forms)
- Groups (bracketed content)
- Annotated groups (inner bracketed content)

Speaker codes, header lines, and dependent tiers are not modified.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
