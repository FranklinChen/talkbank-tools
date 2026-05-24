# LOWCASE -- Lowercase All Words on Main Tiers

**Status:** Current
**Last updated:** 2026-05-22 12:51 EDT

## Purpose

Reimplements CLAN's `lowcase` command, which converts all words on main tiers to lowercase. Speaker codes, headers, and dependent tiers are preserved unchanged. The transformation recurses into annotated words, replaced words, groups, and annotated groups.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409329) for the original command documentation.

## Usage

```bash
chatter clan lowcase file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

LOWCASE is a **transform**. Sources:
`OSX-CLAN/src/clan/lowcase.cpp::usage`,
`crates/talkbank-clan/src/transforms/lowcase.rs`.

### LOWCASE-specific `+`-flags (from `lowcase.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+c` | Lowercase only the first word on a tier | — | Missing | First-word-only mode. |
| `+d` | Do NOT change words listed in `iF` dict file; lower-case the rest | — | Missing | Dictionary-guarded lowercasing — typically a proper-nouns / pronouns list. |
| `+d1` | Capitalize words in dict file; leave the rest unchanged | — | Missing | Inverse mode (capitalization, not lowercasing). |
| `+d2` | Ignore dict file; lower-case everything | (default) | Done | chatter's default matches this — every word on main tier lowercased. |
| `+iF` | Dictionary file `F` with words to NOT lowercase | — | Missing | Pairs with `+d`. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 |
| Partial | 0 |
| Missing | 4 |

LOWCASE's largest gap is the **proper-noun/pronoun preservation
dictionary** (`+iF`+`+d`): CLAN ships a `lowcase.cut` dictionary
that lists words to *keep* capitalised (proper nouns, the pronoun
`I`, etc.), so the typical CLAN invocation preserves "I" and
"Mary" while lowercasing the rest. chatter's `lowcase` lowercases
*everything* unconditionally, which can incorrectly merge the
pronoun "I" with the article "i" (rare but real). Filed as the
top Phase 1.7 follow-up for this command.

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
