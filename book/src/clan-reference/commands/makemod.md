# MAKEMOD -- Generate %mod Tier from Pronunciation Lexicon

**Status:** Current
**Last updated:** 2026-05-22 13:12 EDT

## Purpose

Reimplements CLAN's MAKEMOD command, which looks up each countable word on main tiers in a pronunciation lexicon (CMU dictionary format) and generates a `%mod` dependent tier with the phonemic transcription. Words not found in the lexicon are marked with `???`.

## Usage

```bash
chatter clan makemod --lexicon cmulex.cut file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-l`, `--lexicon` | path | *(required)* | Path to the pronunciation lexicon file |
| `--all-alternatives` | bool | `false` | Show all alternative pronunciations (default: first only) |
| `-o`, `--output` | path | stdout | Output CHAT file path |

## CLAN `+`-flag coverage audit

MAKEMOD is a **transform**. Sources:
`OSX-CLAN/src/clan/makemod.cpp::usage`,
`crates/talkbank-clan/src/transforms/makemod.rs`.

### MAKEMOD-specific `+`-flags (from `makemod.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Print out all alternative pronunciations (default: first) | `--all-alternatives` | Done | Direct mapping; rewriter routes `+a` → `--all-alternatives` (since 2026-05-22). |

Audit summary: 2 Done (default first-only + `+a` all-alts).
0 Missing. Clean parity.

## External Data

Requires a CMU-format lexicon file. CLAN ships a `cmulex.cut` in
its `lib/` directory and uses that by default; `chatter clan
makemod` does not bundle a lexicon, so `--lexicon` must be passed
explicitly.

Format: `WORD  phoneme1 phoneme2 ...` (one entry per line). Lines starting with `#` or `%` are treated as comments. Words with `(N)` suffix (variant number like `READ(2)`) are treated as pronunciation alternatives for the base word.

## Behavior

For each utterance, the transform:

1. Extracts countable words from the main tier (using the framework's `countable_words()` utility).
2. Looks up each word in the loaded pronunciation lexicon (case-insensitive).
3. Builds a `%mod` dependent tier with the phonemic transcriptions. Words not found are marked `???`.
4. Appends the `%mod` tier to the utterance's dependent tiers.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
