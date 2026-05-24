# CORELEX — Core Vocabulary

**Status:** Current
**Last updated:** 2026-05-22 09:47 EDT

## Purpose

Identifies "core" vocabulary items that appear above a frequency threshold. Core vocabulary analysis is used in clinical assessment to evaluate whether a child's lexicon includes expected high-frequency words.

## Usage

```bash
chatter clan corelex file.cha
chatter clan corelex --speaker CHI file.cha
chatter clan corelex --threshold 5 file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--threshold <N>` | — | Minimum frequency for core classification (default: 3) — chatter extension |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

### CORELEX-specific `+`-flags (from `corelex.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `-g` | Do not look for GEMs | — | Missing | Disables gem-filtering. |
| `+lF` | Specify words group name `F` (with `.cut` extension; abbreviation rules around `-`) | — | Missing | CORELEX's primary input. chatter operates on the file's own vocabulary, not a curated group list. |
| `+n` / `-n` | Gem termination semantics | — | Missing | |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | |
| `+w` | Find first-time values for words from `+l` group | — | Missing | First-appearance tracking. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 1 |
| Rewriter only | 4 |
| Missing | 8 |

CORELEX's biggest gap is `+lF`: CLAN's CORELEX is fundamentally a
**curated-word-list comparison** tool (compare a transcript's
vocabulary against a clinical or research word group). chatter's
CORELEX uses the file's own vocabulary, which is a different
question. Filed as a high-priority Phase 1.7 follow-up.

## Output

- Core word list (frequency >= threshold) sorted by frequency descending
- Non-core word list
- Core/total ratio and percentage
- Per-word speaker count (how many speakers used each word)

## Differences from CLAN

- Word identification uses AST-based `is_countable_word()`.
- Output supports text, JSON, and CSV formats.
- Core/non-core classification uses shared `NormalizedWord` for consistency.
