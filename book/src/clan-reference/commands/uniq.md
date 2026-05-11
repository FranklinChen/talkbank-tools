# UNIQ -- Report Repeated Lines

**Status:** Current
## Purpose

Identifies and counts duplicate lines (both `@header` and `*speaker` utterance lines, lowercased) across all input files. Matches CLAN behavior of including all line types in the frequency table.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409094) for related CLAN command specifications.

## Usage

```bash
chatter clan uniq file.cha
chatter clan uniq --sort file.cha
chatter clan uniq --format json corpus/
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--sort` | `-o` | Sort output by descending frequency |
| `--speaker <CODE>` | `+t*CHI` | Restrict to specific speaker |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Output

- Table of unique line texts with frequency counts (headers + utterances + dependent tiers)
- Total lines processed and number of unique lines
- Optional frequency-descending sort

## What Gets Counted

UNIQ counts all line types, including:
- `@header` lines
- `*speaker` utterance lines
- `%dependent` tier lines (including `%mor` and `%gra`)
- Multi-line headers are split and counted individually

This matches CLAN's behavior of including dependent tiers in the frequency table and splitting multi-line headers for counting.

## Differences from CLAN

### Dependent tier inclusion

Includes `%mor`/`%gra` dependent tiers in counts, matching CLAN.

### Multi-line header splitting

Splits multi-line headers for counting, matching CLAN.

### Unicode sort order

**1 accepted divergence**: Unicode sort order for `U+230A` (LEFT FLOOR character). C-locale `strcoll()` places this character differently than Rust's byte-order sorting. Result: a single line position swap with identical content and counts. This is a cosmetic difference with no impact on analysis.

### Line text extraction

Uses the parsed AST and `WriteChat` serialization rather than raw text line reading. This ensures consistent normalization.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

99% parity (1 accepted Unicode sort order divergence).
