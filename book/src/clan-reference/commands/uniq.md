# UNIQ -- Report Repeated Lines

**Status:** Current
**Last updated:** 2026-05-23 13:00 EDT

## Purpose

Identifies and counts duplicate lines (both `@header` and `*speaker` utterance lines, lowercased) across all input files. Matches CLAN behavior of including all line types in the frequency table.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409094) for related CLAN command specifications.

## Usage

```bash
chatter clan uniq file.cha
chatter clan uniq --sort file.cha
chatter clan uniq --format json corpus/
```

## Options (chatter-native)

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--sort` | `-o` | Sort output by descending frequency |
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

UNIQ has the **narrowest flag surface of any analysis command**:
just `-o` (sort by descending frequency). chatter exposes it as
`--sort`, and the rewriter routes `-o` → `--sort` under UNIQ
(pinned by `clan_args::tests::uniq_sort`).

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `-o` | Sort by descending frequency | `--sort` | Done | Rewriter arm in `clan_args.rs` consumes `-o` under UNIQ only; other commands route `-o` differently (FIXBULLETS bullet offset, etc.). |

Inherited general flags: same as [FREQ](./freq.md#general-flags-freq-inherits-from-cutt-cpp-mainusage). Audit summary: 7 Done / 0 Partial / 4 Rewriter only / 3 Missing.

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
