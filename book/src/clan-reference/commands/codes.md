# CODES -- Code Frequency Table

**Status:** Current
## Purpose

Tabulates the frequency and distribution of coding annotations found on `%cod:` dependent tiers, organized by speaker. This is useful for analyzing hand-coded behavioral or discourse annotations attached to transcripts.

Codes on `%cod:` tiers typically use colon-separated hierarchical structure (e.g., `AC:DI:PP`), but this implementation treats each whitespace-delimited token as a single code string without parsing the internal hierarchy.

The [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) does not
appear to contain a standalone `CODES` section. The closest direct evidence is
the `CHAINS` section, which documents coding tiers and the `codes.ord` file
used to order `$`-prefixed codes.

## Usage

```bash
chatter clan codes file.cha
chatter clan codes --speaker CHI file.cha
chatter clan codes --format json file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--max-depth <N>` | Maximum depth of code parsing (0 = all levels) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Output

Per-speaker frequency tables listing each code and its count, plus a per-speaker total and a grand total across all speakers.

## Differences from CLAN

- **Manual coverage gap**: the
  [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) has no
  standalone `CODES` section, so this chapter is based on implemented
  behavior, golden tests, and nearby coding-tier documentation such as
  `CHAINS`.
- **Code extraction**: Codes are extracted from a clan-local semantic `%cod` layer built from the parsed AST, not from raw line text.
- **Selector handling**: `%cod` extraction preserves optional selectors like `<w4>` or `<w4-5>` as item scope rather than counting them as standalone codes.
- **Hierarchy handling**: Each whitespace-delimited token is treated as a single code string; colon-separated hierarchy is preserved but not parsed into sublevels.
- **Output formats**: Supports text, JSON, and CSV formats (CLAN produces text only).
- **Deterministic ordering**: `BTreeMap` ordering ensures deterministic output across runs.
- **Golden test parity**: Verified against CLAN C binary output.
