# KWAL -- Keyword And Line

**Status:** Current
## Purpose

Searches for clusters containing specified keywords and displays the matching lines with context. The legacy manual gives `KWAL` a dedicated section and describes it as operating on "clusters": the main tier plus the selected dependent tiers associated with that line.

In `talkbank-clan`, keywords are currently matched against countable words on the main tier, with the matched utterance shown in context.

## Usage

```bash
chatter clan kwal -k want file.cha
chatter clan kwal -k want --speaker CHI file.cha
chatter clan kwal -k want -k cookie file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `-k <WORD>` / `--keyword <WORD>` | Keyword to search for (repeatable) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `kwal +s"want" file.cha` | `chatter clan kwal file.cha -k want` |
| `kwal +s"want" +t*CHI file.cha` | `chatter clan kwal file.cha -k want --speaker CHI` |

## Output

Each matching utterance with:

- Speaker code
- Full utterance text
- File path (for multi-file searches)
- Match count summary per keyword

## Differences from CLAN

- **Manual intent**: `KWAL` is a cluster-oriented search command, not just a main-tier keyword matcher.
- **Search**: Operates on parsed AST word content rather than raw text lines.
- **Word identification**: Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching.
- **Scope reduction**: The legacy manual describes richer tier-selection and output-shaping behavior, including cluster searches over selected dependent tiers and `%mor`/`%gra` combined searches with `+d7`. The current implementation is narrower.
- **Output formats**: Supports text, JSON, and CSV formats (CLAN produces text only).
- **Golden test parity**: Verified against CLAN C binary output.
