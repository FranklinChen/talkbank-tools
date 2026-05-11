# PHONFREQ — Phonological Frequency Analysis

**Status:** Current
## Purpose

Counts individual phone (character) occurrences from `%pho` tier content, tracking positional distribution within each phonological word: initial (first character), final (last character), and other (middle positions). Only lowercase ASCII letters are counted, matching CLAN's behavior.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409227) for the original PHONFREQ command specification.

## Usage

```bash
chatter clan phonfreq file.cha
chatter clan phonfreq file.cha --speaker CHI
```

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

Per-phone frequency with positional breakdown (initial/final/other), sorted alphabetically by phone character.

## Differences from CLAN

- Phone extraction uses parsed `%pho` tier structure from the AST rather than raw text character scanning
- Positional classification operates on typed `PhoWord` content
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
