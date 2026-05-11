# COOCCUR — Word Co-occurrence (Bigram) Counting

**Status:** Current
## Purpose

Counts adjacent word pairs (bigrams) across utterances. For each utterance, every pair of consecutive countable words is recorded as a directed bigram. Pairs are directional: ("put", "the") and ("the", "put") are counted separately.

COOCCUR is part of the FREQ family of commands and is useful for studying word collocations and sequential patterns in speech.

## Usage

```bash
chatter clan cooccur file.cha
chatter clan cooccur file.cha --speaker CHI
```

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

- Table of adjacent word pairs with co-occurrence counts
- Default sort: by frequency descending, then alphabetically
- CLAN output: sorted alphabetically by pair display form
- Summary: unique pair count, total pair instances, total utterances

## Differences from CLAN

- Word identification uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching
- Bigram extraction operates on parsed AST content rather than raw text
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
