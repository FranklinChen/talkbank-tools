# FREQPOS — Word Frequency by Position

**Status:** Current
## Purpose

Counts how often each word appears in initial, final, other (middle), or one-word positions within utterances. FREQPOS is part of the FREQ family of commands and is useful for studying positional word preferences -- for example, whether a child tends to place certain words at the beginning or end of utterances.

### Position Classification

- **Initial**: first word of a multi-word utterance
- **Final**: last word of a multi-word utterance
- **Other**: any middle word of a multi-word utterance (3+ words)
- **One-word**: the sole word in a single-word utterance

## Usage

```bash
chatter clan freqpos file.cha
chatter clan freqpos file.cha --speaker CHI
```

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

Global word list (sorted alphabetically by display form) with positional breakdown (initial/final/other/one-word counts per word), followed by aggregate position totals.

## Differences from CLAN

- Word identification uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching
- Position classification operates on parsed AST word lists rather than raw text token splitting
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
