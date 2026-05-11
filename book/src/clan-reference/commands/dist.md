# DIST -- Word Distribution Across Turns

**Status:** Current
## Purpose

Counts turns and tracks for each word the first and last turn in which it appears. DIST is part of the FREQ family of commands and is useful for studying when words first appear and how their usage is distributed across a conversation.

## Usage

```bash
chatter clan dist file.cha
chatter clan dist --speaker CHI file.cha
chatter clan dist --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Output

Global word list (sorted alphabetically by display form) with:

- Occurrence count across all turns
- First turn number (1-based) in which the word occurs
- Last turn number (omitted if same as first)
- Total number of turns in the transcript

## Turn Definition

**Every utterance is its own turn**, regardless of whether the speaker changed. This matches CLAN's behavior, which was verified during parity testing. There is no speaker-continuity grouping -- each utterance increments the turn counter.

This is different from how turns are defined in MLT (where consecutive utterances by the same speaker form a single turn).

## Differences from CLAN

### Turn counting

Every utterance = one turn (no speaker-continuity grouping), matching CLAN exactly.

### Word identification

Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
