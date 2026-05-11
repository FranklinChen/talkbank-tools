# CHIP -- Child/Parent Interaction Profile

**Status:** Current
## Purpose

Analyzes interaction patterns between a child speaker and their conversational partners. Categorizes successive utterance pairs to measure imitation, repetition, and overlap. CHIP is commonly used in child language research to quantify how much a child imitates or echoes their interlocutor.

## Usage

```bash
chatter clan chip file.cha
chatter clan chip --speaker CHI file.cha
chatter clan chip --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Interaction Categories

For each adjacent utterance pair (speaker A followed by speaker B):

| Category | Condition |
|----------|-----------|
| **Exact repetition** | B's words are identical to A's (order-independent) |
| **Overlap** | B shares >= 50% of words with A (smaller unique-word set as denominator) |
| **No overlap** | B shares < 50% of words with A |

Only cross-speaker adjacency is considered; consecutive utterances by the same speaker do not produce interaction records. Adjacency state is reset at file boundaries.

## Output

**36-measure matrix format** matching CLAN exactly:

- ADU (adult) / CHI (child) / ASR (adult-speech-related) / CSR (child-speech-related) columns
- Per directed speaker pair (MOT->CHI is distinct from CHI->MOT)
- Counts and percentages for each interaction category
- Grand totals across all pairs

### Echo behavior

When displaying matched utterances, CHIP echoes:
- Main tier text
- `%mor` tier (if present)

It does **not** echo `%gra` tiers, matching CLAN's behavior.

## Differences from CLAN

### Matrix format

Uses the exact **36-measure matrix format** with ADU/CHI/ASR/CSR columns, matching CLAN character-for-character.

### Echo content

Echoes main tier + `%mor` only (not `%gra` tiers), matching CLAN.

### Word identification

Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching. Overlap comparison operates on parsed word content, not raw text.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
