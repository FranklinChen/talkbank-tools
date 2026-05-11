# MLT -- Mean Length of Turn

**Status:** Current
## Purpose

Calculates mean length of turn in utterances and words. A "turn" is a maximal consecutive sequence of utterances by the same speaker; the turn boundary is detected when a different speaker produces the next utterance.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409101) for the original MLT command specification.

## Usage

```bash
chatter clan mlt file.cha
chatter clan mlt --speaker CHI file.cha
chatter clan mlt --format json corpus/
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--format <FMT>` | -- | Output format: text, json, csv, clan |

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `mlt file.cha` | `chatter clan mlt file.cha` |
| `mlt +t*CHI file.cha` | `chatter clan mlt file.cha --speaker CHI` |

## Algorithm

1. Walk utterances in file order
2. Group consecutive utterances by the same speaker into turns
3. For each speaker, compute:
   - Number of turns
   - Total utterances and total words
   - **MLT-u**: mean turn length in utterances
   - **MLT-w**: mean turn length in words
   - **SD**: standard deviation of words per utterance (population SD, dividing by n)

### Turn boundaries

A turn boundary occurs when a different speaker produces the next utterance. For example:

```text
*CHI: I want a cookie .          <- turn 1 (CHI)
*CHI: please .                   <- still turn 1 (CHI)
*MOT: here you go .              <- turn 2 (MOT)
*CHI: thank you .                <- turn 3 (CHI)
```

CHI has 2 turns (3 utterances), MOT has 1 turn (1 utterance).

## Output

```text
Speaker: CHI
  Turns: 15
  Utterances: 42
  Words: 127
  MLT (utterances): 2.800
  MLT (words): 8.467
  SD: 3.217
```

## Differences from CLAN

### Standard deviation

Uses **population SD** (dividing by n), matching CLAN. This was verified during parity testing.

### SD basis

The SD is computed over **per-utterance word counts**, not per-turn totals. This matches CLAN's behavior, which was confirmed through golden test comparison.

### Turn detection

Operates on parsed speaker codes from the AST rather than raw text line prefixes. Functionally identical but type-safe.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
