# TIMEDUR -- Time Duration

**Status:** Current
## Purpose

Computes time duration statistics from media timestamp bullets attached to utterances. Utterances without bullet timing are silently skipped.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409240) for the original TIMEDUR command specification.

## Usage

```bash
chatter clan timedur file.cha
chatter clan timedur --speaker CHI file.cha
chatter clan timedur --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Output

Per speaker:

- Number of timed utterances
- Total duration (formatted as HH:MM:SS.mmm)
- Mean utterance duration
- Min/max duration

Plus a corpus-wide summary:

- Total timed utterances across all speakers
- Total duration
- Recording span (earliest start to latest end)
- Speaker interaction matrix (overlap and gap analysis)

## Differences from CLAN

### Timestamp extraction

Uses parsed media bullet structures from the AST (`Bullet { start_ms, end_ms }`) rather than raw byte scanning in text. This is more robust against formatting variations.

### Interaction matrix header

The interaction matrix header includes a leading space, matching CLAN exactly. This was verified during golden test parity work.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
