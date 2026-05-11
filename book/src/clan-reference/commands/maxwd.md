# MAXWD -- Longest Words

**Status:** Current
## Purpose

Finds the longest words used by each speaker, reporting a ranked table of unique words sorted by character length descending. Word length is measured in characters after normalization (lowercasing, stripping `+` compound markers and `'` apostrophes for CLAN compatibility).

## Usage

```bash
chatter clan maxwd file.cha
chatter clan maxwd --speaker CHI file.cha
chatter clan maxwd --limit 50 file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--limit <N>` | -- | Maximum number of words to show (default: 20) |
| `--format <FMT>` | -- | Output format: text, json, csv |

## Output

Per speaker:

- Table of longest words sorted by length descending (up to `limit`)
- **All occurrences with line numbers** (matching CLAN)
- Maximum word length
- Mean word length
- Total and unique word counts

## Differences from CLAN

### Occurrence reporting

Reports **all occurrences with line numbers**, matching CLAN's output format exactly.

### Word normalization

Length is measured after stripping `+` (compound markers) and `'` (apostrophes), matching CLAN's character counting behavior.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
