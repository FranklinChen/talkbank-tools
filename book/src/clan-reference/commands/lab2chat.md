# LAB2CHAT -- LAB Timing Labels to CHAT Conversion

**Status:** Current
## Purpose

Converts LAB (label) timing files into CHAT format. LAB files contain time-aligned word or segment labels commonly used in speech research tools (e.g., HTK, Kaldi).

## Usage

```bash
chatter clan lab2chat input.lab
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `speaker` | `"SPK"` | Speaker code for all utterances |
| `language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `corpus` | `"lab_corpus"` | Corpus name for the `@ID` header |

## Supported Formats

- **Three-column**: `start_time end_time label` (times in seconds)
- **Two-column**: `time label` (end time inferred from the next entry)

## Input Format

Plain text files with whitespace-separated columns. Silence markers (`sil`, `sp`, `#`) are skipped during conversion. Comment lines starting with `#` and blank lines are ignored.

Example:

```text
0.0 0.5 hello
0.5 1.2 world
1.2 1.5 sil
```

## Output

A well-formed CHAT file where each non-silence label becomes a separate utterance with timing bullets derived from the LAB timestamps (converted from seconds to milliseconds).

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output
