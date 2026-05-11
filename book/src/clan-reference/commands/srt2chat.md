# SRT2CHAT -- SRT Subtitle to CHAT Conversion

**Status:** Current
## Purpose

Parses SRT (SubRip) subtitle files and converts them to CHAT format, mapping each subtitle block to an utterance with timing bullets derived from the SRT timestamps.

## Usage

```bash
chatter clan srt2chat input.srt
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `speaker` | `"SPK"` | Speaker code for all utterances |
| `language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `corpus` | `"srt_corpus"` | Corpus name for the `@ID` header |

## Input Format

SRT files consist of numbered blocks separated by blank lines:

```text
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,200 --> 00:00:06,800
How are you
```

Timestamps use `HH:MM:SS,mmm` format (both comma and period separators are accepted). Multi-line subtitle text within a block is joined with spaces.

## Output

A well-formed CHAT file where each SRT subtitle block becomes a timed utterance. Timing bullets are derived from the SRT timestamps (converted to milliseconds). All utterances are assigned to the configured speaker code.

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output
- Accepts both comma and period as millisecond separators in timestamps
