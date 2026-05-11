# PLAY2CHAT -- PLAY Annotation to CHAT Conversion

**Status:** Current
## Purpose

Converts PLAY (Phonological and Lexical Acquisition in Young children) annotation files into CHAT format.

## Usage

```bash
chatter clan play2chat input.play
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `corpus` | `"play_corpus"` | Corpus name for the `@ID` header |

## Input Format

Tab-separated fields: `speaker`, `start_time`, `end_time`, `text`. Times are in milliseconds and may be empty. Lines starting with `#` or `%` are skipped. Lines with fewer than 2 tab-separated fields are ignored.

Example:

```text
CHI	1000	3500	hello world
MOT	4200	6800	how are you
```

## Output

A well-formed CHAT file with headers and participants. Unique speakers are automatically collected and registered as CHAT participants with the `Unidentified` role. Each PLAY entry becomes an utterance, with timing bullets when start/end times are provided.

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output
