# TEXT2CHAT -- Plain Text to CHAT Conversion

**Status:** Current
## Purpose

Converts plain text files into CHAT format by splitting on sentence-ending punctuation (`.`, `?`, `!`) and assigning all utterances to a default speaker. This is the simplest converter, useful for bootstrapping CHAT files from raw text.

## Usage

```bash
chatter clan text2chat input.txt
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `speaker` | `"SPK"` | Speaker code for all utterances |
| `language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `corpus` | `"text_corpus"` | Corpus name for the `@ID` header |

## Input Format

Plain text files. Newlines within the input are treated as spaces (not sentence boundaries). The text is split into utterances at sentence-ending punctuation (`.`, `?`, `!`).

## Output

A well-formed CHAT file where each sentence becomes an utterance. Sentence terminators are preserved as CHAT terminators (period, question mark, exclamation point). Trailing text without punctuation receives a default period terminator.

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output
