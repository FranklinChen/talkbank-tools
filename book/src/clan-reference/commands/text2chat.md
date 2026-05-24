# TEXT2CHAT -- Plain Text to CHAT Conversion

**Status:** Current
**Last updated:** 2026-05-22 13:40 EDT

## Purpose

Converts plain text files into CHAT format by splitting on sentence-ending punctuation (`.`, `?`, `!`) and assigning all utterances to a default speaker. This is the simplest converter, useful for bootstrapping CHAT files from raw text.

## Usage

```bash
chatter clan text2chat input.txt
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--speaker` | `"SPK"` | Speaker code for all utterances |
| `-l`, `--language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `-o`, `--output` | stdout | Output CHAT file path |

The corpus name in `@ID` headers is hardcoded to `"text_corpus"`
(`crates/talkbank-clan/src/converters/text2chat.rs:37`); there is
no CLI flag to override it. Same pattern as the other converters
in this directory.

## Input Format

Plain text files. Newlines within the input are treated as spaces (not sentence boundaries). The text is split into utterances at sentence-ending punctuation (`.`, `?`, `!`).

## Output

A well-formed CHAT file where each sentence becomes an utterance. Sentence terminators are preserved as CHAT terminators (period, question mark, exclamation point). Trailing text without punctuation receives a default period terminator.

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output

## CLAN `+`-flag coverage audit

TEXT2CHAT is a **converter** — input plain text, output CHAT.
Sources: `OSX-CLAN/src/clan/text2chat.cpp::usage`,
`crates/talkbank-clan/src/converters/text2chat.rs`.

### TEXT2CHAT-specific `+`-flags (from `text2chat.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `+c0` | Insert `@Blank`/`@Indent` headers when appropriate | — | Missing |
| `+c1` | Each line is an utterance regardless of delimiter | — | Missing |
| `+c2` | Convert first capitalized word of utterance/quotation to lowercase | — | Missing |
| `+c3` | Convert `[...]` lines to `*INV: ...` | — | Missing |

Audit summary: 1 Done (default line-per-utterance conversion),
4 Missing. The `+cN` family is a tokenization-policy switch
ladder; chatter implements one fixed policy. Filed as a Phase
1.7 follow-up.
