# PLAY2CHAT -- PLAY Annotation to CHAT Conversion

**Status:** Current
**Last updated:** 2026-05-22 13:40 EDT

## Purpose

Converts PLAY (Phonological and Lexical Acquisition in Young children) annotation files into CHAT format.

## Usage

```bash
chatter clan play2chat input.play
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-l`, `--language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `-o`, `--output` | stdout | Output CHAT file path |

The corpus name in `@ID` headers is hardcoded to `"play_corpus"`
(`crates/talkbank-clan/src/converters/play2chat.rs:92`); there is
no CLI flag to override it. If you need a different corpus name,
post-edit the generated `@ID` lines or call
`play_to_chat_with_options()` from Rust.

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

## CLAN `+`-flag coverage audit

PLAY2CHAT is a **converter** — input Datavyu text, output CHAT.
Sources: `OSX-CLAN/src/clan/Play2Chat.cpp::usage`,
`crates/talkbank-clan/src/converters/play2chat.rs`.

### PLAY2CHAT-specific `+`-flags (from `Play2Chat.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `+d` | Check utterances for illegal overlaps | — | Missing |

Audit summary: 1 Done (default conversion), 1 Missing. The
overlap-check is a validation pass that chatter does not run;
researchers wanting that check can pipe the output to
`chatter clan check`.
