# LENA2CHAT -- LENA Device XML to CHAT Conversion

**Status:** Current
**Last updated:** 2026-05-22 13:40 EDT

## Purpose

Converts LENA (Language Environment Analysis) device output files (`.its` format) into CHAT format. LENA XML contains segment-level annotations with speaker types and timing information but no actual transcribed words.

## Usage

```bash
chatter clan lena2chat input.its
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `language` | `"eng"` | ISO 639 language code for the `@Languages` header |
| `corpus` | `"lena_corpus"` | Corpus name for the `@ID` header |

## Speaker Mapping

LENA segment types are mapped to CHAT speaker codes:

| LENA type | CHAT speaker | Description |
|-----------|-------------|-------------|
| `CHN`/`CXN` | `CHI` | Child near/far |
| `FAN`/`FAF` | `MOT` | Female adult near/far |
| `MAN`/`MAF` | `FAT` | Male adult near/far |
| `OLN`/`OLF` | `OTH` | Other child overlap |
| `TVN`/`TVF` | `ENV` | TV/electronic media |
| `NON`/`NOF` | `ENV` | Noise |
| `SIL` | *(skipped)* | Silence |

## Input Format

LENA `.its` XML files containing segment-level annotations with speaker type attributes and timing information.

## Output

A well-formed CHAT file where each LENA segment becomes a timed utterance. Since LENA does not provide transcribed words, all utterances use `xxx` (untranscribed) as placeholder text, with optional word count annotation.

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output

## CLAN `+`-flag coverage audit

LENA2CHAT is a **converter** — input LENA `.its` XML, output
CHAT. Sources: `OSX-CLAN/src/clan/Lena2Chat.cpp::usage`,
`crates/talkbank-clan/src/converters/lena2chat.rs`.

CLAN's `Lena2Chat.cpp::usage` documents **no command-specific
`+`-flags** — the surface is just the inherited general flag
set (mostly `+re` for recursion). Byte-parity complete for the
documented surface.

Audit summary: 1 Done, 0 Missing.
