# CHAT2ELAN -- CHAT to ELAN XML Conversion

**Status:** Current
**Last updated:** 2026-05-22 13:30 EDT

## Purpose

Converts CHAT files into ELAN annotation format (`.eaf`). The output is a valid ELAN XML file with time-aligned tiers and annotations derived from CHAT main tiers and their timing bullets.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409300) for the original CHAT2ELAN command documentation.

## Usage

```bash
chatter clan chat2elan file.cha
chatter clan chat2elan --media-extension wav file.cha
```

## CLAN Equivalence

| CLAN command                   | Rust equivalent                             |
|--------------------------------|---------------------------------------------|
| `chat2elan +e.wav file.cha`    | `chatter clan chat2elan file.cha`           |

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--media-extension <EXT>` | `+e.EXT` | Include a `MEDIA_DESCRIPTOR` element referencing a media file with the same basename and the given extension |

Without `--media-extension`, the output omits media references.

## CLAN `+`-flag coverage audit

CHAT2ELAN is a **converter** — input CHAT, output ELAN XML
(`.eaf`). Sources: `OSX-CLAN/src/clan/Chat2Elan.cpp::usage`,
`crates/talkbank-clan/src/converters/chat2elan.rs`.

### CHAT2ELAN-specific `+`-flags (from `Chat2Elan.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `+eS` | Media file name extension | `--media-extension <EXT>` | Done |

Audit summary: 1 Done, 0 Missing. Single-flag surface, mapped
one-to-one. Rewriter routing for `+eS` is a follow-up.

## Output

A standard-compliant EAF 3.0 XML file. Each CHAT speaker becomes an ELAN tier (using the full speaker code as tier ID). Utterances with timing bullets become time-aligned annotations. The time slot pool is derived from all utterance start/end timestamps.

## Differences from CLAN

- CLAN requires the `+e` flag to specify a media extension; the Rust version omits media references by default.
- Uses typed AST traversal for utterance extraction rather than text parsing.
- Generates standard-compliant EAF 3.0 XML.
- Speaker codes are used directly as tier IDs (no truncation).
