# CHAT2ELAN -- CHAT to ELAN XML Conversion

**Status:** Current
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

## Output

A standard-compliant EAF 3.0 XML file. Each CHAT speaker becomes an ELAN tier (using the full speaker code as tier ID). Utterances with timing bullets become time-aligned annotations. The time slot pool is derived from all utterance start/end timestamps.

## Differences from CLAN

- CLAN requires the `+e` flag to specify a media extension; the Rust version omits media references by default.
- Uses typed AST traversal for utterance extraction rather than text parsing.
- Generates standard-compliant EAF 3.0 XML.
- Speaker codes are used directly as tier IDs (no truncation).
