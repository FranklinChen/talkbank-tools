# CHAT2PRAAT -- CHAT to Praat TextGrid

**Status:** Current
**Last updated:** 2026-05-22 13:30 EDT

## Purpose

Converts CHAT files to Praat TextGrid format for acoustic/phonetic analysis. Each speaker becomes a separate interval tier, with timed utterances mapped to intervals.

This is the reverse of [PRAAT2CHAT](praat2chat.md). Both conversions are implemented in the same module (`praat2chat`).

## Usage

```bash
chatter clan chat2praat file.cha
chatter clan chat2praat file.cha -o output.TextGrid
```

## CLAN `+`-flag coverage audit

CHAT2PRAAT is a **converter** — input CHAT, output Praat
TextGrid. Sources: `OSX-CLAN/src/clan/Chat2Praat.cpp::usage`,
`crates/talkbank-clan/src/converters/chat2praat.rs`.

### CHAT2PRAAT-specific `+`-flags (from `Chat2Praat.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `+eS` | Media file name extension | — | Missing |

Audit summary: 1 Done (default conversion), 1 Missing. The
`+eS` media-extension flag is a hint for TextGrid's
sound-association header; chatter omits this metadata today.
Filed as a Phase 1.7 follow-up.

## Behavior

- Each speaker in the CHAT file becomes a separate interval tier
- Only timed utterances (those with bullet timing) are included
- Timing bullets are converted from milliseconds to seconds
- Speaker codes become tier names
- Utterance text is extracted with annotations stripped
- If no timed utterances exist, an empty string is returned

## Output format

Produces standard Praat TextGrid long format:

```text
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5.042652
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "CHI"
        ...
```

## Differences from CLAN

- Uses the typed AST to extract timing and text, rather than string scanning
- Bidirectional conversion in one module (CLAN has separate `praat2chat` and `chat2praat` binaries)
- Deterministic speaker tier ordering via `BTreeMap`
