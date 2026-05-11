# CHAT2PRAAT -- CHAT to Praat TextGrid

**Status:** Current
## Purpose

Converts CHAT files to Praat TextGrid format for acoustic/phonetic analysis. Each speaker becomes a separate interval tier, with timed utterances mapped to intervals.

This is the reverse of [PRAAT2CHAT](praat2chat.md). Both conversions are implemented in the same module (`praat2chat`).

## Usage

```bash
chatter clan chat2praat file.cha
chatter clan chat2praat file.cha -o output.TextGrid
```

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
