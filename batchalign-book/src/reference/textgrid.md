# TextGrid Format and Export

**Date**: 2026-02-15
**Status**: Current export support
**Last verified**: 2026-03-05
---

## What is TextGrid?

**TextGrid** is a file format used by [Praat](https://www.fon.hum.uva.nl/praat/), a popular tool for phonetic analysis and speech research.

### Purpose

TextGrid files allow researchers to:
- Visualize waveforms and spectrograms alongside transcriptions
- Measure acoustic properties (formants, pitch, duration, etc.)
- Annotate speech at multiple levels (word, phone, syllable, etc.)
- Align transcriptions with audio for detailed phonetic analysis

### Structure

A TextGrid consists of **tiers** (one per speaker or annotation layer), where each tier contains **intervals**:

```
Interval {
    xmin: 1.000     # Start time (seconds)
    xmax: 1.200     # End time (seconds)
    text: "hello"   # Label text
}
```

Example TextGrid with two speakers:

```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 10.5
tiers? <exists>
size = 2

item [1]:
    class = "IntervalTier"
    name = "CHI"
    xmin = 0.0
    xmax = 10.5
    intervals: size = 5
    intervals [1]:
        xmin = 0.0
        xmax = 1.2
        text = "hello"
    intervals [2]:
        xmin = 1.2
        xmax = 1.8
        text = "world"
    ...

item [2]:
    class = "IntervalTier"
    name = "MOT"
    xmin = 0.0
    xmax = 10.5
    intervals: size = 3
    ...
```

---

## Batchalign's TextGrid Export

TextGrid export support exists today. Remaining work is cleanup and
simplification, not initial implementation.

### Entry Point

**Python**: `batchalign/formats/textgrid/generator.py`

```python
def dump_textgrid(chat_text: str, by_word: bool = True) -> textgrid.Textgrid:
    """Convert CHAT text to a praatio Textgrid.

    Parameters
    ----------
    chat_text : str
        Valid CHAT text (must have timing data).
    by_word : bool
        If True, each word becomes an interval; if False, each utterance.

    Returns
    -------
    Textgrid
        A praatio Textgrid object (can be saved to .TextGrid file).
    """
    import batchalign_core

    # Extract timing data from CHAT
    tiers_json = json.loads(
        batchalign_core.extract_timed_tiers(chat_text, by_word)
    )

    # Build TextGrid
    tg = textgrid.Textgrid()
    for speaker, entries in tiers_json.items():
        intervals = [
            Interval(
                float(e["start_ms"]) / 1000,  # Convert ms → seconds
                float(e["end_ms"]) / 1000,
                str(e["text"]),
            )
            for e in entries
        ]
        if intervals:
            tg.addTier(textgrid.IntervalTier(
                speaker, intervals,
                intervals[0].start, intervals[-1].end,
            ))

    return tg
```

### Usage

```python
from batchalign.formats.textgrid import dump_textgrid

# Read CHAT file (must have timing from forced alignment)
with open("transcript.cha") as f:
    chat_text = f.read()

# Export to TextGrid
tg = dump_textgrid(chat_text, by_word=True)

# Save to file
tg.save("transcript.TextGrid")

# Open in Praat for analysis
```

---

## Implementation

The Rust function `extract_timed_tiers()` in `pyo3/src/pyfunctions.rs` handles
CHAT parsing and timed data extraction. When a `%wor` tier exists, it is
preferred as the timing source; otherwise main-tier word timing is used.
See the source for implementation details.

---

## Dependencies

- **praatio**: Python library for reading/writing TextGrid files
- **batchalign_core.extract_timed_tiers**: Rust function (PyO3 binding)
- **Forced alignment**: TextGrid export only works on files with timing data

TextGrid export is current supported functionality for Praat-oriented workflows.
