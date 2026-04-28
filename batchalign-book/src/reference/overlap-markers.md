# Overlapping Speech in CHAT

**Status:** Current
**Last updated:** 2026-03-16

## Two Encodings for Overlapping Speech

When two speakers talk at the same time, CHAT supports two ways to represent
this in the transcript. Both are valid; which one you use depends on your
transcription conventions and analysis needs.

### `&*` — Embedded Overlap Marker

The `&*` marker embeds one speaker's words inside another speaker's utterance.
The syntax is `&*SPEAKER:word` or `&*SPEAKER:word_word` (underscores join
compound expressions because `&*` only allows a single token).

```
*PAR:	I went to the store &*INV:mhm and bought some milk . 0_6000
```

Here, INV said "mhm" while PAR was talking. The `&*INV:mhm` is placed at
the approximate position in PAR's text where the overlap occurred.

**Properties:**

- INV's backchannel has **no timing of its own** — it is subsumed by PAR's
  bullet.
- INV's backchannel has **no %mor, %gra, or %wor** — it is invisible to all
  dependent tiers and alignment.
- INV's backchannel cannot be counted as an independent utterance by analysis
  tools (FREQ, MLU, etc.).
- Multi-word overlaps use underscores: `&*INV:oh_okay_yeah`.

**Corpus scale:** ~35,000 `&*` markers across ~2,200 files in 8 corpora.

### `+<` with Separate Utterances — Recommended

Each speaker's words go on their own line. The `+<` (lazy overlap) linker
marks that the utterance started before the previous one finished:

```
*PAR:	I went to the store and bought some milk . 0_6000
*INV:	+< mhm . 3500_4000
```

**Properties:**

- INV's backchannel gets **its own timing** from the aligner.
- INV's backchannel can receive **its own %mor and %wor** tiers.
- INV's backchannel is a **separate utterance**, countable by analysis tools.
- PAR's utterance stays intact — the participant's thought is one unit.
- Cross-speaker overlap is valid CHAT (E701 only requires non-decreasing start
  times).

**Corpus scale:** ~327,000 `+<` utterances across ~15,600 files in 14 corpora.

## Which Should I Use?

**For new transcription:** Use `+<` with separate utterances. Each speaker's
words belong on their own tier. This gives backchannels their own timing,
their own dependent tiers, and makes them countable by analysis tools.
`batchalign3 align` automatically uses a two-pass overlap-aware alignment
strategy when `+<` is present — no flags needed.

**For existing files with `&*`:** They work fine as-is. The aligner already
handles `&*` correctly (it is invisible to the DP alignment). No migration is
required. However, backchannels encoded as `&*` will never get independent
timing — they are invisible to the aligner by design.

**Both encodings are valid CHAT.** The aligner supports both. Files with `&*`
and files with `+<` can coexist in the same corpus.

**Summary of tradeoffs:**

| | `&*` encoding | `+<` separate utterances |
|-|---------------|--------------------------|
| Backchannel timing | None (invisible to aligner) | Automatic (two-pass recovery) |
| Backchannel %mor/%wor | None | Yes (own tiers) |
| Countable as utterance | No | Yes |
| Main speaker alignment | Unaffected | Unaffected |
| Readability at density | Poor (3+ `&*` per line) | Clean |
| Requires migration | No (existing files work) | New convention for new transcripts |

## How the Aligner Handles Each Encoding

### `&*` encoding

The content walker skips `OtherSpokenEvent` (`&*`) nodes entirely. They do not
participate in UTR word extraction, forced alignment, or `%wor` generation.
The backchannel words are invisible to the DP reference sequence.

**Result:** The main speaker's alignment is unaffected. The backchannel gets
no independent timing.

### `+<` encoding

When `batchalign3 align` encounters `+<` utterances, it uses a **two-pass
UTR strategy** with automatic fallback:

1. **Pass 1:** Build the global alignment reference from non-`+<` utterances
   only. Main-speaker words align correctly without backchannel interference.
2. **Pass 2:** For each `+<` utterance, search the previous utterance's audio
   window with adaptive widening to recover the backchannel's timing.
3. **Fallback:** If two-pass timed fewer utterances than the standard global
   algorithm would have, the global results are used instead. This ensures
   the strategy is **never worse** than the original algorithm — important
   for languages where ASR quality is lower.

**This is the default behavior — no flags needed.** Files without `+<` use
the original single-pass algorithm (identical results to previous versions).

The `--utr-strategy` flag exists for experimentation and diagnostics:

```bash
# Default: auto-detect (two-pass when +< present, global otherwise)
batchalign3 align corpus/ -o output/

# Force global single-pass (ignore +< signal, original algorithm)
batchalign3 align corpus/ -o output/ --utr-strategy global

# Force two-pass even on files without +<
batchalign3 align corpus/ -o output/ --utr-strategy two-pass
```

## Multi-Backchannel Example

The `&*` encoding becomes hard to read with multiple backchannels:

```
*PAR:	but I grew up in Princeton &*INV:oh_okay_yeah and came to
	graduate school &*INV:mhm at Chapel_Hill &*INV:oh in ninety
	one &*INV:mhm or maybe ninety two . 104745_118254
```

The `+<` encoding is cleaner:

```
*PAR:	but I grew up in Princeton and came to graduate school at
	Chapel_Hill and in ninety one or maybe ninety two . 104745_118254
*INV:	+< oh okay yeah .
*INV:	+< mhm .
*INV:	+< oh .
*INV:	+< mhm .
```

Each backchannel is a separate conversational act on its own line. PAR's
narrative is one unbroken utterance.

## CHAT Validation Rules

Cross-speaker overlapping bullets are valid:

- **E701** (global timeline): Start times must be non-decreasing. PAR at
  104745 ≤ INV at any time after that. Passes.
- **E704** (same-speaker self-overlap): Only prohibits the same speaker
  overlapping themselves beyond 500ms. Different speakers can overlap freely.
- **E362** (monotonicity): Same as E701. Passes.

## References

- [Utterance Linkers](https://talkbank.org/0info/manuals/CHAT.html#Utterance_Linkers)
- [Lazy Overlap Linker](https://talkbank.org/0info/manuals/CHAT.html#LazyOverlap_Linker)
