# Parity Improvement Plan

**Status:** Current
Current: **95% (113/118)** — 5 accepted divergences across 2 commands.

## Progress

Started at 74% (88/118). Fixed 25 divergences across 10 commands:

| Command | Divergences Fixed | Key Changes |
|---------|-------------------|-------------|
| MLU | 4 → 0 | Brown's morpheme counting, population SD |
| MLT | 4 → 0 | Per-utterance SD, population SD |
| DIST | 2 → 0 | Every-utterance turn counting |
| MAXWD | 1 → 0 | All occurrences with line numbers |
| TIMEDUR | 4 → 0 | Interaction matrix header leading space |
| VOCD | 4 → 0 | Echo %mor lemmas (stripped fusional), insufficient tokens |
| WDLEN | 4 → 0 | 6-section format, Brown's morphemes, POS-inclusive section 6, apostrophe stripping, reversed speaker order, XML footer |
| UNIQ | 3 → 1 | Include %mor/%gra tiers, split multi-line headers |
| CHIP | 4 → 0 | 36-measure CLAN matrix format (main + %mor echo, no %gra) |

## Remaining Divergences (5) — All Accepted

### DELIM — 4 divergences (all test files)

CLAN writes an empty `.cex` file when no changes are needed (all utterances already
have terminators). Our transform always writes the full file. This is by design —
the transform pipeline always writes output, and empty-file detection would require
change tracking that adds complexity without user benefit.

### UNIQ — 1 divergence (overlaps only)

Unicode sort order edge case: `⌊` (U+230A, LEFT FLOOR) sorts differently between
CLAN's C-locale `strcoll()` and Rust's BTreeMap byte-order comparison. Single line
position swap, identical content and counts. Not worth adding locale-dependent sorting.

## Key Discoveries

1. **Brown's Morpheme Rules**: CLAN MLU/WDLEN count only 7 suffix strings as
   bound morphemes: `PL`, `PAST`, `Past`, `POSS`, `PASTP`, `Pastp`, `PRESP`.

2. **Population SD**: Both MLU and MLT use population SD (/ n), not sample (/ n-1).

3. **MLT SD basis**: Computed over per-utterance word counts, not per-turn totals.

4. **DIST turns**: Every utterance is its own turn (no speaker-continuity grouping).

5. **WDLEN morphemes**: Section 5 = stem + Brown's suffix (no POS). Section 6 =
   POS + stem + Brown's suffix. Clitic pairs = one word in section 5, POS counted
   only for main word in section 6.

6. **WDLEN characters**: CLAN strips apostrophes before counting character length.

7. **Speaker ordering**: CLAN outputs speakers in reverse encounter order (linked-list
   prepend pattern from the C implementation).

8. **Fusional features**: `&PRES`, `&INF` etc. are stored as part of the lemma string
   in our model (for roundtrip fidelity). Strip with `split('&')` when needed.

9. **CHIP echo**: CLAN echoes main tiers + %mor only, not %gra tiers.
