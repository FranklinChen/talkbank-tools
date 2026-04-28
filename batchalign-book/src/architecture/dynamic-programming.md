# Dynamic Programming in batchalign3

**Status:** Current
**Last updated:** 2026-03-26 00:52 EDT

## Purpose

This document inventories all runtime uses of dynamic programming (DP) in
`batchalign3`, and distinguishes algorithmically necessary uses from
architecturally avoidable runtime remap uses.

## Runtime DP Inventory

| Area | Call site | DP algorithm | Notes |
|---|---|---|---|
| Whisper ASR timestamp extraction | `batchalign/inference/audio.py` | DTW (`_dynamic_time_warping`) | Used to map decoder tokens to audio frames from cross-attention matrices. |
| Whisper FA token timing | `batchalign/inference/fa.py` | DTW (`_dynamic_time_warping`) | Token jump times are extracted in Python and mapped in Rust. |
| Wave2Vec forced alignment | `batchalign/inference/fa.py` | CTC forced alignment (Viterbi-style DP) | `torchaudio.functional.forced_align` on emission matrix vs transcript. |
| FA word-level remapping | `crates/batchalign-chat-ops/src/fa/alignment.rs:apply_indexed_timings` | **None** | Indexed callback protocol maps timings 1:1 by index. |
| FA token-level remapping | `crates/batchalign-chat-ops/src/fa/alignment.rs:align_token_timings` | **None** | Deterministic token→word stitching only; unmatched words remain untimed (no DP skip/remap). |
| UTR timing recovery | `crates/batchalign-chat-ops/src/fa/utr.rs:inject_utr_timing` | Hirschberg edit-distance DP | Global alignment of all document words against all ASR tokens. Correctness-critical: per-utterance windowing caused token exhaustion on hand-edited transcripts. Matches old batchalign's proven approach. |
| Morphosyntax retokenize mapping | `crates/batchalign-chat-ops/src/retokenize/mapping.rs:build_word_token_mapping` | **None** | Deterministic span-join mapping first; length-aware monotonic fallback when text diverges (no DP). |
| WER evaluation | `batchalign-chat-ops/src/dp_align.rs` + `wer_conform.rs` (Rust) | Hirschberg edit-distance DP | Canonical use of DP for transcript comparison. Python `benchmark.py` is a thin wrapper around `batchalign_core.wer_compute()`. |
| Compare command | `crates/batchalign-chat-ops/src/compare.rs` via `dp_align::align` | Hirschberg edit-distance DP | Aligns main vs gold transcript words to compute WER and inject `%xsrep` / `%xsmor` tiers. Same algorithm as WER evaluation. |

## Necessity Assessment

### Clearly legitimate DP

- **WER evaluation / compare command**: comparing two independent word sequences
  is exactly edit distance territory. The compare command uses the same Hirschberg
  algorithm to align main vs gold transcripts for `%xsrep` / `%xsmor`
  annotation.
- **CTC forced alignment (`forced_align`)**: DP is intrinsic to the model family.
- **Whisper DTW path**: if cross-attention DTW is the chosen alignment method,
  DP is part of the method.

### Architecturally avoidable DP

- **FA remapping DP (word/token to transcript)** can be removed if callbacks
  return timings indexed to the exact word list supplied by Rust.
  - Current status: removed from runtime remapping paths (indexed or deterministic
    stitching only).
- **Retokenize char-level DP** is often overkill when deterministic structural
  mapping is available.
  - Current status: removed; fallback is length-aware monotonic mapping.

### Correctness-critical DP

- **UTR global ASR→transcript DP** is a correctness-critical runtime use of DP.
  UTR has to align two independent full-document word sequences: transcript
  words and ASR tokens. A local/windowed matcher can starve later utterances of
  tokens that earlier utterances consumed, which is exactly what happened in the
  407-style hand-edited transcript regression.
  - Current status: UTR uses a single global Hirschberg alignment. This is
    classified as a legitimate DP use (same category as WER/compare), not an
    avoidable runtime remap.
  - Important limitation: this is still a monotonic aligner. Dense overlap and
    text/audio reordering can still remain unmatched in heavily reworked
    hand-edited transcripts.

```mermaid
flowchart TD
    input["Transcript words + ASR words"]
    cheap{"Unique exact subsequence?"}
    exact["Cheap monotonic mapping\n(no DP)"]
    global["Single global Hirschberg DP\n(correctness baseline)"]
    output["UTR timing injection"]

    input --> cheap
    cheap -->|"yes"| exact --> output
    cheap -->|"no"| global --> output
```

## Current status

- UTR uses a single global Hirschberg DP alignment of all document words
  against all ASR tokens. This is a correctness requirement — per-utterance
  windowing caused token starvation on real-world data. Timed utterances
  participate to anchor the alignment but their bullets are left unchanged.
  This fixes the 407-style failure class, not every possible hand-edited
  overlap/reordering case.
- Whisper FA callback emits indexed timings when deterministic stitching works.
- FA token-level remapping is deterministic-only (no DP).
- Retokenize mapping is deterministic-first with monotonic fallback (no DP).

Policy guard added: `batchalign/tests/test_dp_allowlist.py` fails CI if new
runtime DP callsites appear outside allowlisted surfaces. The allowlist permits:
- `pyo3/src/pyfunctions.rs` — PyO3 bridge (1 call)
- `crates/batchalign-chat-ops/src/benchmark.rs` — WER (1 call)
- `crates/batchalign-chat-ops/src/compare.rs` — transcript comparison (1 call)
- `crates/batchalign-chat-ops/src/fa/utr.rs` — UTR timing recovery (1 call)

## Fuzzy Matching in DP Alignment

The DP aligner supports three word-comparison modes via the `MatchMode` enum:

```mermaid
flowchart LR
    words["Word pair:<br/>transcript 'gonna'<br/>ASR 'gona'"]

    subgraph Exact["MatchMode::Exact"]
        exact_cmp["'gonna' == 'gona'?"]
        exact_no["NO match<br/>cost = 2 (substitution)"]
    end

    subgraph CaseInsensitive["MatchMode::CaseInsensitive"]
        ci_cmp["'gonna' =~ 'gona'?<br/>(case-insensitive)"]
        ci_no["NO match<br/>cost = 2"]
    end

    subgraph Fuzzy["MatchMode::Fuzzy threshold=0.85"]
        fast["Fast path:<br/>exact case-insensitive?"]
        jw["Jaro-Winkler similarity:<br/>JW('gonna','gona') = 0.95"]
        threshold{"0.95 >= 0.85?"}
        fuzzy_yes["YES match!<br/>cost = 0"]
    end

    words --> exact_cmp --> exact_no
    words --> ci_cmp --> ci_no
    words --> fast -->|"no"| jw --> threshold -->|"yes"| fuzzy_yes

    style Fuzzy fill:#e8f4e8
    style fuzzy_yes fill:#90ee90
```

**How Jaro-Winkler works:** Compares two strings by counting matching
characters within a distance window, penalizing transpositions, and
boosting for common prefixes. Returns a similarity score from 0.0
(completely different) to 1.0 (identical). Better than Levenshtein for
short words because it doesn't penalize length differences as harshly.

**Why fuzzy helps DP alignment:** Without fuzzy matching, a single ASR
substitution ("gonna" vs "gona") forces the DP to treat it as a gap
(cost 2) rather than a match (cost 0). This can cascade — the mismatched
word shifts subsequent alignments, potentially leaving entire utterances
unmatched. With fuzzy matching, the substitution is recognized as a match,
keeping the alignment anchored.

**Impact on DP cost matrix:**

```
Without fuzzy:            With fuzzy (0.85):
  g o n a                   g o n a
g [0 1 2 3]               g [0 _ _ _]  ← match at (0,0)
o [1 0 1 2]               o [_ _ _ _]
n [2 1 0 1]               n [_ _ _ _]
n [3 2 1 2]  ← mismatch   a [_ _ _ 0]  ← fuzzy match!
a [4 3 2 2]
```

The fuzzy match produces cost 0 for the "gonna"/"gona" pair, allowing
the DP to find a shorter path through the cost matrix.

**Default threshold (0.85):** Tuned empirically across 6 corpora and
59 files. Jaro-Winkler similarity examples at this threshold:

| Pair | JW | Match at 0.85? |
|------|-----|---------------|
| "gonna" / "gona" | 0.95 | Yes |
| "went" / "wen" | 0.94 | Yes |
| "yesterday" / "yestarday" | 0.92 | Yes |
| "mhm" / "mmhm" | 0.83 | No |
| "the" / "da" | 0.00 | No |
| "he" / "she" | 0.61 | No |
| "cat" / "dog" | 0.00 | No |

## Algorithmic optimizations (dp_align.rs)

The Hirschberg implementation includes two optimizations beyond the
textbook algorithm:

**Prefix/suffix stripping.** Before entering the O(mn) DP core,
`align()` strips matching prefixes and suffixes in O(n). For the primary
use case (WER/transcript comparison where accuracy is 80-95%), this
reduces the effective DP problem size by 10-100x. Only the differing
middle portion enters Hirschberg recursion.

**Generic `Alignable` trait.** Both `String` (word-level) and `char`
(character-level) entry points share a single generic implementation.
Monomorphization ensures zero overhead while eliminating ~200 lines of
duplicated code (4 pairs of copy-pasted functions unified into 4 generic
functions).

## UTR Strategy: Two-Pass Overlap-Aware Alignment

When the file has overlap markers (`+<` linkers or CA `⌊` markers), UTR
uses a two-pass strategy to handle backchannel timing recovery:

```mermaid
flowchart TD
    file["CHAT file with overlaps"]
    density{"Overlap density<br/>> 30%?"}

    subgraph Pass1["Pass 1: Global DP"]
        exclude["Exclude overlap utterances<br/>from word sequence"]
        include["Include ALL utterances<br/>(density too high to exclude)"]
        global["Hirschberg DP alignment<br/>(fuzzy matching 0.85)"]
    end

    subgraph Pass2["Pass 2: Backchannel Recovery"]
        find_pred["Find predecessor utterance"]
        ca_check{"Predecessor has<br/>CA marker ⌈?"}
        full_window["Search full predecessor<br/>time range"]
        narrow_window["Narrow search to<br/>onset position ± buffer"]
        windowed_dp["Small windowed DP<br/>on ASR tokens in window"]
    end

    subgraph Fallback["Best-of-Both Comparison"]
        compare{"Two-pass creates<br/>fewer FA groups?"}
        keep["Keep two-pass result"]
        revert["Revert to global result"]
    end

    file --> density
    density -->|"no (<= 30%)"| exclude --> global
    density -->|"yes (> 30%)"| include --> global
    global --> find_pred --> ca_check
    ca_check -->|"yes"| narrow_window --> windowed_dp
    ca_check -->|"no"| full_window --> windowed_dp
    windowed_dp --> compare
    compare -->|"no (equal or more)"| keep
    compare -->|"yes (fewer groups)"| revert

    style Pass1 fill:#e8e8f4
    style Pass2 fill:#e8f4e8
    style Fallback fill:#f4e8e8
```

### Configurable Parameters

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--utr-strategy` | `auto` | `auto` (detect overlaps), `global`, `two-pass` |
| `--utr-ca-markers` | `enabled` | Use ⌈⌉⌊⌋ for onset windowing |
| `--utr-density-threshold` | `0.30` | Max overlap fraction before skipping exclusion |
| `--utr-tight-buffer` | `500` | Pass-2 tight window ±ms |
| `--utr-fuzzy` | `0.85` | Jaro-Winkler similarity threshold |

### CA Marker Onset Estimation

When a predecessor utterance has ⌈ markers, the proportional position of
⌈ among the utterance's words estimates when overlap begins:

```mermaid
flowchart LR
    utt["*SPK: no pues de lo que sea ⌈tengo media hora⌉<br/>12660_15585"]

    frac["⌈ at word 7 of 10<br/>fraction = 0.6"]
    onset["onset = 12660 + 0.6 × (15585-12660)<br/>= 14415ms"]
    window["Search window:<br/>14415 ± 500ms<br/>(vs full 12660-15585)"]

    utt --> frac --> onset --> window

    style window fill:#90ee90
```

This narrows the pass-2 search window from the full predecessor range
(~3 seconds in this example) to ~1 second around the estimated onset —
roughly a 3x reduction in search space.

## Known DP Failure Modes

1. **Crossing alignments / rapid overlaps**  
   Global monotonic aligners cannot represent crossing matches, so one side is
   dropped or mis-assigned.

2. **Repeated-token ambiguity**  
   Repeated words create many equal-cost paths; deterministic tie-breaks may pick
   semantically wrong matches.

3. **ASR drift and hallucinations**  
   Large payload/reference divergence causes sparse matches and low timing
   coverage.

4. **Tokenization or normalization mismatch**  
   Char-level DP may align surprising spans when punctuation/case/tokenization
   differ.

5. **Temporal validity vs textual order**  
   Correct per-utterance times can still violate CHAT monotonicity when transcript
   order diverges from audio order.

## Existing Mitigations in Code

- **Monotonicity enforcement (E362)**: `enforce_monotonicity()` strips timing from
  regressions after FA.
- **Same-speaker overlap enforcement (E704)**:
  `strip_e704_same_speaker_overlaps()` strips earlier conflicting timing.
- **Untimed fallback windows**: proportional boundary estimates keep FA from
  skipping all untimed utterances.
- **Retokenize diagnostics + safe fallback**: invalid token mappings keep original
  words and mark parse taint.

## Plausible Next Step: Trouble-Window Escalation

There is a plausible hybrid option between "always global DP" and "never global
DP":

1. run a cheap whole-file anchor pass,
2. identify local divergence regions where coverage or ordering collapses,
3. run heavier DP only inside those trouble windows, and
4. keep the current global-DP path as the fallback when the detector is not
   trustworthy.

That design is promising for performance, especially on mostly-clean files with
small hand-edited regions. It is **not** implemented today. The current global
Hirschberg path remains the correctness reference, because a bad trouble-window
detector would simply reintroduce the same token-starvation and misassignment
class under a new name.

## Regression gate

DP-related behavior is protected by golden and policy tests. A typical focused
run sequence is:

```bash
uv run pytest batchalign/tests/golden/test_dp_golden.py -m golden -v -n 0
```

If intentionally changing behavior, update only the relevant expected files via:

```bash
uv run pytest batchalign/tests/golden/test_dp_golden.py -m golden -v -n 0 --update-golden
```
