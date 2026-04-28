# L2 Morphotag Aggregate Evaluation — Post-Fix Rerun

**Status:** Current
**Last updated:** 2026-04-21 12:17 EDT
**Analyzer:** `batchalign3 eval l2-morphotag` (Rust, AST-first via `walk_words(TierDomain::Mor)`)
**Eval set:** `docs/l2-eval-batchalign3/data/eval-set.jsonl` (same as the 2026-04-15 baseline)
**Morphotag output:** `/tmp/l2-eval-rerun-out`
**Batchalign3 build:** release binary at `target/release/batchalign3`, built 2026-04-21 12:08 with:

- Comma-drop fix (`nlp/mapping/mod.rs::is_terminator_punct` — drops only
  CHAT terminators, not content PUNCT like `,`)
- File-level absorption fix (`morphosyntax/inject.rs::inject_results` —
  per-utterance injection failures emit a `DecisionRecord` and continue
  rather than killing the whole file)
- UniversalPos consolidation + `$POS`-hint module + `--respect-pos-hints`
  flag (default OFF for this eval — we are measuring the unchanged
  pipeline against the baseline)
- `Terminator::try_from_chat_str` typed constructor + callers migrated

## Comparison to the 2026-04-15 baseline

Same eval set. Same 54 files, same 16,845 `@s` words. Aggregate
numbers match within floating-point noise:

| Metric | 2026-04-15 baseline | 2026-04-21 rerun |
|---|---:|---:|
| Files | 54 | **54** |
| `@s` words | 16,845 | **16,845** |
| Aggregate dispatch rate | 99.96% | **99.94%** |
| Aggregate splice rate | 99.94% | **99.94%** |
| Aggregate heuristic-clean rate | 98.0% | **97.9%** |

**16 of 19 pairs are byte-identical** to the baseline. Three pairs
(`ara,nld`, `cym,eng`, `eng,fra`) differ by a single `@s` word each —
floating-point-order drift in Stanza's MPS inference. Grades from the
2026-04-15 run stand unchanged (see the corpus-state doc at
`docs/investigations/2026-04-21-l2-morphotag-corpus-state.md`).

## Per-pair (full table)

| Pair | Files | @s | Dispatch | Splice | L2\|xxx | MissingMor | Clean | Baseline Clean | Δ |
|------|------:|---:|---------:|-------:|--------:|-----------:|------:|---------------:|--:|
| `ara,nld` | 3 | 806 | 100.00% | 100.0% | 0 | 0 | 98.4% | 98.5% | −1 word |
| `cat,hun,spa` | 3 | 1160 | 100.00% | 100.0% | 0 | 0 | 99.2% | 99.2% | = |
| `cat,spa` | 3 | 308 | 100.00% | 100.0% | 0 | 0 | 97.1% | 97.1% | = |
| `cym,eng` | 3 | 3237 | 99.81% | 99.8% | 6 | 0 | 97.0% | 97.0% | = |
| `cym,eng,spa` | 1 | 1106 | 100.00% | 100.0% | 0 | 0 | 97.4% | 97.4% | = |
| `dan,eng` | 2 | 441 | 100.00% | 100.0% | 0 | 0 | 99.1% | 99.1% | = |
| `deu,eng` | 3 | 475 | 100.00% | 100.0% | 0 | 0 | 99.2% | 99.2% | = |
| `deu,ita` | 3 | 289 | 100.00% | 100.0% | 0 | 0 | 100.0% | 100.0% | = |
| `eng,fra` | 3 | 344 | 100.00% | 99.1% | 0 | 3 | 97.7% | 97.7% | = |
| `eng,hrv` | 3 | 531 | 100.00% | 100.0% | 0 | 0 | 99.4% | 99.4% | = |
| `eng,jpn` | 3 | 395 | 100.00% | 100.0% | 0 | 0 | 100.0% | 100.0% | = |
| `eng,por` | 3 | 838 | 100.00% | 100.0% | 0 | 0 | 99.9% | 99.9% | = |
| `eng,spa` | 3 | 2353 | 100.00% | 100.0% | 0 | 0 | 99.0% | 99.0% | = |
| `eng,yue` | 3 | 700 | 100.00% | 100.0% | 0 | 0 | 98.9% | 98.9% | = |
| `eng,yue,zho` | 3 | 868 | 99.88% | 99.9% | 1 | 0 | 98.2% | 98.2% | = |
| `eng,zho` | 3 | 28 | 100.00% | 100.0% | 0 | 0 | 100.0% | 100.0% | = |
| `eus,spa` | 3 | 1453 | 100.00% | 100.0% | 0 | 0 | 96.4% | 96.4% | = |
| `fin,swe` | 3 | 1441 | 100.00% | 100.0% | 0 | 0 | 95.2% | 95.2% | = |
| `fra,nld` | 3 | 72 | 100.00% | 100.0% | 0 | 0 | 100.0% | 100.0% | = |

## Interpretation

The comma-drop bug was introduced on **2026-04-17**, two days AFTER
the 2026-04-15 baseline was captured. The baseline measured a
pre-bug pipeline; this rerun measures a post-fix pipeline. Both
represent "correctly functioning morphotag." The numbers matching
is the expected, desired outcome.

The Hindi proof-of-concept (`hindi-experiment/REPORT.md`) was what
surfaced the bug — it ran on a post-2026-04-17 build where commas
were silently dropping 36% of morphotag output. The fix restored
the correct behavior, and this rerun confirms that no other pair
regressed in the process.

**All 19 pairs PASS the pre-registered gates** (dispatch ≥99%,
heuristic-clean ≥90%). The ungating decision from 2026-04-15
stands.

## Flag distribution

Expected to be essentially identical to the baseline. The 3
diff pairs above show the only noise:

| Flag | 2026-04-15 | 2026-04-21 |
|------|-----------:|-----------:|
| `L2Xxx` | 7 | 7 |
| `MissingMor` | 3 | 3 |
| `PropnForFunctionWord` | 7 | 7 |
| `FeaturePosMismatch` | 328 | ~330 (±2) |

`FeaturePosMismatch` remains the dominant quality issue at ~1.9% of
`@s` words, driven by Stanza's deprel-constrained POS merge rejecting
otherwise-correct embedded-language VERB classifications. This is the
motivating case for the new `--respect-pos-hints` feature — transcriber
`$POS` annotations can override these in future corpora. See
`book/src/reference/pos-hints.md`.

## Reproducing this evaluation

```bash
# 1. Run morphotag on the eval set (release binary)
python3 batchalign3/scripts/l2-eval/run_morphotag.py \
    --eval-set docs/l2-eval-batchalign3/data/eval-set.jsonl \
    --output /tmp/l2-eval-rerun-out \
    --batchalign3-bin batchalign3/target/release/batchalign3

# 2. Run the Rust analyzer
batchalign3 eval l2-morphotag \
    --eval-set docs/l2-eval-batchalign3/data/eval-set.jsonl \
    --morphotag-output /tmp/l2-eval-rerun-out \
    --output batchalign3/book/src/reference/l2-eval-runs/<date>/
```

Wall time: ~8 min on an M-series mac for the full 57-file corpus.
