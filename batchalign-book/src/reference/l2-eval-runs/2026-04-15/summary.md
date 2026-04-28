# L2 Morphotag Aggregate Evaluation

**Status:** Current
**Last updated:** 2026-04-15 17:58 EDT
**Analyzer:** `batchalign3 eval l2-morphotag` (Rust, AST-first via `walk_words(TierDomain::Mor)`)
**Eval set:** `docs/l2-eval/eval-set.jsonl`
**Morphotag output:** `/tmp/l2-eval-full-out`

> This run replaced the 2026-04-15 figures produced by the now-deleted
> `scripts/l2-eval/analyze.py` regex analyzer. The Rust analyzer
> walks the typed CHAT AST and eliminates the ~2% `missing_mor` noise
> the regex tool produced under retrace markers. Net effect on these
> numbers: splice rate rose from 97.7% to 99.9%, heuristic-clean rose
> from 95.6% to 98.0%, and `missing_mor` cases fell from 310 to 3.
> Every pair still passes the pre-registered gates.

- Files: **54**
- `@s` words: **16845**
- Aggregate **dispatch rate**: **99.96%** (gate ≥99%: PASS) — counts only `L2|xxx` fallbacks against the feature.
- Aggregate splice rate: 99.94% — includes `missing_mor` as failures (pessimistic floor; AST walker should keep this near 0).
- Aggregate heuristic-clean rate: **98.0%** (gate ≥90%: PASS)

## Per-pair

| Pair | Files | @s | Dispatch | Splice | L2\|xxx | MissingMor | Clean | Gate |
|------|------:|---:|---------:|-------:|--------:|-----------:|------:|:----:|
| `ara,nld` | 3 | 806 | 100.00% | 100.0% | 0 | 0 | 98.5% | PASS |
| `cat,hun,spa` | 3 | 1160 | 100.00% | 100.0% | 0 | 0 | 99.2% | PASS |
| `cat,spa` | 3 | 308 | 100.00% | 100.0% | 0 | 0 | 97.1% | PASS |
| `cym,eng` | 3 | 3237 | 99.81% | 99.8% | 6 | 0 | 97.0% | PASS |
| `cym,eng,spa` | 1 | 1106 | 100.00% | 100.0% | 0 | 0 | 97.4% | PASS |
| `dan,eng` | 2 | 441 | 100.00% | 100.0% | 0 | 0 | 99.1% | PASS |
| `deu,eng` | 3 | 475 | 100.00% | 100.0% | 0 | 0 | 99.2% | PASS |
| `deu,ita` | 3 | 289 | 100.00% | 100.0% | 0 | 0 | 100.0% | PASS |
| `eng,fra` | 3 | 344 | 100.00% | 99.1% | 0 | 3 | 97.7% | PASS |
| `eng,hrv` | 3 | 531 | 100.00% | 100.0% | 0 | 0 | 99.4% | PASS |
| `eng,jpn` | 3 | 395 | 100.00% | 100.0% | 0 | 0 | 100.0% | PASS |
| `eng,por` | 3 | 838 | 100.00% | 100.0% | 0 | 0 | 99.9% | PASS |
| `eng,spa` | 3 | 2353 | 100.00% | 100.0% | 0 | 0 | 99.0% | PASS |
| `eng,yue` | 3 | 700 | 100.00% | 100.0% | 0 | 0 | 98.9% | PASS |
| `eng,yue,zho` | 3 | 868 | 99.88% | 99.9% | 1 | 0 | 98.2% | PASS |
| `eng,zho` | 3 | 28 | 100.00% | 100.0% | 0 | 0 | 100.0% | PASS |
| `eus,spa` | 3 | 1453 | 100.00% | 100.0% | 0 | 0 | 96.4% | PASS |
| `fin,swe` | 3 | 1441 | 100.00% | 100.0% | 0 | 0 | 95.2% | PASS |
| `fra,nld` | 3 | 72 | 100.00% | 100.0% | 0 | 0 | 100.0% | PASS |

## Flag distribution

| Flag | Count | % of @s |
|------|------:|--------:|
| `L2Xxx` | 7 | 0.0% |
| `MissingMor` | 3 | 0.0% |
| `PropnForFunctionWord` | 7 | 0.0% |
| `FeaturePosMismatch` | 328 | 1.9% |

## Reproducing this evaluation

```bash
batchalign3 eval l2-morphotag \
    --eval-set <eval-set>.jsonl \
    --morphotag-output /tmp/l2-eval-full-out \
    --output <report-dir>/
```
