# L2 Morphotag Aggregate Evaluation

**Eval set:** `docs/l2-eval-batchalign3/data/eval-set.jsonl`
**Morphotag output:** `/tmp/l2-eval-post-mwt-audit`

- Files: **54**
- `@s` words: **16845**
- Aggregate **dispatch rate**: **99.96%** (gate ≥99%: PASS) — counts only `L2|xxx` fallbacks against the feature.
- Aggregate splice rate: 99.96% — includes `missing_mor` as failures (pessimistic floor; AST walker should keep this near 0).
- Aggregate heuristic-clean rate: **98.0%** (gate ≥90%: PASS)

## Per-pair

| Pair | Files | @s | Dispatch | Splice | L2\|xxx | MissingMor | Clean | Gate |
|------|------:|---:|---------:|-------:|--------:|-----------:|------:|:----:|
| `ara,nld` | 3 | 806 | 100.00% | 100.0% | 0 | 0 | 98.4% | PASS |
| `cat,hun,spa` | 3 | 1160 | 100.00% | 100.0% | 0 | 0 | 99.2% | PASS |
| `cat,spa` | 3 | 308 | 100.00% | 100.0% | 0 | 0 | 97.1% | PASS |
| `cym,eng` | 3 | 3237 | 99.81% | 99.8% | 6 | 0 | 97.0% | PASS |
| `cym,eng,spa` | 1 | 1106 | 100.00% | 100.0% | 0 | 0 | 97.4% | PASS |
| `dan,eng` | 2 | 441 | 100.00% | 100.0% | 0 | 0 | 99.1% | PASS |
| `deu,eng` | 3 | 475 | 100.00% | 100.0% | 0 | 0 | 99.2% | PASS |
| `deu,ita` | 3 | 289 | 100.00% | 100.0% | 0 | 0 | 100.0% | PASS |
| `eng,fra` | 3 | 344 | 100.00% | 100.0% | 0 | 0 | 98.5% | PASS |
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
| `PropnForFunctionWord` | 7 | 0.0% |
| `FeaturePosMismatch` | 329 | 2.0% |

## Per-utterance outcome distribution

Across **45923 utterances** (all pairs, all files):

- `Aligned`: 42771 (93.1%) — `%mor` matched CHAT alignable count.
- `NotApplicable`: 3152 (6.9%) — no Mor-alignable content; correctly no `%mor`.
- `CountMismatchInFile`: 0 (0.0%) — `%mor` size ≠ alignable count. Post-fix this should be 0.
- `PipelineAbsorbedFailure`: 0 (0.0%) — alignable content present but no `%mor`; pipeline absorbed a `MisalignmentBug`.
- **Anomaly rate: 0.00%** (sum of last two; non-zero means a bug to investigate).

## Reproducing this evaluation

```bash
batchalign3 eval l2-morphotag \
    --eval-set docs/l2-eval-batchalign3/data/eval-set.jsonl \
    --morphotag-output /tmp/l2-eval-post-mwt-audit \
    --output <report-dir>/
```
