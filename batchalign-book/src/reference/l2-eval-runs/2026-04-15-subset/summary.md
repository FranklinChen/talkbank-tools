# L2 Morphotag Aggregate Evaluation

**Eval set:** `<eval-set-subset>.jsonl` (operator-specific corpus manifest, kept out-of-tree)
**Morphotag output:** `/tmp/l2-eval-subset-out`

- Files: **12**
- `@s` words: **3278**
- Aggregate splice rate: **98.9%** (gate ≥99%: FAIL)
- Aggregate heuristic-clean rate: **97.8%** (gate ≥90%: PASS)

## Per-pair

| Pair | Files | @s | Spliced | Splice rate | Clean | Clean rate | Gate |
|------|------:|---:|--------:|------------:|------:|-----------:|:----:|
| `cat,spa` | 3 | 308 | 308 | 100.0% | 299 | 97.1% | PASS |
| `deu,eng` | 3 | 477 | 473 | 99.2% | 470 | 98.5% | PASS |
| `eng,spa` | 3 | 2420 | 2389 | 98.7% | 2363 | 97.6% | FAIL |
| `fra,nld` | 3 | 73 | 73 | 100.0% | 73 | 100.0% | PASS |

## Flag distribution

| Flag | Count | % of @s |
|------|------:|--------:|
| `FeaturePosMismatch` | 29 | 0.9% |
| `MissingMor` | 35 | 1.1% |
| `PropnForFunctionWord` | 9 | 0.3% |

## Reproducing this evaluation

```bash
# 1. Build the eval corpus (selector script is kept out-of-tree —
#    it reads corpus paths from the operator's data drive).

# 2. Run morphotag on the eval set (one invocation per primary language)
#    — see scripts/l2-eval/run_morphotag.py for the driver.

# 3. Run the evaluator
batchalign3 eval l2-morphotag \
    --eval-set <eval-set-subset>.jsonl \
    --morphotag-output /tmp/l2-eval-subset-out \
    --output <report-dir>/
```
