# VOCD -- Vocabulary Diversity (D Statistic)

**Status:** Current
**Last updated:** 2026-04-28 23:16 EDT

## Purpose

Computes the D statistic for lexical diversity using bootstrap sampling of type-token ratios (TTR). The D statistic provides a more stable measure of vocabulary diversity than raw TTR because it accounts for sample size effects.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409241) for the original VOCD command specification.

## Usage

```bash
chatter clan vocd file.cha
chatter clan vocd --speaker CHI file.cha
chatter clan vocd --format json file.cha
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--format <FMT>` | -- | Output format: text, json, csv, clan |

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `vocd file.cha` | `chatter clan vocd file.cha` |
| `vocd +t*CHI file.cha` | `chatter clan vocd file.cha --speaker CHI` |

## Algorithm

### Overview

VOCD fits empirical type-token data to a theoretical curve, finding the D parameter that best explains the relationship between sample size and vocabulary diversity.

### Steps

1. **Collect tokens**: Gather all countable word tokens per speaker from the main tier
2. **Bootstrap sampling** (3 independent trials):
   - For each sample size `N` from 35 through 50:
     - Draw 100 random samples of N tokens (without replacement)
     - Compute mean TTR across the 100 samples
   - Fit the empirical (N, TTR) curve to the theoretical D-curve using gradient-descent least-squares optimization
   - Record the optimal D value
3. **Report**: Per-trial D values and their average

### Theoretical TTR Curve

```text
TTR(N) = (D/N) * [sqrt(1 + 2*N/D) - 1]
```

This models the expected type-token ratio for a sample of size N given a lexical diversity parameter D. Higher D means greater diversity.

### Interpretation

| D value | Interpretation |
|---------|---------------|
| < 30 | Low lexical diversity |
| 30-70 | Typical range for young children |
| 70-100 | Typical range for older children/adults |
| > 100 | High lexical diversity |

(Values are approximate and depend on the population.)

## Output

Per-speaker D statistic with per-trial breakdown tables:

```text
Speaker: CHI
  Trial 1:
    N    samples  TTR     std_dev   D
    35   100      0.743   0.045     42.1
    36   100      0.735   0.043     42.3
    ...
    50   100      0.680   0.038     43.0
    D = 42.5
  Trial 2: D = 41.8
  Trial 3: D = 43.2
  Average D = 42.5
```

## Differences from CLAN

### Stochastic variation

Because VOCD uses random sampling, D values may differ slightly between runs and between CLAN and our implementation. This is expected behavior, not a bug. Differences of +/- 5 are normal.

### Fusional feature stripping

Fusional features (`&PRES`, `&INF`, etc.) are stripped from lemmas in `%mor` echo output. This ensures clean lemma display when VOCD echoes the morphological tier for insufficient-token warnings.

### Word identification

Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching. Token collection operates on parsed AST content rather than raw text.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output (within expected stochastic variation).
