# VOCD -- Vocabulary Diversity (D Statistic)

**Status:** Current
**Last updated:** 2026-05-22 23:32 EDT

## Purpose

Computes the D statistic for lexical diversity using bootstrap sampling of type-token ratios (TTR). The D statistic provides a more stable measure of vocabulary diversity than raw TTR because it accounts for sample size effects.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409241) for the original VOCD command specification.

## Usage

```bash
chatter clan vocd file.cha
chatter clan vocd --speaker CHI file.cha
chatter clan vocd --format json file.cha
```

## Options (chatter-native)

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `vocd` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/vocd/vocd.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` VOCD branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Vocd` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

VOCD has the **largest command-specific flag surface** of any CLAN
analysis tool — dominated by the D_optimum sampling parameters.

### VOCD-specific `+`-flags (from `vocd.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b0` | D_optimum: use split half — even | — | Missing | Sampling-method switch. |
| `+b1` | D_optimum: use split half — odd | — | Missing | |
| `+bsN` | D_optimum: size N of starting sample (default 35) | — | Missing | |
| `+blN` | D_optimum: size N of largest sample (default 50) | — | Missing | |
| `+biN` | D_optimum: size N of increments (default 1) | — | Missing | |
| `+bnN` | D_optimum: N number of samples (default 100) | — | Missing | |
| `+br` | D_optimum: random sampling *with* replacement (default: without) | — | Missing | |
| `+be` | D_optimum: sequential sampling | — | Missing | |
| `+c` / `+c0` / `+c1` | Find capitalised words only (`+c`/`+c0`) or mid-uppercase (`+c1`) | `--capitalization <initial\|mid>` | Done | Landed 2026-05-22. Shares the `CapitalizationFilter` enum with FREQ. `+c`/`+c0` (alias) → `initial` filter; `+c1` → `mid` (e.g. `McDonald`, `iPhone`). Applied before VOCD's token sequence reaches the D-statistic sampler. |
| `+d`, `+d1`, `+d2`, `+d3` | Output mode (utterances + types/tokens; summary only; etc.) | — | Rewriter only | `--display-mode N`; no consuming clap field. |
| `+gnS` / `-gnS` | Compute LRD — `S` = NUMERATOR `+s` directives | — | Missing | LRD = Limiting Relative Diversity. |
| `+gdS` / `-gdS` | Compute LRD — `S` = DENOMINATOR `+s` directives | — | Missing | |
| `+o` | Override default lemma-based analysis | — | Missing | Switches VOCD from `%mor` lemma-based to main-tier word-based. |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | |

### General `+`-flags VOCD inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+s"word"` / `-s"word"` | Include/exclude word | `--include-word` / `--exclude-word` | Done | VOCD's `+sm` suffix variants for `%mor` searches not special-cased. |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | Gem filter (without N suffix) | `--gem` | Done | Distinct from `+gnS`/`+gdS`. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | |
| `+re` | Recurse | (default) | Done | |
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`. Two-layer fix: `WordFilter::case_sensitive` controls `--include-word`/`--exclude-word` pattern matching; the token-stream layer in `process_utterance` skips the default `text.to_lowercase()` so the D-statistic sampler sees case variants as distinct types. Pinned by `vocd_case_sensitive_preserves_case_in_tokens` and `vocd_default_lowercases_tokens`. |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 9 |
| Partial | 2 |
| Rewriter only | 6 |
| Missing | 12 |

VOCD has the **largest Missing bucket** of any audited command,
driven by the D_optimum sampling parameter family (`+b0`, `+b1`,
`+bsN`, `+blN`, `+biN`, `+bnN`, `+br`, `+be`) and the LRD family
(`+gnS`, `+gdS`). chatter's `vocd` uses a fixed default sampling
strategy with no exposed knobs. The `+o` lemma-vs-main-tier
override is also missing — researchers studying main-tier
diversity directly cannot get there from chatter today. Filed as
Phase 1.7 follow-ups.

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
