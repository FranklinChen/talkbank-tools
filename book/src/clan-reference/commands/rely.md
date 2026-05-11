# RELY — Inter-Rater Reliability (Cohen's Kappa)

**Status:** Current
## Purpose

Compares two parallel CHAT files for coder agreement. The legacy manual gives `RELY` five functions: coder agreement, Cohen's kappa, student-vs-master evaluation, rough transcript overlap on the main line, and selective dependent-tier merging.

The current `talkbank-clan` implementation focuses on the coding-tier comparison use case: it compares coded data on a specified dependent tier (default `%cod`) across two files to compute per-code agreement statistics, overall agreement percentage, and Cohen's kappa coefficient.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409232) for the original RELY command specification.

## Usage

```bash
chatter clan rely file1.cha file2.cha
chatter clan rely file1.cha file2.cha --tier spa
```

## Algorithm

1. Parse both input files and extract codes per utterance from the specified tier
2. Align utterances by position (index)
3. For each aligned pair, count per-code agreements (minimum of the two counts for each code in that utterance)
4. Compute overall observed agreement (Po) and expected agreement (Pe) for Cohen's kappa: `k = (Po - Pe) / (1 - Pe)`

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--tier <name>` | — | Tier label to compare (default: `cod`) |
| `--format <fmt>` | — | Output format: text, json, csv |

## Output

- Per-code agreement statistics (count in each file, agreed count, agreement percentage)
- Overall agreement percentage
- Cohen's kappa coefficient

## Differences from CLAN

- RELY requires two-file input and does not use the standard `AnalysisCommand` trait; it is invoked directly
- **Manual intent**: The legacy manual gives special semantics for coding tiers such as `%cod` and `%spa`, and documents `+c1` as comparing only the main part of a colon-delimited code.
- Code extraction for `%cod` now uses a clan-local semantic `%cod` item layer derived from the parsed AST
- **Selector handling**: `%cod` selectors such as `<w4>` and `<w4-5>` are preserved as item scope, not counted as compared code values.
- **Scope reduction**: The current implementation does not yet cover all five legacy `RELY` functions described in the manual, and it does not yet implement the documented `+c1` colon-prefix comparison mode.
- Output supports text, JSON, and CSV formats
- **Golden test parity**: Verified against CLAN C binary output
