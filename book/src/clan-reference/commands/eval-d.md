# EVAL-D -- Language Sample Evaluation (DementiaBank)

**Status:** Current
## Purpose

EVAL-D is a variant of [EVAL](eval.md) used for data collected with the DementiaBank protocol. The analysis logic is identical -- only the normative comparison database differs.

EVAL uses AphasiaBank norms; EVAL-D uses DementiaBank norms.

## Usage

```bash
chatter clan eval-d file.cha
chatter clan eval-d --speaker PAR file.cha
chatter clan eval-d --format json file.cha
```

## Options

All options are identical to [EVAL](eval.md).

## Differences from CLAN

- Same as EVAL -- see [EVAL differences](eval.md#differences-from-clan).
- The `EvalVariant::Dialect` config flag selects DementiaBank norms automatically.

## Implementation

EVAL-D is not a separate command module. It reuses `EvalCommand` with `EvalConfig { variant: EvalVariant::Dialect, .. }`. The variant determines which `.cut` database directory to use for normative comparison.
