# EVAL-D -- Language Sample Evaluation (DementiaBank)

**Status:** Current
**Last updated:** 2026-05-22 09:21 EDT

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

## CLAN `+`-flag coverage audit

EVAL-D's `+`-flag surface is **byte-identical** to EVAL's. The
only divergence is the normative-comparison database name
embedded in `+dS` choices (DementiaBank cohorts instead of
AphasiaBank cohorts). The coverage matrix, status counts, and
audit conclusions for EVAL-D are the same as
[EVAL](./eval.md#clan--flag-coverage-audit).

EVAL-D is a **required-flag refusal** command in chatter — same
as EVAL — emitting the exact CLAN refusal message when `+t*X` is
absent.

The DementiaBank database-comparison engine is missing for the
same reason EVAL's AphasiaBank engine is missing: chatter does
not bundle the `.cut` normative databases.

## Differences from CLAN

- Same as EVAL -- see [EVAL differences](eval.md#differences-from-clan).
- The `EvalVariant::Dialect` config flag selects DementiaBank norms automatically.

## Implementation

EVAL-D is not a separate command module. It reuses `EvalCommand` with `EvalConfig { variant: EvalVariant::Dialect, .. }`. The variant determines which `.cut` database directory to use for normative comparison.
