# TRNFIX — Compare Two Dependent Tiers

**Status:** Current
**Last updated:** 2026-05-22 09:49 EDT

## Purpose

Compares two dependent tiers (default: `%mor` and `%trn`) word-by-word across all utterances, reporting unique mismatch pairs with frequency counts and an overall accuracy percentage. Useful for verifying tier consistency after automatic annotation or manual correction.

When tiers have different lengths for a given utterance, missing positions are represented as the null symbol `∅` (empty set).

## Usage

```bash
chatter clan trnfix file.cha
chatter clan trnfix file.cha --tier1 mor --tier2 gra
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--tier1 <name>` | `+bS` (first instance) | First tier to compare (default: `mor`) |
| `--tier2 <name>` | `+bS` (second instance) | Second tier to compare (default: `trn`) |
| `--speaker <code>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <code>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--format <fmt>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

### TRNFIX-specific `+`-flags (from `trnfix.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Disambiguate words before compare (default: compare whole words) | — | Missing | Affects mismatch resolution semantics. |
| `+bS` | Specify a tier to compare (repeatable; first → tier1, second → tier2) | `--tier1` / `--tier2` | Partial | chatter splits into two explicit fields rather than positional `+b` semantics. |
| `+d` | Include speaker tier in output | — | Rewriter only | |
| `+d1` | `+d` + include utterances in mismatches summary file | — | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 2 |
| Rewriter only | 4 |
| Missing | 3 |

TRNFIX's `+a` disambiguate-before-compare is the most semantically
significant gap: it changes whether multi-analysis `%mor` tokens
like `det|the^pro|the` are compared by their first analysis only
or by full text.

## Output

- Table of unique mismatch pairs with frequency counts
- Total items compared
- Total mismatched items
- Accuracy percentage

## Differences from CLAN

- Tier content is compared from parsed AST data rather than raw text
- `%trn` is treated as an alias of `%mor`, and `%grt` as an alias of `%gra`
- `%mor`/`%gra` token comparison preserves typed token boundaries from the AST rather than comparing whitespace-split serialized payload strings.
- Length mismatches are handled with explicit `∅` null symbols
- Configurable tier names (CLAN uses fixed `%mor`/`%trn` comparison)
- Output supports text, JSON, and CSV formats
- **Golden test parity**: Verified against CLAN C binary output
