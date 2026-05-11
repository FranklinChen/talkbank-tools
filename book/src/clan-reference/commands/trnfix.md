# TRNFIX — Compare Two Dependent Tiers

**Status:** Current
## Purpose

Compares two dependent tiers (default: `%mor` and `%trn`) word-by-word across all utterances, reporting unique mismatch pairs with frequency counts and an overall accuracy percentage. Useful for verifying tier consistency after automatic annotation or manual correction.

When tiers have different lengths for a given utterance, missing positions are represented as the null symbol `∅` (empty set).

## Usage

```bash
chatter clan trnfix file.cha
chatter clan trnfix file.cha --tier1 mor --tier2 gra
```

## Options

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--tier1 <name>` | — | First tier to compare (default: `mor`) |
| `--tier2 <name>` | — | Second tier to compare (default: `trn`) |
| `--speaker <code>` | `+t*CODE` | Restrict to specific speaker |
| `--format <fmt>` | — | Output format: text, json, csv |

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
