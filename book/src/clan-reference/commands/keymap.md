# KEYMAP — Contingency Tables for Coded Data

**Status:** Current
**Last updated:** 2026-05-22 09:48 EDT

## Purpose

Builds contingency tables for coded interactional data. The legacy manual describes `KEYMAP` as choosing initiating or beginning codes on a specific coding tier, then examining all codes on that same tier in the next utterance.

In `talkbank-clan`, given a set of keyword codes, `KEYMAP` tracks each keyword occurrence on a specified coding tier and records what code items appear in the immediately following utterance, broken down by speaker.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409207) for the original KEYMAP command specification.

## Usage

```bash
chatter clan keymap file.cha --keyword code1 --keyword code2
chatter clan keymap file.cha --keyword code1 --tier spa
chatter clan keymap file.cha -k code1 -k code2          # short form
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `-k, --keyword <code>` | `+bS` | Primary code to track (required, repeatable) |
| `--tier <name>` | `+t%X` (rewriter target) | Tier label to read codes from (default: `cod`) |
| `--speaker <code>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <code>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--format <fmt>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

### KEYMAP-specific `+`-flags (from `keymap.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bS` / `+b@F` | Set key code(s) to `S` (or codes in file `@F`) | `-k / --keyword` (only the inline `S` form) | Partial | Inline `+bS` is now routed (since 2026-05-22) — rewriter sends `--keyword S`. File-list form `+b@F` still missing; it passes through unrewritten so clap rejects loudly. |
| `+cS` / `+c@F` | Set complimentary key code to `S` (or file `@F`) | — | Missing | Pair-completion key for the contingency table. |
| `+d` | Output in spreadsheet format | — | Rewriter only | |
| `+o` | Include codes that precede target code(s) | — | Missing | Two-sided contingency (before + after). |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 1 |
| Rewriter only | 4 |
| Missing | 5 |

KEYMAP's largest gap is the **complimentary code** (`+cS`):
without it, the contingency table is one-sided — chatter tracks
what follows each keyword, but CLAN's KEYMAP also pairs keywords
with their complimentary codes for a true cross-tabulation. The
`+o` two-sided mode (codes before AND after the target) is also
missing.

## Output

Per speaker per keyword:

- Total keyword occurrences
- Following codes with speaker attribution and frequency counts

## Differences from CLAN

- **Manual intent**: The legacy manual explicitly treats `KEYMAP` as a coding-tier command and says that only symbols beginning with `$` are considered on that tier; all other strings are ignored.
- Code extraction for `%cod` now uses a clan-local semantic `%cod` item layer derived from the parsed AST rather than flattened tier text
- **Selector handling**: `%cod` selectors such as `<w4>` and `<w4-5>` are treated as item scope, not as stand-alone codes, when deriving keyword and following-code items.
- **Manual constraint not yet fully enforced**: `KEYMAP` currently retains a generic non-`%cod` tier fallback. The manual suggests tighter coding-tier semantics than that fallback provides.
- Keyword matching is case-insensitive by default
- Output supports text, JSON, and CSV formats
- Deterministic ordering via `BTreeMap`
- **Golden test parity**: Verified against CLAN C binary output
