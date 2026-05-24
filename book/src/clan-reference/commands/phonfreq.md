# PHONFREQ — Phonological Frequency Analysis

**Status:** Current
**Last updated:** 2026-05-22 09:40 EDT

## Purpose

Counts individual phone (character) occurrences from `%pho` tier content, tracking positional distribution within each phonological word: initial (first character), final (last character), and other (middle positions). Counts alphabetic characters (Unicode, including IPA) plus the `+` compound marker; stress marks (`ˈ`, `ˌ`), length marks (`ː`), digits, and other non-letter symbols are skipped (`crates/talkbank-clan/src/commands/phonfreq.rs:178`).

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409227) for the original PHONFREQ command specification.

## Usage

```bash
chatter clan phonfreq file.cha
chatter clan phonfreq file.cha --speaker CHI
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `phonfreq` flag. Sources:

* `OSX-CLAN/src/clan/phonfreq.cpp` — `usage()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` PHONFREQ branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Phonfreq` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

PHONFREQ has the narrowest command-specific surface — a single
flag.

### PHONFREQ-specific `+`-flags (from `phonfreq.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bS` | Set phonological tier name to `S` (default `%pho`) | — | Missing | chatter hard-codes `%pho`. Users with non-standard phonological tiers (e.g. `%xpho`) cannot redirect. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 6 |
| Partial | 1 |
| Rewriter only | 4 |
| Missing | 7 |

PHONFREQ's body parity already matches CLAN byte-for-byte
(Unicode-block sort + byte-width column padding landed in the
squash). The remaining gap is the `+bS` tier-name selector — a
one-line clap field addition.

## Output

Per-phone frequency with positional breakdown (initial/final/other), sorted alphabetically by phone character.

## Differences from CLAN

- Phone extraction uses parsed `%pho` tier structure from the AST rather than raw text character scanning
- Positional classification operates on typed `PhoWord` content
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
