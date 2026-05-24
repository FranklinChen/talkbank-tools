# SUGAR -- Sampling Utterances and Grammatical Analysis Revised

**Status:** Current
**Last updated:** 2026-05-22 09:22 EDT

## Purpose

Computes language sample analysis metrics from `%mor` and `%gra` tiers, providing a quick clinical assessment of grammatical complexity. SUGAR is designed as a time-efficient alternative to more detailed scoring systems like DSS or IPSYN.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409287) for the original SUGAR command specification.

## Usage

```bash
chatter clan sugar file.cha
chatter clan sugar --speaker CHI file.cha
chatter clan sugar --format json file.cha
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

Authoritative enumeration of every CLAN `sugar` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/sugar.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` SUGAR branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Sugar` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

SUGAR is a **required-flag refusal** command in chatter — same
refusal byte-parity as EVAL/KIDEVAL/DSS/IPSYN.

### SUGAR-specific `+`-flags (from `sugar.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+aN` | Minimum-utterance threshold (default 50) | `--min-utterances N` | Done | Fixed 2026-05-22. New clap field on `Sugar`; rewriter routes `+aN` → `--min-utterances N`. |
| `+bS` / `-bS` | Morpheme-delimiter customization | — | Missing | Shared with WDLEN/MAXWD/EVAL. |
| `-d` | Show utterances that fail at least one verb requirement | — | Missing | Diagnostic listing. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 0 |
| Rewriter only | 5 |
| Missing | 7 |

SUGAR's narrowest gap was `+aN`: closed 2026-05-22 with a new
`--min-utterances N` clap field and rewriter routing.

SUGAR has no command-specific flags beyond the shared
`CommonAnalysisArgs` set. The minimum-utterance threshold (50) is a
fixed internal default; there is no `--min-utterances` switch in the
current CLI.

## Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| **MLU-S** | Mean Length of Utterance in morphemes | `%mor` tier |
| **TNW** | Total Number of Words (tokens with POS tags) | `%mor` tier |
| **WPS** | Words Per Sentence | Utterances containing verbs |
| **CPS** | Clauses Per Sentence | `%gra` subordination relations |

## Algorithm

1. For each utterance, count morphemes and words from `%mor`
2. Detect **verb-containing utterances** using POS tags: `v`, `verb`, `cop`, `aux`, `mod`, `part` (`crates/talkbank-clan/src/commands/sugar.rs:198` — both `v` and the longer `verb` form are accepted)
3. For verb utterances with `%gra`, count **subordinate clauses** via grammatical relations (`COMP`, `CSUBJ`, `CMOD`, etc.)
4. Compute per-speaker ratios at finalization:
   - WPS = total words / number of verb utterances
   - CPS = total clauses / number of verb utterances

### Minimum utterance threshold

If a speaker has fewer than `min_utterances` (default: 50), the sample is flagged as insufficient. This ensures statistical reliability of the computed ratios.

## Differences from CLAN

### Verb detection

Uses mapped POS tags from the parsed `%mor` tier structure. CLAN may use a slightly different POS tag set for verb identification. Both implementations identify the same core verb categories.

Post-clitic `%mor` chunks are included in verb detection, so clitic-bearing items still contribute when the verb-like chunk appears only after `~`.

### Clause counting

Uses `%gra` subordination relations only (dependency structure). CLAN's clause detection may use additional heuristics beyond grammatical relations.

### Morpheme counting

Morpheme counts are computed from typed `%mor` structure, including post-clitics and their features.

### Minimum utterance threshold

The Rust implementation uses a fixed internal default of 50, matching
CLAN. There is currently no CLI flag to override this.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

Verified against CLAN C binary output.
