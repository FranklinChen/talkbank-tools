# DIST -- Word Distribution Across Turns

**Status:** Current
**Last updated:** 2026-05-26 10:59 EDT

## Purpose

Counts turns and tracks for each word the first and last turn in which it appears. DIST is part of the FREQ family of commands and is useful for studying when words first appear and how their usage is distributed across a conversation.

## Usage

```bash
chatter clan dist file.cha
chatter clan dist --speaker CHI file.cha
chatter clan dist --format json file.cha
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

Authoritative enumeration of every CLAN `dist` flag. Sources:

* `OSX-CLAN/src/clan/dist.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` DIST branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Dist` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### DIST-specific `+`-flags (from `dist.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bC` | Break apart words at character `C` | — | Missing | Word-segmentation customization. |
| `+g` | Count only one occurrence of each word per turn | `--once-per-turn` | Done | Landed 2026-05-23. Per-turn word dedup via HashSet; `first_turn` / `last_turn` are unaffected (they only ever update on first/most-recent encounter). The `+g` overload (bare = once-per-turn, `+gLABEL` = gem filter) matches CLAN's: rewriter checks for empty rest before falling through to the gem branch. Pinned by `dist_g_bare_routes_to_once_per_turn` and `dist_g_with_label_still_routes_to_gem`. |
| `+o` | Only consider words containing the character `C` given in `+bC` | — | Missing | |
| `+d` | Output sdata in form suitable for statistical analysis | — | Missing | DIST routes `+d` through the shared `maingetflag` path at `OSX-CLAN/src/clan/cutt.cpp:9382` (via `dist.cpp::getflag` `default:` at line 545); DIST appears with empty per-program body at `cutt.cpp:9437`, confirming it consumes `+d` for an `onlydata` output-detail level. chatter has no `--only-data` flag for DIST. Per-DIST rewriter arm in `clan_args.rs` passes the token through so clap reports the literal `+d`/`+dN` argument rather than the misleading `--display-mode` rewrite. |

### General `+`-flags DIST inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`. `process_utterance` picks the key derivation: default uses `NormalizedWord::from_word` (lowercased), `+k` uses `NormalizedWord(cleaned_text().to_owned())` (case-preserving) so `Want`/`want`/`WANT` get distinct distribution rows. Pinned by `dist_case_sensitive_splits_case_variants` and `dist_default_collapses_case_variants`. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 8 |
| Partial | 1 |
| Rewriter only | 3 |
| Missing | 6 |

DIST's `+g` is another instance of the **`+g` overload** pattern
documented in MLU/MLT/FREQPOS. Researchers pasting
`dist +g file.cha` (count once per turn) into chatter get
chatter's gem-segment filter requirement (and an "empty"-style
output for files with no matching gem).

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** Rewriter
> at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`; no `clap` field consumes it today.
> Drafted from CLAN manual §7.9.1 (`Unique Options`, DIST) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Output data in a form suitable for statistical analysis." |

DIST's `+d7` is mentioned in passing in the manual as part of the
FREQ-family `+d7` cross-tier comparison.

### Open questions for PI review

1. "Form suitable for statistical analysis" maps cleanly to
   `--format csv` in chatter. Should `+d` translate directly to
   `--format csv` at rewrite time (drop the `--display-mode`
   translation entirely for DIST), or honour both?
2. The DIST `+d7` mention is a stub — is DIST genuinely a `+d7`
   user, or is the manual cross-referencing FREQ's `+d7`?

## Output

Global word list (sorted alphabetically by display form) with:

- Occurrence count across all turns
- First turn number (1-based) in which the word occurs
- Last turn number (omitted if same as first)
- Total number of turns in the transcript

## Turn Definition

**Every utterance is its own turn**, regardless of whether the speaker changed. This matches CLAN's behavior, which was verified during parity testing. There is no speaker-continuity grouping -- each utterance increments the turn counter.

This is different from how turns are defined in MLT (where consecutive utterances by the same speaker form a single turn).

## Differences from CLAN

### Turn counting

Every utterance = one turn (no speaker-continuity grouping), matching CLAN exactly.

### Word identification

Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
