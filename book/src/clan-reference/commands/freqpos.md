# FREQPOS — Word Frequency by Position

**Status:** Current
**Last updated:** 2026-05-23 15:55 EDT

## Purpose

Counts how often each word appears in initial, final, other (middle), or one-word positions within utterances. FREQPOS is part of the FREQ family of commands and is useful for studying positional word preferences -- for example, whether a child tends to place certain words at the beginning or end of utterances.

### Position Classification

- **Initial**: first word of a multi-word utterance
- **Final**: last word of a multi-word utterance
- **Other**: any middle word of a multi-word utterance (3+ words)
- **One-word**: the sole word in a single-word utterance

## Usage

```bash
chatter clan freqpos file.cha
chatter clan freqpos file.cha --speaker CHI
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

Authoritative enumeration of every CLAN `freqpos` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/freqpos.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` FREQPOS branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Freqpos` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

FREQPOS has the narrowest command-specific flag set of any CLAN
analysis tool: just two flags beyond the general inherited set.

### FREQPOS-specific `+`-flags (from `freqpos.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+d` | Use first / second / other classification instead of first / last / other | `--position-classification <last\|second>` | Done | Landed 2026-05-23. `FreqposConfig::position_classification` (enum `FirstLastOther` / `FirstSecondOther`) gates the per-utterance classification: with `+d`, position 1 becomes the "second" slot and positions ≥ 2 go to "other"; without `+d`, position `len-1` is "final" and middle positions are "other". `render_clan` swaps the column header (`final =` ↔ `second =`) and footer label to match. Pinned by `freqpos_second_mode_reclassifies_position_one`, `freqpos_default_mode_keeps_final_label`, and rewriter tests `freqpos_second_mode_classification` + `freq_d_bare_does_not_match_position_classification` (scope-narrowing). The generic `+dN` rewrites for FREQPOS are tracked separately under "Display Modes". |
| `+gS` / `+g@S` | Display only word(s) `S` (or words in file `@S`) | `--gem` | Partial | The FREQPOS `+g` flag means *display only certain words* — a vocabulary filter — whereas the inherited `--gem` is the gem-segment filter. Identical syntax, different semantics, like the `+g` overload in MLU / MLT. |

### General `+`-flags FREQPOS inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22; see [FREQ](./freq.md) for the shared implementation. |
| `+s"word"` / `-s"word"` | Include/exclude word | `--include-word` / `--exclude-word` | Done | xxx/yyy/www are excluded by default (init-time `addword`). |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | (in FREQPOS: vocabulary filter, see above) | `--gem` | Partial | Overloaded. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | Inverse default vs CLAN. |
| `+re` | Recurse | (default) | Done | |
| `+pS` | Word delimiter | — | Missing | |
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`. `process_utterance` picks the key derivation: default uses `NormalizedWord::from_word` (lowercased), `+k` uses `NormalizedWord(cleaned_text().to_owned())` (case-preserving) so `Want`/`want`/`WANT` land in separate by-word entries. Pinned by `freqpos_case_sensitive_splits_case_variants` and `freqpos_default_collapses_case_variants`. |
| `+wN` / `-wN` | Context window | `--context-window` (rewriter target) | Rewriter only | |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1. |

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 9 |
| Partial | 2 |
| Rewriter only | 3 |
| Missing | 3 |

The `+gS` overload (FREQPOS: vocabulary filter; inherited: gem
filter) deserves the most attention before this command's body
parity can be claimed complete. Same overload as MLU and MLT,
tracked as a Phase 1.7 follow-up.

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** Rewriter
> at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`; no `clap` field consumes it today.
> Drafted from CLAN manual §7.12.1 (`Unique Options`, FREQPOS) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Count words in either first, second, or other positions. The default is to count by first, last, and other positions." |

### Open questions for PI review

1. FREQPOS's `+d` switches the position-classification scheme from
   "first / last / other / one-word" (default) to
   "first / second / other". That's not a display change — it's a
   *bucketing* change. Map to `--positions <first-last|first-second>`
   enum rather than `--display-mode`.
2. If we keep the `--display-mode` translation, `+d` (no number)
   corresponds to a single behavior (the alternative bucketing), so
   `--display-mode 0` is the only valid value. The clap field should
   probably be an enum with two variants (`Default`,
   `FirstSecondOther`), not a numeric `Option<u8>`.

## Output

Global word list (sorted alphabetically by display form) with positional breakdown (initial/final/other/one-word counts per word), followed by aggregate position totals.

## Differences from CLAN

- Word identification uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching
- Position classification operates on parsed AST word lists rather than raw text token splitting
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
