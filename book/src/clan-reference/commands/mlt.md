# MLT -- Mean Length of Turn

**Status:** Current
**Last updated:** 2026-05-22 09:04 EDT

## Purpose

Calculates mean length of turn in utterances and words. A "turn" is a maximal consecutive sequence of utterances by the same speaker; the turn boundary is detected when a different speaker produces the next utterance.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409101) for the original MLT command specification.

## Usage

```bash
chatter clan mlt file.cha
chatter clan mlt --speaker CHI file.cha
chatter clan mlt --format json corpus/
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

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `mlt file.cha` | `chatter clan mlt file.cha` |
| `mlt +t*CHI file.cha` | `chatter clan mlt file.cha --speaker CHI` |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `mlt` flag, mapped against
chatter's coverage. Sources:

* `OSX-CLAN/src/clan/mlt.cpp` — `usage()` at line 43 and the
  command-specific `getflag()` intercept at line 582.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` MLT branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's `+flag` to
  `--flag` rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Mlt` plus
  `clan_common.rs::CommonAnalysisArgs` — chatter's clap field
  surface for MLT.

(Status legend: same as
[FREQ](./freq.md#status-legend) — Done / Partial / Rewriter only /
Missing.)

### MLT-specific `+`-flags (from `mlt.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Do not count empty utterances (`0` utterances) | — | Missing | CLAN's default is to count them; `+a` flips that. chatter does not yet expose an empty-utterance switch. |
| `+at` | Count empty utterances when they carry the `[+ trn]` postcode | — | Missing | Subtle conversational-analysis convention. |
| `+cS` | Clause-marker delimiter `S` | — | Missing | Used to split utterances into clauses for turn counting. |
| `+c@F` | Clause markers from file `F` | — | Missing | File-list workflow. |
| `+gS` | Exclude utterances consisting solely of word `S` | `--exclude-solo-word S` | Done | Fixed 2026-05-22. Per-subcommand rewriter routing + new clap field; same semantic as MLU. See the [MLU page](./mlu.md) for the broader `+g` overload story. |
| `+g@F` | `+g` from file | `--exclude-solo-word-file` | Done | Landed 2026-05-23. Same shape as MLU `+g@F`; see the [MLU page](./mlu.md) for the rewriter/loader details. Pinned by `mlt_solo_word_from_file`. |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | chatter's aggregate-vs-per-file model is the inverse choice. |
| `+t%X` (implicit) | Switch to dependent-tier mode for the turn split | (default for non-mlt commands) | Partial | MLT's `nomain = TRUE` for `+t%X` has no chatter analog today; MLT chatter operates on main tier exclusively. |

### General `+`-flags MLT inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` (no `*`) also accepted post-2026-05-21 fix. |
| `+t%X` / `-t%X` | Include/exclude dependent tier `%X` | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | No `--tier` field on the `Mlt` struct. |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22; see [FREQ](./freq.md) for the shared implementation. |
| `+s"word"` / `-s"word"` | Include/exclude word | `--include-word` / `--exclude-word` | Partial | MLT's `+s` special-cases `xxx`/`yyy`/`www` exclusion; chatter's filter does not. |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | (in MLT: utterance-elision filter, see above) | `--gem` | Partial | Same `+g` overload as MLU. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic-symbol / replacement controls | `--include-retracings` (handles `+r6` only) | Partial | |
| `+u` | Combine across files | (default) | Done | chatter combines by default; inverse default vs CLAN. |
| `+re` | Recurse subdirectories | (default for directory args) | Done | |
| `+pS` | Add `S` to word delimiters | — | Missing | |
| `+k` | Case-sensitive matching | `--case-sensitive` (rewriter target) | Rewriter only | |
| `+wN` / `-wN` | Context window | `--context-window` (rewriter target) | Rewriter only | |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1. |

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 7 |
| Partial (chatter abstraction differs) | 6 |
| Rewriter only (would error at parse time) | 5 |
| Missing (no rewriter, no clap field) | 5 |

The `+a` / `+at` empty-utterance switch and the clause-delimiter
flags (`+cS`, `+c@F`) are MLT's most distinctive omissions: they
affect what counts as "an utterance" and "a turn" and so directly
drive the MLT ratio. Filed as Phase 1.7 follow-ups.

## Algorithm

1. Walk utterances in file order
2. Group consecutive utterances by the same speaker into turns
3. For each speaker, compute:
   - Number of turns
   - Total utterances and total words
   - **MLT-u**: mean turn length in utterances
   - **MLT-w**: mean turn length in words
   - **SD**: standard deviation of words per utterance (population SD, dividing by n)

### Turn boundaries

A turn boundary occurs when a different speaker produces the next utterance. For example:

```text
*CHI: I want a cookie .          <- turn 1 (CHI)
*CHI: please .                   <- still turn 1 (CHI)
*MOT: here you go .              <- turn 2 (MOT)
*CHI: thank you .                <- turn 3 (CHI)
```

CHI has 2 turns (3 utterances), MOT has 1 turn (1 utterance).

## Output

```text
Speaker: CHI
  Turns: 15
  Utterances: 42
  Words: 127
  MLT (utterances): 2.800
  MLT (words): 8.467
  SD: 3.217
```

## Differences from CLAN

### Standard deviation

Uses **population SD** (dividing by n), matching CLAN. This was verified during parity testing.

### SD basis

The SD is computed over **per-utterance word counts**, not per-turn totals. This matches CLAN's behavior, which was confirmed through golden test comparison.

### Turn detection

Operates on parsed speaker codes from the AST rather than raw text line prefixes. Functionally identical but type-safe.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
