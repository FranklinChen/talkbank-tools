# MLU -- Mean Length of Utterance

**Status:** Current
**Last updated:** 2026-05-26 08:21 EDT

## Purpose

Calculates mean length of utterance in morphemes from the `%mor` tier. When no `%mor` tier is available and `--words` was not passed, reports "utterances = 0, morphemes = 0" (matching CLAN behavior -- no fallback to word counting).

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409094) for the original MLU command specification.

## Usage

```bash
chatter clan mlu file.cha
chatter clan mlu --speaker CHI file.cha
chatter clan mlu --words file.cha
chatter clan mlu --format json corpus/
```

## Options (chatter-native)

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--words` | `-bw` | Count words from main tier instead of morphemes from `%mor` |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `mlu file.cha` | `chatter clan mlu file.cha` |
| `mlu +t*CHI file.cha` | `chatter clan mlu file.cha --speaker CHI` |
| `mlu -bw file.cha` | `chatter clan mlu file.cha --words` |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `mlu` flag, mapped against
chatter's coverage. Sources:

* `OSX-CLAN/src/clan/mlu.cpp` — `usage()` at line 51 and the
  command-specific `getflag()` intercept at line 669.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` MLU branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's `+flag` to
  `--flag` rewriter.
* `crates/talkbank-cli/src/cli/args/clan_common.rs` and
  `clan_commands.rs::Mlu` — chatter's clap field surface.

(Status legend: same as
[FREQ](./freq.md#status-legend) — Done / Partial / Rewriter only /
Missing.)

### MLU-specific `+`-flags (from `mlu.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `-bw` | Count words, not morphemes | `--words` | Done | Direct mapping; doc note above. |
| `-bc` | Count characters, not morphemes | — | Missing | non-UNX only in CLAN; chatter has no analog. |
| `+cS` | Clause-marker delimiter `S` | — | Missing | Used to break utterances into clauses for MLU calculation. |
| `+c@F` | Clause-markers listed in file `F` | — | Missing | File-list workflow. |
| `+gS` | Exclude utterances consisting solely of word `S` | `--exclude-solo-word S` | Done | Fixed 2026-05-22. CLAN's MLU `+gS` overload (vs the inherited gem-segment filter) is now routed via per-subcommand rewriter branch to a new clap field `--exclude-solo-word`. Drops utterances whose every countable word is in the list. Case-insensitive. |
| `+g@F` | `+g` from file | `--exclude-solo-word-file` | Done | Landed 2026-05-23. Same idiom as COMBO/KWAL `+s@F` — rewriter intercepts `+g@F` before the per-word `+gS` arm, dispatch loads via `load_search_expr_file` and extends `--exclude-solo-word`. File format matches CLAN's `cutt.cpp::rdexclf`: one pattern per line, skip blank lines, `#`-comments, and `;%*`-annotation lines. Repeatable. Pinned by `mlu_solo_word_from_file`. |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | chatter's aggregate-vs-per-file model is the inverse choice. |
| `+t%mor` (implicit) | Switch to `%mor` tier (special handling) | (default) | Done | chatter reads `%mor` by default; `+t%mor` is a CLAN re-confirmation. |
| `-t%mor` | Exclude `%mor` tier — implies `--words` semantics | `--words` | Done | Landed 2026-05-23. Rewriter special-cases `-t%mor` under MLU/MLT to emit `--words` instead of the generic `--exclude-tier mor` (which MLU/MLT's clap doesn't accept). Pinned by `mlu_exclude_mor_tier_maps_to_words`, `mlt_exclude_mor_tier_maps_to_words`, and the fall-through `mlu_exclude_non_mor_tier_falls_through` (which confirms `-t%pho` and other non-`%mor` values still route to the generic `--exclude-tier`). |

### General `+`-flags MLU inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` (no `*`) also accepted post-2026-05-21 fix. |
| `+t%X` / `-t%X` | Include/exclude dependent tier `%X` | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | No clap field on MLU; only `+t%mor` is handled implicitly by the default-mor logic. Other `%X` errors at parse time. |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | Banner mapping deferred (PLAN §1.6). |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22; see [FREQ](./freq.md) for the shared implementation. |
| `+s"word"` / `-s"word"` | Include/exclude word in counting | `--include-word` / `--exclude-word` | Partial | MLU's `+s` has command-specific scope (postcode/bracket forms); chatter's word filter is the simple form only. |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | (in MLU: utterance-elision, see above) | `--gem` (banner default) | Partial | `+g` collides between MLU's elision semantic and the general gem filter. CLAN's MLU `+g` is the *elision* meaning per `getflag`; `--gem` is the general filter (semantics differ). Confusing in CLAN itself. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Various retrace / clitic / prosodic-symbol / replacement controls | `--include-retracings` (handles `+r6` only) | Partial | |
| `+u` | Combine across files | (default) | Done | chatter combines by default. Inverse default vs CLAN. |
| `+re` | Recurse subdirectories | (default for directory args) | Done | |
| `+pS` | Add `S` to word delimiters | — | Missing | |
| `+k` | Case-sensitive matching | `--case-sensitive` (via `CommonAnalysisArgs`) | Done (no-op per CLAN) | MLU does no word-keying; `+k` is silently accepted per CLAN's `cutt.cpp::mainusage` no-op semantic. Covered by `CommonAnalysisArgs.case_sensitive` flatten on `ClanCommands::Mlu`. |
| `+wN` / `-wN` | Context window | `--context-after` / `--context-before` (via `InheritedContextArgs`) | Done (no-op per CLAN) | MLU emits aggregate output (mean per speaker) with no per-match emission to surround. CLAN accepts and silently ignores; chatter does the same via the hidden `InheritedContextArgs` flatten on `ClanCommands::Mlu`. |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1 sidecar work. |

### MLU `+d` display modes

See the "Display Modes (`+dN` / `--display-mode N`) — DRAFT" section
below for the per-N table. All `+d` and `+d1` invocations are
**Rewriter only** today — the rewriter rewrites to
`--display-mode N` but the `Mlu` clap struct has no consuming field.

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 12 |
| Partial (chatter abstraction differs) | 5 |
| Rewriter only (would error at parse time) | 3 |
| Missing (no rewriter, no clap field) | 5 |

The `+g` overload is the most subtle issue: MLU's command-specific
`+g` means "exclude an utterance if it consists solely of the given
word" (a special-case elision filter), but chatter's `--gem`
inherited from `CommonAnalysisArgs` means "restrict to gem segment
labelled S" (a general gem filter). Identical syntax, different
semantics — a CLAN user pasting `mlu +gum file.cha` (skip
`um`-only utterances) gets gem-label filtering in chatter (a
no-op for files with no `@G um` gem). Tracked as a Phase 1.7
follow-up.

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** The
> rewriter at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`, but no `clap` field consumes that token
> today. Drafted from CLAN manual §7.21.2 (`Unique Options`, MLU) for
> PI review. Plan: `<workspace>/docs/superpowers/plans/2026-05-11-clan-rewriter-honor-three-flags.md`
> Phase 3.

MLU's `+d` table is small — two N-values, both Excel-friendly output
formats. Quoted from CLAN manual §7.21.2:

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "You can use this switch, together with the ID specification to output data for Excel." Example: `mlu +d +tCHI sample.cha` produces a one-line @ID-keyed record: ``en\|sample\|CHI\|1;10.4\|female\|\|\|Target_Child\|\| 5  7 1.400 0.490`` (fields: @ID, utterance count, morpheme count, MLU, MLU std dev). Requires `@ID` headers per participant. |
| `+d1` | "This level of the `+d` switch outputs data in another systematic format, with data for each speaker on a single line. However, this form is less adapted to input to a statistical program than the output for the basic `+d` switch. Also, this switch works with the `+u` switch, whereas the basic `+d` switch does not." Example: ``*CHI:  5  7 1.400 0.490``. |

### Open questions for PI review

1. `+d` (no number) maps cleanly to `--format csv` in chatter. Should
   `--display-mode 0` (or absent N) imply `--format csv`, or remain a
   separate axis?
2. `+d1` is "less adapted to statistical input" yet combinable with
   `+u`. That combinability is the differentiating feature; should
   chatter expose it as a `--display-mode merged-by-speaker` enum
   variant?
3. The `+d` output requires `@ID` headers per participant. Should
   `--display-mode` error early if `@ID` rows are missing for any
   matched speaker, or fall back to the speaker-code-only form
   silently?

## Algorithm

For each utterance with a `%mor` tier:

1. Count **1 per stem** (the base morpheme word)
2. Count **+1 per bound morpheme suffix** -- but ONLY these 7 suffix strings: `PL`, `PAST`, `Past`, `POSS`, `PASTP`, `Pastp`, `PRESP`
3. Count **+1 per clitic stem** (`~` separated)
4. Count clitic suffixes using the same 7-string rule
5. **Fusional features** (`&PRES`, `&INF`, etc.) do NOT count

Per speaker, compute:
- Number of utterances
- Total morphemes
- **MLU** (mean = total morphemes / utterances)
- **Standard deviation** (population SD, dividing by n)
- **Range** (min, max morphemes per utterance)

### Brown's Morpheme Rules

This was a key discovery during parity verification. CLAN only counts 7 specific suffix strings as bound morphemes:

| Suffix | Meaning |
|--------|---------|
| `PL` | Plural |
| `PAST` | Past tense |
| `Past` | Past tense (alternate) |
| `POSS` | Possessive |
| `PASTP` | Past participle |
| `Pastp` | Past participle (alternate) |
| `PRESP` | Present participle |

All other suffixes (including fusional features like `&PRES`, `&INF`, `&3S`) are ignored for MLU counting. This matches Brown's (1973) original operationalization of "morpheme" for child language analysis.

### Example

Given `%mor: pro|I v|want-PAST det|a n|cookie-PL`:

- `pro|I` = 1 stem = **1**
- `v|want-PAST` = 1 stem + 1 suffix (PAST) = **2**
- `det|a` = 1 stem = **1**
- `n|cookie-PL` = 1 stem + 1 suffix (PL) = **2**
- Total: **6 morphemes**

## Output

```text
Speaker: CHI
  Utterances: 42
  Morphemes: 168
  MLU: 4.000
  SD: 1.732
  Range: 1-9
```

## Differences from CLAN

### Standard deviation

Uses **population SD** (dividing by n), not sample SD (dividing by n-1). Verified against CLAN output -- CLAN uses population SD too.

### Morpheme counting

Uses parsed `%mor` tier structure (`MorWord` features and post-clitics) rather than text splitting on spaces and delimiters. The semantic result is identical thanks to applying Brown's 7-suffix rule, but the mechanism is type-safe.

### No %mor tier behavior

When no `%mor` tier exists and `--words` was not passed, reports 0 utterances for the speaker (matching CLAN). Does not silently fall back to word counting.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
