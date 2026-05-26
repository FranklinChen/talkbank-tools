# FREQ -- Word Frequency

**Status:** Current
**Last updated:** 2026-05-26 08:21 EDT

## Purpose

Counts word tokens and types and computes type-token ratio (TTR). The legacy manual describes `FREQ` as one of CLAN's most powerful and easiest-to-use programs, producing word-frequency counts and lexical-diversity measures over selected files and speakers.

In `talkbank-clan`, `FREQ` counts words on the main tier by default, or morphemes from the `%mor` tier when `--mor` is set.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409093) for the original FREQ command specification.

## Usage

```bash
chatter clan freq file.cha
chatter clan freq --speaker CHI file.cha
chatter clan freq --format json corpus/
chatter clan freq --mor file.cha
chatter clan freq --include-word "the" file.cha
```

> **`+k` / `--case-sensitive` is wired as of 2026-05-22 (pattern
> matching) + 2026-05-23 (frequency-table keying).** Without the
> flag, word matching is case-insensitive (CLAN's default and
> chatter's default). With `+k` (or `--case-sensitive`):
> - `+s`/`--include-word` patterns and the searched words skip
>   lower-casing, so an exact-case match is required;
> - the frequency-table key preserves original case, so `Want`,
>   `want`, and `WANT` produce three separate entries (each with
>   count 1) instead of collapsing to one.

## Options (chatter-native)

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` | Exclude speaker |
| `--include-word <WORD>` | `+s"word"` | Only count matching word |
| `--exclude-word <WORD>` | `-s"word"` | Skip matching word |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--case-sensitive` | `+k` | Match `+s` / `--include-word` patterns case-sensitively (default: case-insensitive) |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |
| `--mor` | -- | Count morphemes from `%mor` tier instead of words from main tier |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `freq` flag, mapped against
chatter's coverage. Sources:

* `OSX-CLAN/src/clan/freq.cpp` — `usage()` at line 152 and the
  command-specific `getflag()` intercept at line 621.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` at line 9090
  (program-keyed `FREQ` branches throughout).
* `crates/talkbank-clan/src/clan_args.rs` — chatter's `+flag` to
  `--flag` rewriter.
* `crates/talkbank-cli/src/cli/args/clan_common.rs` and
  `crates/talkbank-cli/src/cli/args/clan_commands.rs::Freq` —
  chatter's clap field surface for FREQ.

### Status legend

* **Done** — chatter accepts the flag and the semantic is implemented.
* **Partial** — chatter accepts a related abstraction with non-identical
  semantics; gap noted in the *Notes* column.
* **Rewriter only** — `clan_args.rs` rewrites the `+flag` to a chatter
  flag, but no clap field on `Freq` consumes that token; passing the
  flag errors out at parse time.
* **Missing** — neither rewriter nor clap field handles it.

### Freq-specific `+`-flags (from `freq.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Compute standard / dialectal / actual forms | — | Missing | No chatter analog. |
| `+bN` | Frame size for MATTR (Moving-Average TTR) | — | Missing | Sliding-window TTR is unimplemented. |
| `+c` / `+c0` | Find capitalised words only | `--capitalization initial` | Done | Landed 2026-05-22. Skips any countable word whose first character is not uppercase before frequency accumulation. Both `+c` and `+c0` are accepted as aliases (CLAN treats them identically); subcommand-guarded so MAXWD/CHECK/IPSYN/DSS keep their existing `+cN` meaning. Shares the `CapitalizationFilter` enum with VOCD. |
| `+c1` | Find words with upper-case letters in the middle | `--capitalization mid` | Done | Landed 2026-05-22. Keeps only words whose surface form contains an uppercase letter AFTER position 0 (e.g. `McDonald`, `iPhone`). Initial-capital words like `Cookie` are dropped. Shares the `CapitalizationFilter::MidUpper` predicate with VOCD. |
| `+c2` | Match every string `+s` specifies (not just first) | — | Missing | Multi-word search variant. |
| `+c3` | Match multi-word groups in any order on a tier | — | Missing | |
| `+c4` | Match only if tier is *solely* the multi-word group | — | Missing | |
| `+c5` | With `+d7`, reverse source/target tier priority | — | Missing | Linked-tier rendering. |
| `+c6` | Count only repeat segments | — | Missing | Repeat-marker `↫` workflow. |
| `+c7` | For multi-word searches output actual words matched | — | Missing | |
| `+o` / `+o0` | Sort by descending frequency | (default) | Done | chatter's default sort already matches. |
| `+o1` | Sort by reverse concordance | `--reverse-concordance` | Done | Landed 2026-05-23. Replaces the default frequency-descending sort with a sort by the reversed character sequence of each word — groups words sharing a suffix. Pinned by `freq_reverse_concordance_groups_by_suffix` (with `cat`/`bat`/`dog`/`log` input, the sorted result reflects reversed-string comparison) and `freq_default_sort_is_alphabetical_when_freqs_equal` (regression companion). End-to-end smoke: `cat bat dog log apple maple` with `+o1` clusters maple/apple, dog/log, bat/cat. `+o2` (reverse concordance + non-CHAT output) is a separate Missing item. |
| `+o2` | Sort by reverse concordance of first word, preserve full line | — | Missing | Non-CHAT output. |
| `+o3` | Combine selected speakers per file into one list | partial via `--per-file` inverse | Partial | chatter's aggregate-vs-per-file model is the inverse choice; not byte-identical. |
| `+d` | All selected words + freq + line numbers | — | Rewriter only | `+dN` rewrites to `--display-mode N`, no consuming field. See "Display Modes" §. |
| `+d0` | Concordance with frequencies and line text | — | Rewriter only | |
| `+d1` | One word per line, no frequencies | `--word-list-only` | Done | Rewriter maps bare `+d1`. Emits an alphabetized deduped word list merged across all speakers, suitable as `kwal +s@FILE` input. |
| `+d2` | Spreadsheet output (Excel-ready) | `--format csv` | Done | Rewriter maps bare `+d2`; existing `render_csv` path already produces the per-speaker per-word CSV. `+d20` (per-speaker+word row variant) is a separate Rewriter-only item. |
| `+d20` | Spreadsheet with one row per speaker+word | — | Rewriter only | |
| `+d3` | Spreadsheet, types/tokens/TTR only | `--types-tokens-only --format csv` | Done | Rewriter emits both flags together: shares the `types_tokens_only` mode with `+d4`, then routes through `render_csv` instead of `render_clan`. |
| `+d4` | Type/token info only | `--types-tokens-only` | Done | Rewriter maps bare `+d4`. Per-speaker banner + separator + totals + TTR note all kept; per-word frequency lines dropped. `+d3` (same content, CSV/spreadsheet form) is a separate item. |
| `+d5` | Output `+s` words including those with 0 freq | — | Rewriter only | |
| `+d6` | Limited search-word surrounding context | — | Rewriter only | |
| `+d7` | Frequencies linked between dependent tier and speaker | — | Rewriter only | |
| `+d8` | Cross-tabulation of one dependent tier with another | — | Rewriter only | |
| `+dCN`, `+d<N`, `+d>=N`, `+d=N`, `+d>N` | Output words used by `<`, `<=`, `=`, `=>`, `>` than N percent of speakers | — | Missing | Separate from plain `+dN`; rewriter does not handle `+dC...`. |

### General `+`-flags FREQ inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` | Include speaker `*X` | `--speaker X` | Done | |
| `-t*X` | Exclude speaker `*X` | `--exclude-speaker X` | Done | |
| `+t%X` | Include dependent tier `%X` | `--tier X` (rewriter target) | Rewriter only | No `--tier` field on `CommonAnalysisArgs`; tier scope is per-command via `clan_scope_mode()`. |
| `-t%X` | Exclude dependent tier `%X` | `--exclude-tier X` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | Banner mapping deferred (see PLAN §1.6). |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22. Rewriter routes `+t#ROLE` → `--role ROLE`; the per-utterance filter checks the speaker's `@ID:` role field case-insensitively; banner-scope renders `ONLY speaker main tiers with role(s): ROLE;`. |
| `+s"word"` / `+sword` | Search for word | `--include-word` | Done | |
| `-s"word"` / `-sword` | Exclude word | `--exclude-word` | Done | |
| `+s@F` / `-s@F` | Search / exclude words listed in file F | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. Universal across non-SCRIPT, non-COMBO commands. File format matches CLAN's `cutt.cpp::rdexclf`: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped; UTF-8 BOM stripped. Repeatable. |
| `+gX` | Include gem labelled X | `--gem` | Done | |
| `-gX` | Exclude gem labelled X | `--exclude-gem` | Done | |
| `+rN` (N=1..8) | Various retrace / clitic / prosodic-symbol / replacement controls | `--include-retracings` (handles `+r6` only) | Partial | Only `+r6` ↔ `--include-retracings` is wired; `+r1`..`+r5`, `+r50`, `+r7`, `+r8` are missing. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+pS` | Add `S` to word delimiters | — | Missing | |
| `+f` / `+fEXT` | Output to file with extension | `--output-ext` (rewriter target) | Rewriter only | chatter writes to stdout by default; sidecar-file pattern is Phase 1.1. |
| `+u` | Combine across files (no per-file split) | (default) | Done | chatter combines by default; `--per-file` opts in to per-file output. Inverse default vs CLAN. |
| `+re` | Recurse subdirectories | (default for directory args) | Done | chatter's path argument accepts a directory and recurses. |
| `+oS` / `-oS` | Include / exclude extra output tier `S` | — | Missing | |
| `+x` | Exclude utterances by content | — | Missing | `+x=0w`, `+x>0w`, `+xword` shapes. |
| `+k` | Case-sensitive matching | `--case-sensitive` | Done | Two-layer fix. Pattern matching layer (`WordFilter::case_sensitive`) landed 2026-05-22 — `--include-word`/`--exclude-word` patterns skip the default `.to_lowercase()` on both sides. Frequency-table KEYING layer landed 2026-05-23 — `process_utterance` skips `NormalizedWord::from_word`'s lowercasing on the map key (main-tier and `%mor` branches), so `Want`/`want`/`WANT` become three distinct entries instead of collapsing to one. Pinned by `freq_case_sensitive_preserves_case_in_keys` and `freq_default_collapses_case_variants`. |
| `+wN` / `-wN` | Context window around matched word | `--context-after` / `--context-before` (via `InheritedContextArgs`) | Done (no-op per CLAN) | FREQ emits per-word frequency totals; no per-match emission to surround. CLAN accepts and silently ignores; chatter does the same via the hidden `InheritedContextArgs` flatten on `ClanCommands::Freq`. |
| `+y` | (CLAN: include all utterances including non-tier) | — | Missing | |

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 18 |
| Partial (chatter abstraction differs) | 2 |
| Rewriter only (would error at parse time) | 12 |
| Missing (no rewriter, no clap field) | 14 |

The 17 "Rewriter only" entries are the single biggest correctness
hazard today: a user pasting CLAN-style `freq +d2 file.cha` gets
`error: unexpected argument '--display-mode' found`. Either the
rewriter must stop emitting those flags, or chatter must accept and
implement them. Tracked under Phase 1.7 (this audit) and Phase 2
(per-command body parity).

### Confirmed-broken invocations (2026-05-21)

These were exercised end-to-end during the audit and produced wrong
output for a CLAN-equivalent invocation:

| Invocation | What chatter does | What CLAN does |
|---|---|---|
| `chatter clan freq +d2 file.cha` | parse error: `unexpected argument '--display-mode'` | spreadsheet output |
| `chatter clan freq +k file.cha` | parse error: `unexpected argument '--case-sensitive'` | case-sensitive search |
| `chatter clan freq +t%mor file.cha` | parse error: `unexpected argument '--tier'` | analyses `%mor` dependent tier |
| ~~`chatter clan freq +tCHI file.cha` (no `*`)~~ | **fixed 2026-05-21** — `+tCHI` and `-tMOT` now rewrite identically to `+t*CHI` / `-t*MOT` | identical to `+t*CHI` (silently prepends the `*`) |

The `+tCHI` case was a `clan_args.rs::rewrite_tier_speaker` gap: the
function required the first byte of `rest` to be `*`, `%`, or `@`,
and fell through to `None` otherwise. The default branch now treats
`+t<word>` as an implicit speaker code, matching CLAN's behaviour
exactly. Closed in commit landed alongside this audit, with two new
unit tests (`speaker_include_no_asterisk`,
`speaker_exclude_no_asterisk`) in `clan_args::tests`.

## CLAN Equivalence

| CLAN command | Rust equivalent |
|---|---|
| `freq file.cha` | `chatter clan freq file.cha` |
| `freq +t*CHI file.cha` | `chatter clan freq file.cha --speaker CHI` |
| `freq +s"the" file.cha` | `chatter clan freq file.cha --include-word "the"` (case-sensitive matching not currently supported — see callout above) |
| `freq *.cha` | `chatter clan freq corpus/` |

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** The legacy
> rewriter at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`, but no `clap` field consumes that token
> today. This section drafts the per-N table from CLAN manual
> §7.10.15 (`Unique Options`, FREQ) verbatim, for PI review and
> subsequent TDD implementation. Tracked in
> `<workspace>/docs/superpowers/plans/2026-05-11-clan-rewriter-honor-three-flags.md`
> Phase 3.

`FREQ` uses `+d` to switch output format, *not* to vary verbosity. Each
value of N selects a different report shape. Quoted from CLAN manual
§7.10.15:

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Perform a particular level of data analysis. By default, the output consists of all selected words found in the input data file(s) and their corresponding frequencies." (Equivalent to no-flag default.) |
| `+d0` | "Output provides a concordance with the frequencies of each word, the files and line numbers where each word, and the text in the line that matches." |
| `+d1` | "Outputs each of the words found in the input data file(s) one word per line with no further information about frequency. Later this output could be used as a word list file for `kwal` or `combo` programs." |
| `+d2` | "Output is sent to a file in a form that can be opened directly in Excel. To do this, you must include information about the speaker roles you wish to include in the output spreadsheet." (Manual example: `freq +d2 +t@ID="*|Target_Child|*" *.cha`.) |
| `+d3` | "Essentially the same as that for `+d2`, but with only the statistics on types, tokens, and the type–token ratio. Word frequencies are not placed into the output." (Note: `+d2` and `+d3` assume `+f`; no need to pass it explicitly.) |
| `+d4` | "Allows you to output just the type–token information." |
| `+d5` | "Output all words you are searching for, including those that occur with zero frequency. ... Can be combined with other `+d` switches." |
| `+d6` | "When used for searches on the main line, outputs matched forms with a separate tabulation of replaced forms, errors, partial omissions, and full forms." Also `+d6 +sm\|n*,o%` on `%mor` line: produces separate counts per part-of-speech instantiation. |
| `+d7` | "Links forms on a 'source' tier with their corresponding words on a 'target' tier." Default source is `%mor`; pass a tier name to change source. Items on the two tiers must be in one-to-one correspondence. `+c5` swaps source ↔ target. |
| `+d8` | "Outputs words and frequencies of cross tabulation of one dependent tier with another." |

### Open questions for PI review

1. `+d0`: emits a concordance — overlaps with `KWAL` semantically. Should
   chatter's `freq --display-mode 0` reuse the `kwal` output path
   internally, or produce its own concordance shape?
2. `+d1`: word-list output suitable as input to `kwal +s@file`. Should
   the file be auto-named (`<basename>.fre`?) or printed to stdout by
   default?
3. `+d2`/`+d3`: "form that can be opened directly in Excel" maps to
   `--format csv` in chatter. Is this duplication acceptable, or should
   `--display-mode 2` *imply* `--format csv` (and conflict-error
   otherwise)?
4. `+d4`: "type-token information only" — same content as the existing
   text/json default, minus the word frequencies. Add a new
   `Truncated` variant to the output struct, or emit a CSV row with
   just types/tokens/TTR?
5. `+d5`: combinable with other `+d` values. How should this combine in
   clap — a `Vec<DisplayMode>` rather than scalar `Option<u8>`?
6. `+d6`/`+d7`/`+d8`: deeply specific to `%mor` and cross-tier
   tabulation. Are these in scope for chatter's freq, or are they
   future work (probably alongside or instead of `mortable` /
   `freqpos`)?

The `+dCN` form (capital `C` plus a number — "output only words used by
<, <=, =, => or > than N percent of speakers") is a separate flag from
plain `+dN`; the rewriter does not currently handle `+dC...`. It would
get its own clap field (`--speaker-percentage`-style) rather than
overload `--display-mode`.

## Output

Per-speaker frequency tables with:

- Word frequency counts (sorted by count descending, then alphabetically)
- Total types (unique words) and tokens (total words)
- TTR (type-token ratio = types / tokens)

### Example output (text)

```text
Speaker: CHI
  the       12
  I         8
  want      6
  a         5
  go        4
  ...
Types: 45
Tokens: 127
TTR: 0.354
```

### Example output (JSON)

```json
{
  "speakers": {
    "CHI": {
      "words": { "the": 12, "I": 8, "want": 6, ... },
      "types": 45,
      "tokens": 127,
      "ttr": 0.354
    }
  }
}
```

## Word Normalization

Words are grouped using `NormalizedWord`, which lowercases and strips compound markers (`+`) for counting purposes, while preserving the original CLAN display form (with `+`) for output. This means `wanna+go` and `Wanna+Go` are counted as the same word.

## Differences from CLAN

### Word identification

The legacy manual says `FREQ` ignores `xxx`, `www`, and words beginning with `0`, `&`, `+`, `-`, or `#` by default, and also ignores header and code tiers unless selected. CLAN implements much of this with character-level string-prefix matching:

```c
if (word[0] == '0') continue;     // omitted words
if (word[0] == '&') continue;     // fillers/nonwords
if (word[0] == '+') continue;     // terminators
```

Our implementation uses AST-based `is_countable_word()`, which checks semantic type rather than string prefixes. This is more precise -- a filler (`&-um`) and a phonological fragment (`&+fr`) have distinct semantic types in our model, even though CLAN lumps them together under the `&` prefix.

### Manual features not yet mirrored directly

The legacy manual documents several advanced `FREQ` workflows, including `+s@file` lexical-group lists, `%mor`/`%gra` combined search with `+d7`, and multilingual searches. Some of those behaviors are covered in `talkbank-clan` through broader filtering infrastructure, but the command chapter should not imply one-for-one flag parity unless explicitly implemented.

### Output ordering

Output is deterministic via sorted collections (count descending, then alphabetically). CLAN's ordering can vary across runs.

### Output formats

Supports text, JSON, and CSV formats. CLAN produces text only. Use `--format clan` for character-level CLAN-compatible output.

### Multi-file behavior

Results are merged across files by default (`+u` behavior). CLAN requires explicit `+u` flag. Use `chatter clan freq dir/` for recursive directory traversal (CLAN uses shell globs).

### Golden test parity

Verified against CLAN C binary output. 100% parity.
