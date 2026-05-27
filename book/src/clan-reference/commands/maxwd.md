# MAXWD -- Longest Words

**Status:** Current
**Last updated:** 2026-05-27 10:39 EDT

## Purpose

Finds the longest words used by each speaker, reporting a ranked table of unique words sorted by character length descending. Word length is measured in characters after normalization (lowercasing, stripping `+` compound markers and `'` apostrophes for CLAN compatibility).

## Usage

```bash
chatter clan maxwd file.cha
chatter clan maxwd --speaker CHI file.cha
chatter clan maxwd --limit 50 file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--limit <N>` | `+cN` | Maximum number of words to show (default: 20) |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `maxwd` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/maxwd.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` MAXWD branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Maxwd` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### MAXWD-specific `+`-flags (from `maxwd.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Consider ONLY unique-length utterances/words | `--unique-length-only` | Done | Landed 2026-05-23. Per-speaker filter: drops words whose character length is shared with another word in the same speaker's lexicon. `max_length` is recomputed over the surviving entries, so the output reflects only unique-length words. Pinned by `maxwd_unique_length_only_drops_shared_length_words` and the regression companion `maxwd_default_keeps_shared_length_words`. CLAN's flag covers both "words" and "utterances"; chatter currently only implements the words sense (MAXWD's `+gN` utterance-mode is a separate Missing item). |
| `+bS` | Add chars in `S` to morpheme-delimiter list | — | Missing | Shared with WDLEN. |
| `-bS` | Remove chars from delimiter list (`-b` clears all) | — | Missing | |
| `+cN` | Display the `N` longest items | `--limit N` (or `-n N`) | Done | Rewriter handles this since 2026-05-22: `+c50` → `--limit 50` for the MAXWD subcommand specifically (CHECK still gets `--bullets N`). |
| `+gN` | Find longest utterance instead of longest word; N selects metric (1=morph, 2=word, 3=char) | — | Missing | A different domain (utterances vs words). Per-MAXWD rewriter arm in `clan_args.rs` returns None for digit-only `+gN` so the literal token passes through to clap (which rejects it) rather than silently mis-routing to `--gem N` via the generic `+g` → `rewrite_gem` arm. MAXWD's `+gX` non-digit gem-filter form continues to fall through to `rewrite_gem`. |
| `+xN` | Exclude lengths | `--exclude-length N` (repeatable) | Done | Landed 2026-05-23. Drops words whose character length matches any value in `exclude_lengths`. Repeatable on the CLI (`+x5 +x7` or `--exclude-length 5 --exclude-length 7`). `max_length` is recomputed over surviving entries. Applied per-speaker after `+a`. Pinned by `maxwd_exclude_lengths_drops_listed_lengths` plus rewriter tests `maxwd_exclude_length_single` and `maxwd_exclude_length_multiple`. End-to-end smoke: input `["I", "go", "bye", "hi", "hello", "world", "cookie"]` with `+x6` reports max_length 5 instead of 6 (cookie dropped). |
| `+d` | Display modes (see "Display Modes" below) | — | Missing | MAXWD has no local `case 'd'`; consumption via the shared `maingetflag` path at `OSX-CLAN/src/clan/cutt.cpp:9382` with non-empty per-program body at `cutt.cpp:9475` (`onlydata == 1` → `puredata = 0`). chatter has no `--display-mode` consumer for MAXWD. Per-MAXWD rewriter arm in `clan_args.rs` passes the token through so clap reports the literal `+d`/`+dN` argument rather than the misleading `--display-mode` rewrite. |

### General `+`-flags MAXWD inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+s"word"` / `-s"word"` | Include/exclude word | `--include-word` / `--exclude-word` | Done | |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | Gem filter (without `N`) | `--gem` | Done | Distinct from `+gN`. |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | |
| `+re` | Recurse | (default) | Done | |
| `+k` | Case-sensitive | `--case-sensitive` | Done | Landed 2026-05-23. Reads `CommonAnalysisArgs::case_sensitive`. `process_utterance` picks the key derivation: default uses `NormalizedWord::from_word` (lowercased), `+k` uses `NormalizedWord(cleaned_text().to_owned())` (case-preserving) so case variants count as distinct words for the unique-length and exclude-length filters. Pinned by `maxwd_case_sensitive_splits_case_variants` and `maxwd_default_collapses_case_variants`. |
| `+wN` / `-wN` | Context window | `--context-after` / `--context-before` (via `InheritedContextArgs`) | Done (no-op per CLAN) | MAXWD emits a longest-words list per speaker; no per-match emission to surround. CLAN accepts and silently ignores; chatter does the same via the hidden `InheritedContextArgs` flatten on `ClanCommands::Maxwd`. |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 12 |
| Partial | 1 |
| Rewriter only | 2 |
| Missing | 6 |

`+cN` ↔ `--limit N` was a one-line rewriter follow-up — closed
2026-05-22. The rewriter now distinguishes the MAXWD subcommand
(`+cN` → `--limit N`) from the CHECK subcommand (`+cN` →
`--bullets N`); two RED→GREEN tests pin the behaviour.

Other one-line rewriter follow-ups identified during this audit
(e.g. FREQ's `+c0..7`, VOCD's `+c`) remain open and are tracked
under Phase 1.7.

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** Rewriter
> at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`; no `clap` field consumes it today.
> Drafted from CLAN manual §7.19.1 (`Unique Options`, MAXWD) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "The `+d` level of this switch produces output with one line for the length level and the next line for the word." |
| `+d1` | "Produces output with only the longest words, one per line, in order, and in legal chat format." |

### Open questions for PI review

1. `+d` is a two-line-per-result format (length on one line, word on
   next). chatter's current MAXWD output is a single-line table.
   Should `--display-mode 0` produce the two-line legacy form, or
   should we treat the table form as the modern default and only
   honour `+d1` (legal CHAT format)?
2. `+d1` "legal chat format" suggests the output is itself a CHAT
   file. That's a transform-flavoured output, not analyze. The
   chatter approach might be to route `--display-mode 1` to an
   explicit `chatter clan maxwd-extract` transform rather than
   overloading the analyze command.

## Output

Per speaker:

- Table of longest words sorted by length descending (up to `limit`)
- **All occurrences with line numbers** (matching CLAN)
- Maximum word length
- Mean word length
- Total and unique word counts

## Differences from CLAN

### Occurrence reporting

Reports **all occurrences with line numbers**, matching CLAN's output format exactly.

### Word normalization

Length is measured after stripping `+` (compound markers) and `'` (apostrophes), matching CLAN's character counting behavior.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
