# WDLEN -- Word Length Distribution

**Status:** Current
**Last updated:** 2026-05-22 09:05 EDT

## Purpose

Computes six distribution tables matching CLAN's output format. WDLEN provides detailed histograms of word and utterance lengths, useful for studying vocabulary complexity and utterance structure development.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409247) for the original WDLEN command specification.

## Usage

```bash
chatter clan wdlen file.cha
chatter clan wdlen --speaker CHI file.cha
chatter clan wdlen --format json file.cha
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

Authoritative enumeration of every CLAN `wdlen` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/wdlen.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` WDLEN branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Wdlen` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### WDLEN-specific `+`-flags (from `wdlen.cpp::getflag`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Compute both words and morphemes from `%mor` only (no main tier) | — | Missing | CLAN's "depend-only" mode. chatter's sections 5/6 already use `%mor` per the [six-section design](#six-distribution-sections), so the divergence is whether sections 1–4 use main tier or skip. |
| `+bS` | Add chars in `S` to morpheme-delimiter list | — | Missing | Morpheme-boundary customization. |
| `-bS` | Remove chars in `S` from delimiter list (`-b` clears all) | — | Missing | |
| `+cS` | Clause-marker delimiter `S` | — | Missing | |
| `+c@F` | Clause markers from file `F` | — | Missing | |

### General `+`-flags WDLEN inherits (from `cutt.cpp::mainusage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+t*X` / `-t*X` | Include/exclude speaker | `--speaker` / `--exclude-speaker` | Done | `+tX` accepted post-2026-05-21. |
| `+t%X` / `-t%X` | Include/exclude dependent tier | `--tier` / `--exclude-tier` (rewriter target) | Rewriter only | |
| `+t@ID="..."` | Filter by @ID pattern | `--id-filter` | Done | |
| `+t#ROLE` | Filter by role | `--role` | Done | Fixed 2026-05-22; see [FREQ](./freq.md) for the shared implementation. |
| `+s"word"` / `-s"word"` | Include/exclude word | `--include-word` / `--exclude-word` | Done | WDLEN docs say `+sm` is the recommended search-spec form for `%mor`-tier searches; chatter does not yet special-case the `m` suffix. |
| `+s@F` / `-s@F` | Search / exclude words from file | `--include-word-file` / `--exclude-word-file` | Done | Landed 2026-05-22. File format: one pattern per line; blank lines, `# `-comments, and `;%* `-annotation lines skipped. Repeatable. |
| `+gX` | Gem filter | `--gem` | Done | |
| `+zN-M` | Utterance range | `--range` | Done | |
| `+rN` | Retrace / clitic / prosodic controls | `--include-retracings` (`+r6`) | Partial | |
| `+u` | Combine across files | (default) | Done | Inverse default vs CLAN. |
| `+re` | Recurse | (default) | Done | |
| `+pS` | Word delimiter | — | Missing | |
| `+k` | Case-sensitive | `--case-sensitive` (rewriter target) | Rewriter only | |
| `+wN` / `-wN` | Context window | `--context-window` (rewriter target) | Rewriter only | |
| `+f` / `+fEXT` | Output to file | `--output-ext` (rewriter target) | Rewriter only | Phase 1.1. |

### Audit summary

| Bucket | Count |
|---|---|
| Done (byte-parity or in scope) | 8 |
| Partial | 2 |
| Rewriter only | 5 |
| Missing | 6 |

WDLEN's specific gaps (`+a` depend-only mode, `+bS`/`-bS`
morpheme-delimiter customization, `+cS` clause delimiters) all
affect *what counts as a unit* in the six distribution
sections — i.e., they change the numbers in the histograms.
Filed as Phase 1.7 follow-ups.

## Six Distribution Sections

| Section | What it measures | Source |
|---------|-----------------|--------|
| 1. Word lengths in characters | Character count per word | Main tier |
| 2. Utterance lengths in words | Word count per utterance | Main tier |
| 3. Turn lengths in utterances | Utterances per turn | Main tier |
| 4. Turn lengths in words | Words per turn | Main tier |
| 5. Word lengths in morphemes | Morphemes per word (stem + Brown's suffixes) | `%mor` tier |
| 6. Utterance lengths in morphemes | Morphemes per utterance (POS + stem + Brown's suffixes) | `%mor` tier |

Each section shows a histogram (value -> count), mean, and total.

## Differences from CLAN

### Brown's morpheme rules

Sections 5 and 6 use distinct counting methods:

- **Section 5**: stem + Brown's suffix count (no POS tag counted). Clitic pairs (`~`) are merged as one word.
- **Section 6**: POS tag + stem + Brown's suffix count. POS is counted only for the main word (not clitics).

Brown's suffix strings: `PL`, `PAST`, `Past`, `POSS`, `PASTP`, `Pastp`, `PRESP` (same 7 as MLU).

### Character counting

CLAN strips apostrophes before counting character length. Our implementation matches this behavior.

### Speaker ordering

CLAN outputs speakers in reverse encounter order (an artifact of its C linked-list prepend pattern). Our implementation replicates this ordering for parity.

### XML footer

CLAN appends `</Table></Worksheet></Workbook>` XML tags at the end of output. Our implementation matches this for parity.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
