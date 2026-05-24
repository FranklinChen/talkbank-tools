# FLUCALC -- Fluency Calculation

**Status:** Current
**Last updated:** 2026-05-22 09:22 EDT

## Purpose

Detects and quantifies disfluencies in speech transcripts, producing per-speaker counts of stuttering-like disfluencies (SLD) and typical disfluencies (TD). FLUCALC is the standard tool in CLAN for analyzing fluency in stuttering research.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409273) for the original FLUCALC command specification.

## Usage

```bash
chatter clan flucalc file.cha
chatter clan flucalc --speaker CHI file.cha
chatter clan flucalc --format json file.cha
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

Authoritative enumeration of every CLAN `flucalc` flag. Sources:

* `OSX-CLAN/src/clan/flucalc.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` FLUCALC branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Flucalc` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### FLUCALC-specific `+`-flags (from `flucalc.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Get pause durations from `%wor` tier | — | Missing | chatter ignores `%wor` for pauses. |
| `+b` | Word-mode analyses (default: syllable mode) | — | Missing | chatter is word-mode only. |
| `+b1` | Word mode with repetition/retraces | — | Missing | |
| `+c` | Compute pause duration from `(2.4)` notation | — | Missing | Inline pause-time parsing. |
| `+c1` | Add phonological fragment to SLD instead of TD | — | Missing | Counting policy switch. |
| `+c5` | With `+p`: reverse source/target tier priority | — | Missing | |
| `+dN` | Sample size `N` (s, w) — e.g. `+d100s` for 100 syllables | — | Rewriter only | `--display-mode N`; no consuming clap field. |
| `+e1`..`+e5` | Side-effect file creation (syllable counts, fluent utterances, disfluencies) | — | Missing | |
| `+u` | Compute output per utterance | partial via `--per-file` | Partial | Different granularity. |
| `+pS` / `+p@F` | Search word `S` and match POS | — | Missing | |
| `+g` / `-g` / `+gS` | Gem semantics (same overload as EVAL) | `--gem` (S form only) | Partial | |
| `+n` / `-n` | Gem termination semantics | — | Missing | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 4 |
| Partial | 2 |
| Rewriter only | 5 |
| Missing | 13 |

FLUCALC's largest gap is the **syllable-vs-word mode** (`+b` vs
default syllable). chatter operates exclusively in word mode,
which produces *different counts* from CLAN's default. This is a
**silent wrong** for researchers expecting CLAN-default
syllable-mode output. Filed as a Phase 1.7 high-priority
follow-up.

## Disfluency Categories

### Stuttering-Like Disfluencies (SLD)

| Type | CHAT notation | Example | Status |
|------|--------------|---------|--------|
| Prolongation | `:` within word | `wa:nt` | Implemented |
| Broken word | `^` notation | `base^ball` | Implemented |
| Whole-word repetition | Consecutive identical words | `I I want` | Implemented |
| Part-word repetition | -- | -- | Partial |
| Block | -- | -- | Partial |

### Typical Disfluencies (TD)

| Type | CHAT notation | Example | Status |
|------|--------------|---------|--------|
| Phrase repetition | `[/]` | `I want [/] I want` | Implemented |
| Revision | `[//]` | `I want [//] I need` | Implemented |
| Filled pause | `&-` prefix | `&-uh`, `&-um` | Implemented |
| Phonological fragment | `&+` prefix | `&+fr` | Implemented |

### Output measures

All counts are reported as:
- **Raw values**: Total count per disfluency type
- **Percentages**: Per 100 words

## Algorithm

1. For each utterance, walk the AST content nodes
2. Identify disfluency markers:
   - Retrace groups (`[/]`, `[//]`) from parsed AST annotations
   - Fillers and fragments from word category annotations
   - Prolongations and broken words from within-word notation
   - Whole-word repetitions from consecutive identical countable words
3. Accumulate per-speaker counts by disfluency category
4. Compute percentages relative to total words (or syllables)

## Differences from CLAN

### Detection method

Some categories (specifically `[/]` and `[//]` retrace markers) are currently counted via substring matching on the serialized CHAT text rather than fully through the parsed AST. This produces equivalent results but is a known area for future improvement.

### Part-word repetitions and blocks

Counted via CHAT notation markers rather than acoustic analysis. Full detection of these categories requires audio-linked analysis that is beyond the scope of text-based transcript processing.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

Verified against CLAN C binary output.
