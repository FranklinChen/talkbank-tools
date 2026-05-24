# CHIP -- Child/Parent Interaction Profile

**Status:** Current
**Last updated:** 2026-05-22 09:42 EDT

## Purpose

Analyzes interaction patterns between a child speaker and their conversational partners. Categorizes successive utterance pairs to measure imitation, repetition, and overlap. CHIP is commonly used in child language research to quantify how much a child imitates or echoes their interlocutor.

## Usage

```bash
chatter clan chip file.cha
chatter clan chip --speaker CHI file.cha
chatter clan chip --format json file.cha
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

Authoritative enumeration of every CLAN `chip` flag. Sources:

* `OSX-CLAN/src/clan/chip.cpp` — `usage()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` CHIP branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Chip` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### CHIP-specific `+`-flags (from `chip.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bS` | Speaker ID `S` is an adult | — | Missing | CHIP's role tagging — the partner-vs-child distinction is CHIP's whole purpose. |
| `+cS` | Speaker ID `S` is a child | — | Missing | |
| `+g` | Enable substitution option | — | Missing | Substitution coding. |
| `-hF` | File `F` has words to be excluded | — | Missing | Word-list exclusion. |
| `-nC` | Do not code `C: b` (adult), `c` (child), or `s` (asr/csr) responses | — | Missing | Response-type filter. |
| `+qN` | Set utterance window to `N` utterances before response | — | Missing | Context-window for response pairing. |
| `+wN` | Set minimum number of words on source utterance | — | Rewriter only | Collides with the inherited `+wN` context window (see KWAL). |
| `+xN` | Set minimum repetition index for coding | — | Missing | |
| `+dN` | Various display modes (full N table omitted here) | — | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 1 |
| Rewriter only | 6 |
| Missing | 12 |

CHIP's largest gap is the **adult/child speaker-role tagging**
(`+bS` / `+cS`) — without these, CHIP cannot determine which
utterance pair is being analysed (target child speaking, partner
speaking, both). chatter's CHIP today processes pairs but with
chatter-side heuristics; CLAN-compatible interaction profiling
requires the explicit role tags. The collision between CHIP's
`+wN` (source minimum-word count) and the inherited `+wN`
(context window from KWAL) is a documented overload trap.

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** The
> rewriter at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`, but no `clap` field consumes that token
> today. Drafted from CLAN manual §7.4.5 (`Unique Options`, CHIP) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Using `+d` with no further number outputs only coding tiers, which are useful for iterative analyses." |
| `+d1` | "Using `+d1` outputs only summary statistics, which can then be sent to a statistical program." |

### Open questions for PI review

1. CHIP's `+d` shape (coding-tier-only vs summary-only) is orthogonal
   to the FREQ/KWAL/MLU shape (output format selectors). Should the
   `--display-mode` enum's variants be CHIP-specific
   (`coding-tiers` / `summary`) or share a name space with FREQ's
   variants?
2. "Useful for iterative analyses" implies the coding-tier output is
   intended to be piped to another `chatter clan` command. Should
   chatter prefer making this the *default* JSON output shape, with
   `--display-mode summary` collapsing to just the matrix?

## Interaction Categories

For each adjacent utterance pair (speaker A followed by speaker B):

| Category | Condition |
|----------|-----------|
| **Exact repetition** | B's words are identical to A's (order-independent) |
| **Overlap** | B shares >= 50% of words with A (smaller unique-word set as denominator) |
| **No overlap** | B shares < 50% of words with A |

Only cross-speaker adjacency is considered; consecutive utterances by the same speaker do not produce interaction records. Adjacency state is reset at file boundaries.

## Output

**36-measure matrix format** structurally matching CLAN:

- ADU (adult) / CHI (child) / ASR (adult-speech-related) / CSR (child-speech-related) columns
- Per directed speaker pair (MOT->CHI is distinct from CHI->MOT)
- Counts and percentages for each interaction category
- Grand totals across all pairs

**Current implementation status (2026-05-12):** the matrix header,
row labels, and ADU/CHI/ASR/CSR column layout are implemented and
written by `chip.rs:246-260`. The actual *measure values* are
written as zeros — `chip.rs:253` carries a comment "currently all
zeros — full computation not yet implemented." So the structural
parity is in place but the analytical numbers are not yet
computed. See §Golden test parity below for the consequence.

### Echo behavior

When displaying matched utterances, CHIP echoes:
- Main tier text
- `%mor` tier (if present)

It does **not** echo `%gra` tiers, matching CLAN's behavior.

## Differences from CLAN

### Matrix format

Uses the exact **36-measure matrix format** with ADU/CHI/ASR/CSR columns. Header, row labels, and column structure match CLAN character-for-character; per §Output, the measure-value cells are currently emitted as zeros pending full implementation.

### Echo content

Echoes main tier + `%mor` only (not `%gra` tiers), matching CLAN.

### Word identification

Uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching. Overlap comparison operates on parsed word content, not raw text.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

No CHIP-specific golden tests are currently checked in (`rg
'chip_golden|chip.*golden'` returns nothing in `tests/` and
`crates/talkbank-clan/`). The previous "100% parity" claim was
incorrect given the stub measure values described above. Once the
36 measures are computed, golden tests should be added against a
real CLAN CHIP run.
