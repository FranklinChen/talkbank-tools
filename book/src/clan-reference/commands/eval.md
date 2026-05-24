# EVAL -- Language Sample Evaluation

**Status:** Current
**Last updated:** 2026-05-22 09:21 EDT

## Purpose

Comprehensive morphosyntactic analysis computing lexical diversity, grammatical category counts, error rates, and MLU. EVAL was originally designed for clinical evaluation of adult aphasic speech samples (Saffran, Berndt & Schwartz, 1989) and produces a detailed profile of morphosyntactic abilities.

Requires a `%mor` dependent tier for morpheme-level metrics. Word-level metrics are computed from the main tier regardless.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc87376473) for the original EVAL command specification.

## Usage

```bash
chatter clan eval file.cha
chatter clan eval --speaker CHI file.cha
chatter clan eval --format json file.cha
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

Authoritative enumeration of every CLAN `eval` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/eval.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` EVAL branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Eval` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

EVAL is a **required-flag refusal** command in chatter — invoking
without `+t*X` (or `--speaker X`) emits CLAN's exact `Please
specify at least one speaker tier code with "+t" option on command
line.` to stderr and exits 1. That refusal byte-parity is verified.

### EVAL-specific `+`-flags (from `eval.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+bS` / `-bS` | Morpheme-delimiter customization | — | Missing | Shared with WDLEN/MAXWD/KIDEVAL. |
| `+dS` | Database keyword (Anomic, Global, Broca, Wernicke, control, Fluent, Nonfluent, AllAphasia, …) | — | Missing | EVAL's whole reason for existing — AphasiaBank normative comparison. |
| `+e1` | Create list of database files used | — | Missing | Database-side. |
| `+e2` | Create proposition word list per file | — | Missing | |
| `+g` (no S) | Gem tier should contain all words specified by `+gS` | — | Missing | EVAL-specific override of the inherited gem semantic. |
| `-g` | Look for gems in database only | — | Missing | |
| `+gS` | Select gems labelled `S` | `--gem` | Done | |
| `+lF` | Language database file (eng, fra) | — | Missing | |
| `+n` / `-n` | Gem termination semantics | — | Missing | |
| `+o4` | Output raw values instead of percentages | — | Rewriter only | Falls to `--display-mode` rewrite. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 0 |
| Rewriter only | 5 |
| Missing | 14 |

EVAL's largest gap is the **AphasiaBank database-comparison
engine** (`+dS`, `+lF`, `+e1`, `+e2`). Without `+dS` filters,
chatter's EVAL produces a self-contained morphosyntactic profile
without the clinical-norm overlay that is EVAL's purpose. Filed
as a Phase 1.7 follow-up alongside KIDEVAL's database engine.

EVAL has no command-specific flags beyond the shared `CommonAnalysisArgs`
set; the per-speaker normative-comparison aspect lives in EVAL-D
(DementiaBank norms) and KIDEVAL (child norms), invoked as separate
subcommands.

## Metrics

EVAL produces a comprehensive profile per speaker:

### Lexical measures
- **Utterances**: Total utterance count
- **Total words**: All countable words
- **NDW**: Number of different words (types)
- **TTR**: Type-token ratio (types / tokens)

### MLU
- **MLU-w**: Mean length of utterance in words
- **MLU-m**: Mean length of utterance in morphemes (from %mor)

### Part-of-speech counts (from %mor)
- Nouns, verbs, auxiliaries, modals
- Prepositions, adjectives, adverbs
- Conjunctions, determiners, pronouns

### Inflectional morphology (from %mor)
- Plurals (`PL`)
- Past tense (`PAST`)
- Present participle (`PRESP`)
- Past participle (`PASTP`)

### Error and ratio measures
- **Word errors**: Count of `[*]` markers
- **Open/closed class ratio**: Content words vs function words

## Differences from CLAN

### Word and morpheme identification

Uses AST-based `is_countable_word()` and typed POS categories instead of CLAN's string-prefix matching. POS classification operates on structured `MorWord` types rather than parsing POS tag strings at analysis time.

### Error extraction

`[*]` error markers are extracted from parsed AST annotations (the `ErrorMarker` node type) rather than raw text pattern matching. This ensures accurate counting even with complex nested annotations.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only. JSON output provides structured access to all metrics for programmatic use.

### Golden test parity

Verified against CLAN C binary output.

## EVAL-D variant

`chatter clan eval-d` is identical to EVAL but uses DementiaBank protocol norms instead of AphasiaBank norms for normative comparison. See [EVAL-D](eval-d.md) for details.
