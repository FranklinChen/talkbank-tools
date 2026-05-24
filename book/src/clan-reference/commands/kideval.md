# KIDEVAL -- Child Language Evaluation

**Status:** Current
**Last updated:** 2026-05-22 09:20 EDT

## Purpose

Produces a comprehensive child language evaluation report by combining multiple analysis methods into a single per-speaker summary. KIDEVAL is designed for evaluating children's language development and aggregates results from several individual CLAN commands.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409281) for the original KIDEVAL command specification.

## Usage

```bash
chatter clan kideval file.cha
chatter clan kideval --speaker CHI file.cha
chatter clan kideval --format json file.cha
chatter clan kideval --dss-rules english.scr file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--dss-rules <PATH>` | — | Custom DSS rules file (.scr) — chatter extension |
| `--ipsyn-rules <PATH>` | `+lF` (Phase 1.7 follow-up) | Custom IPSYN rules file |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

KIDEVAL does not expose per-component utterance caps as CLI flags
(despite the standalone DSS / IPSYN commands having `--max-utterances`
of their own); the combined report uses each component's built-in
default (DSS 50, IPSYN 100). To override, run those components
separately and aggregate.

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `kideval` flag, mapped
against chatter's coverage. Sources:

* `OSX-CLAN/src/clan/kideval.cpp` — `usage()` and `getflag()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` KIDEVAL branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Kideval` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### KIDEVAL-specific `+`-flags (from `kideval.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+aN` | Limit to files with > N utterances | — | Missing | Sample-size guard. |
| `+bS` / `-bS` | Morpheme-delimiter customization | — | Missing | Shared with WDLEN/MAXWD/EVAL. |
| `+dtd~S` / `+dnarrative~S` / similar | Database keyword filters (age range, sex) | — | Missing | KIDEVAL's database-comparison engine; not wired in chatter. |
| `+eS`, `+e+`, `+e-`, `+e1` | Dataset keyword / metadata listing | — | Missing | Database-side. |
| `+lF` | Language script file (eng, engu, fra, jpn, nld, spa, yue, zho) | partial via `--ipsyn-rules` / `--dss-rules` | Partial | chatter exposes only the per-component rules files; CLAN's `+lF` picks a packaged language bundle. |
| `+n` / `-n` | Gem termination semantics | — | Missing | |
| `+oN` | Output format/verbosity sub-modes | — | Rewriter only | Falls to `--display-mode` rewrite. |
| `+s` | Postcode filter `[+ ...]` | partial via `--include-word` | Partial | Different semantic — chatter filters by main-tier word, CLAN's KIDEVAL `+s` matches postcodes. |

### General `+`-flags KIDEVAL inherits (from `cutt.cpp::mainusage`)

Same shape as MLU / FREQPOS. KIDEVAL is currently a **required-
flag refusal** command in chatter — invoking without `+t*X` (or
`--speaker X`) emits CLAN's exact `Please specify at least one
speaker tier code with "+t" option on command line.` to stderr
and exits 1. That refusal byte-parity is verified.

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 3 |
| Rewriter only | 5 |
| Missing | 11 |

KIDEVAL's largest gaps are the **database-comparison engine**
(`+dtd~`, `+dnarrative~`, `+eS`, `+lF` language bundles). These
are KIDEVAL's whole point — comparison to age-/sex-stratified
normative samples — and currently chatter's KIDEVAL emits the
combined-metrics report without the comparison overlay. Filed
as a Phase 1.7 follow-up.

## Combined Metrics

KIDEVAL produces a single report combining:

| Metric | Source | Details |
|--------|--------|---------|
| MLU (words and morphemes) | Main tier + `%mor` | See [MLU](mlu.md) |
| NDW / TTR | Main tier word types/tokens | See [FREQ](freq.md) |
| DSS score | `%mor` tier | See [DSS](dss.md) |
| VOCD (D statistic) | Main tier words | See [VOCD](vocd.md) |
| IPSyn score | `%mor` tier | See [IPSYN](ipsyn.md) |
| POS category counts | `%mor` tier | Nouns, verbs, auxiliaries, etc. |
| Error counts | `[*]` markers | Word-level errors |

This is the primary tool for clinical assessment of child language samples, providing a comprehensive profile in a single command invocation.

## Differences from CLAN

### VOCD simplification

KIDEVAL uses a simplified TTR-based D estimate rather than the full bootstrap sampling approach used by the standalone [VOCD](vocd.md) command. This trades precision for speed when computing the combined report.

### IPSYN rules

Uses the built-in simplified rule subset unless a custom rules file is provided via `--ipsyn-rules`. For full 56-rule coverage, supply the official IPSYN rules file.

### DSS rules

Uses the built-in simplified rule subset unless a custom rules file is provided via `--dss-rules`. For full clinical scoring, supply a complete `.scr` rules file.

### AST-based analysis

All component analyses share the same AST-based infrastructure, ensuring consistent word identification and morpheme counting across all metrics. In CLAN, each component command has its own independent word-filtering logic, which can lead to subtle inconsistencies.

### Golden test parity

Verified against CLAN C binary output.
