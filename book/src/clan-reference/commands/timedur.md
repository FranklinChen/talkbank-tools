# TIMEDUR -- Time Duration

**Status:** Current
**Last updated:** 2026-05-22 09:41 EDT

## Purpose

Computes time duration statistics from media timestamp bullets attached to utterances. Utterances without bullet timing are silently skipped.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409240) for the original TIMEDUR command specification.

## Usage

```bash
chatter clan timedur file.cha
chatter clan timedur --speaker CHI file.cha
chatter clan timedur --format json file.cha
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

Authoritative enumeration of every CLAN `timedur` flag. Sources:

* `OSX-CLAN/src/clan/timedur.cpp` — `usage()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` TIMEDUR branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Timedur` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

TIMEDUR has **no command-specific flags** beyond the general
inherited set. All variation is in `+d` display modes:

### TIMEDUR-specific flags (from `timedur.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+d` | Default results in spreadsheet format | — | Rewriter only | `--display-mode` rewrite. |
| `+d1` | Ratio of words/utterances over time duration | — | Rewriter only | |
| `+d10` | `+d1` results in spreadsheet format | — | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 6 |
| Partial | 1 |
| Rewriter only | 6 |
| Missing | 4 |

TIMEDUR's gap profile is dominated by `+dN` display modes. Body
parity already matches CLAN. The default (`+d`) emits in
spreadsheet format — chatter's text/json/csv formats cover the
spreadsheet use case but with different shapes; researchers
expecting CLAN's exact spreadsheet column layout need
`--format csv` plus the layout audit.

## Output

Per speaker:

- Number of timed utterances
- Total duration (formatted as HH:MM:SS.mmm)
- Mean utterance duration
- Min/max duration

Plus a corpus-wide summary:

- Total timed utterances across all speakers
- Total duration
- Recording span (earliest start to latest end)
- Speaker interaction matrix (overlap and gap analysis)

## Differences from CLAN

### Timestamp extraction

Uses parsed media bullet structures from the AST (`Bullet { start_ms, end_ms }`) rather than raw byte scanning in text. This is more robust against formatting variations.

### Interaction matrix header

The interaction matrix header includes a leading space, matching CLAN exactly. This was verified during golden test parity work.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

100% parity with CLAN C binary output.
