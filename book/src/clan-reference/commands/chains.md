# CHAINS -- Clause Chain Analysis

**Status:** Current
**Last updated:** 2026-05-22 09:45 EDT

## Purpose

Analyzes sequences of interactional codes on a single specified coding tier. In the legacy manual, `CHAINS` is described as tracking hand-entered codes on one coding tier, with `%cod`, `%spa`, and `%sit` given as examples, and with dependent-tier periods used to mark correspondence to `[^c]` clause units on the main tier.

In `talkbank-clan`, a "chain" is a run of consecutive utterances by the same speaker that all contain a given code item on the selected coding tier. When the code disappears or the speaker changes, the chain is flushed and its length is recorded.

Reports chain count, average/min/max length, and standard deviation per code and speaker.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409147) for the original CHAINS command specification.

## Usage

```bash
chatter clan chains file.cha
chatter clan chains --speaker CHI file.cha
chatter clan chains --tier cod file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--tier <NAME>` | `+t%X` (rewriter target) | Tier label to read codes from (default: `cod`) |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `chains` flag. Sources:

* `OSX-CLAN/src/clan/chains.cpp` — `usage()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` CHAINS branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Chains` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

CHAINS is a **required-flag refusal** command in chatter — emits
the exact CLAN refusal when `+t%X` (or `--tier X`) is absent.

### CHAINS-specific `+`-flags (from `chains.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+cS` / `+c@S` | Clause-marker delimiter `S` (or markers in file `@S`) | — | Missing | Clause-unit detection. |
| `+d` | Change zeroes to spaces in output | — | Rewriter only | |
| `+d1` | `+d` + display every input line in output | — | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 6 |
| Partial | 1 |
| Rewriter only | 5 |
| Missing | 5 |

## Display Modes (`+dN` / `--display-mode N`) — DRAFT, awaiting PI review

> **Status: drafted from CLAN manual; not yet implemented.** Rewriter
> at `crates/talkbank-clan/src/clan_args.rs:101` translates
> `+dN` → `--display-mode N`; no `clap` field consumes it today.
> Drafted from CLAN manual §7.1.2 (`Unique Options`, CHAINS) for
> PI review.

| N | CLAN behavior (verbatim from manual) |
|---|---|
| `+d` (no number) | "Use this switch to change zeroes to spaces in the output." Example: `chains +d +t%spa chains.cha +s$nia:%` |
| `+d1` | "Works the same as `+d`, while also displaying every input line in the output." |

### Open questions for PI review

1. "Change zeroes to spaces" is a CLAN text-formatting artifact —
   columnar output where `0` marks absence. Does chatter's typed
   output already render absence as empty/space? If so, `--display-mode 0`
   may be a no-op in chatter and the flag should error rather than
   silently accept.
2. `+d1` "displays every input line": this is a verbose-trace mode,
   not an output-shape change. Map to `--include-input` boolean
   instead of overloading `--display-mode`?

## Output

Per code and speaker:

- Number of separate chains
- Average chain length
- Standard deviation of chain lengths (sample, N-1)
- Minimum chain length
- Maximum chain length

## Differences from CLAN

- **Manual intent**: The legacy manual frames `CHAINS` as a coding-tier command, not a generic dependent-tier tokenizer.
- **Speaker-change detection**: Flushes all open chains for the previous speaker, matching CLAN's behavior of treating chains as speaker-scoped.
- **Coding-tier semantics**: `CHAINS` operates on `%cod`-style coding items, not `%gra`.
- **Code extraction**: For `%cod`, chains now read a clan-local semantic `%cod` item stream derived from the parsed AST rather than flattening the tier back to whitespace text. For other tiers passed via `--tier`, extraction remains content-token based.
- **Manual constraint not yet fully enforced**: The manual examples all use coding tiers. Non-`%cod` `--tier` fallback still tokenizes tier text; this should eventually be restricted to tiers with explicit coding semantics.
- **Standard deviation**: Uses the sample (N-1) formula rather than population.
- **Golden test parity**: Verified against CLAN C binary output.
