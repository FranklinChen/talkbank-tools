# MORTABLE -- Morphological Category Cross-Tabulation

**Status:** Current
**Last updated:** 2026-05-22 09:49 EDT

## Purpose

Produces a per-speaker frequency table of morphosyntactic categories by matching POS tags from the `%mor` tier against patterns defined in a language-specific script file.

Requires a language script file (e.g., `eng.cut`) that defines patterns and their labels for categorizing morphemes from the `%mor` tier. Each rule line contains a quoted label and `+`/`-` prefixed POS patterns. Rules can be grouped as OR (first match wins) or AND (all must match).

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409286) for the original MORTABLE command specification.

> **Note on `-f` short flag.** Both `--script` and `--format`
> declare `-f` as their short flag in
> `crates/talkbank-cli/src/cli/args/clan_commands.rs:260` and
> `:345`. clap currently accepts this — `chatter clan mortable
> --help` runs successfully and `-f test.cut` resolves to
> `--script` — so the previous warning that the command was
> "unusable" no longer applies. Verified 2026-05-12 by invoking
> `chatter clan mortable --help` and `chatter clan mortable -f
> test.cut file.cha`. Prefer the long form `--script` /
> `--format` in scripts to avoid ambiguity in future clap
> upgrades.

## Usage

```bash
chatter clan mortable --script eng.cut file.cha
chatter clan mortable --script eng.cut --speaker CHI file.cha
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--script <PATH>` | `+lF` | Language script file (.cut) — required |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

MORTABLE is a **required-flag refusal** command in chatter — emits
the exact CLAN refusal when `+lF` (or `--script F`) is absent.

### MORTABLE-specific `+`-flags (from `mortable.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+lF` | Specify language script file `F` (required) | `--script <PATH>` | Done | Direct mapping; rewriter routes `+lF` → `--script F` for MORTABLE (since the per-subcommand routing batch). |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | |
| `+o4` | Output raw values instead of percentage values | — | Rewriter only | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 6 |
| Partial | 1 |
| Rewriter only | 4 |
| Missing | 4 |

## Differences from CLAN

- **POS matching**: Operates on parsed `%mor` tier data rather than raw text line scanning.
- **POS matching detail**: POS tags are read directly from typed `%mor` items instead of reparsing serialized `%mor` content.
- **Script file format**: Compatible with CLAN's `.cut` files.
- **Output formats**: Supports text, JSON, and CSV formats.
- **Deterministic ordering**: `BTreeMap` ordering ensures deterministic output across runs.
- **Golden test parity**: Verified against CLAN C binary output.
