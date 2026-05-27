# LAB2CHAT -- LAB Timing Labels to CHAT Conversion

**Status:** Current
**Last updated:** 2026-05-27 10:39 EDT

## Purpose

Converts LAB (label) timing files into CHAT format. LAB files contain time-aligned word or segment labels commonly used in speech research tools (e.g., HTK, Kaldi).

## Usage

```bash
chatter clan lab2chat input.lab
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--speaker` | `"SPK"` | Speaker code for all utterances |
| `-L`, `--language` | `"eng"` | ISO 639 language code for the `@Languages` header (note: uppercase `-L` because lowercase `-l` would conflict) |
| `-o`, `--output` | stdout | Output CHAT file path |

The corpus name in `@ID` headers is hardcoded to `"lab_corpus"`
(`crates/talkbank-clan/src/converters/lab2chat.rs:110`); there is
no CLI flag to override it.

## Supported Formats

- **Three-column**: `start_time end_time label` (times in seconds)
- **Two-column**: `time label` (end time inferred from the next entry)

## Input Format

Plain text files with whitespace-separated columns. Silence markers (`sil`, `sp`, `#`) are skipped during conversion. Comment lines starting with `#` and blank lines are ignored.

Example:

```text
0.0 0.5 hello
0.5 1.2 world
1.2 1.5 sil
```

## Output

A well-formed CHAT file where each non-silence label becomes a separate utterance with timing bullets derived from the LAB timestamps (converted from seconds to milliseconds).

## Differences from CLAN

- Uses typed AST for CHAT generation
- Produces valid, well-formed CHAT output

## CLAN `+`-flag coverage audit

LAB2CHAT is a **converter** — input WaveSurfer `.lab` files,
output CHAT. Sources: `OSX-CLAN/src/clan/lab2chat.cpp::usage`,
`crates/talkbank-clan/src/converters/lab2chat.rs`.

### LAB2CHAT-specific `+`-flags (from `lab2chat.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| `+dF` | Tags-dependencies file `F` | — | Missing |
| `+fN` | (CLAN-internal) | — | Missing |
| `+mF` | Movie file name `F` (default: input file name) | — | Missing |
| `+p` | Plain file conversion (default: merge per attribute file) | — | Missing |
| `+oS` | Code page selection | — | Missing |
| `+tN` | Movie segment start time offset | — | Missing | Per-LAB2CHAT rewriter arm in `clan_args.rs` returns None for digit-only `+tN` so the literal token passes through to clap (which rejects it) rather than silently mis-routing to `--speaker N` via the generic `+t` → `rewrite_tier_speaker` default branch. |
| `+re` | Recurse subdirectories | (default for directory input) | Done |

Audit summary: 1 Done, 6 Missing. LAB2CHAT's metadata-rich
options (tags, movie association, code-page) are the biggest
gap; chatter handles the timing-label → utterance core but
not the surrounding metadata machinery.
