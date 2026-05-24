# CHSTRING -- String Replacement Using a Changes File

**Status:** Current
**Last updated:** 2026-05-22 12:50 EDT

## Purpose

Reimplements CLAN's `chstring` command, which reads a changes file containing find/replace pairs (alternating lines) and applies text substitutions to main-tier words. Replacements are applied to all word nodes, including words inside annotated groups, replacement forms, and bracketed groups.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409309) for the original command documentation.

## Usage

```bash
chatter clan chstring --changes changes.cut file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c`, `--changes` | path | *(required)* | Path to the changes file containing find/replace pairs |
| `-o`, `--output` | path | stdout | Output CHAT file path |

## CLAN `+`-flag coverage audit

CHSTRING is a **transform** — it mutates CHAT input and writes
CHAT output. It does not emit a banner and does not share
`CommonAnalysisArgs` (no speaker/role/gem/range filters apply).
The flag set is entirely command-specific.

Sources: `OSX-CLAN/src/clan/chstring.cpp::usage`,
`crates/talkbank-clan/src/transforms/chstring.rs`,
`crates/talkbank-cli/src/cli/args/clan_commands.rs::Chstring`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### CHSTRING-specific `+`-flags (from `chstring.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b` | Work only on text right of the colon (CHAT format) | (default) | Done | chatter only mutates the main-tier word content; speaker codes are preserved. |
| `+cF` / `-c` | Dictionary file path / do not change inside `[...]` codes | `--changes <PATH>` (file form only) | Partial | chatter requires the path explicitly (no `changes.cut`-in-cwd default). The `-c` inside-codes guard is implicit — chatter's AST-based replacement only touches word leaves, never code-bracket content. |
| `+d` | Do not re-wrap tiers | — | Missing | Output line-wrapping is a separate concern; chatter never wraps on output. |
| `+l` | Work only on codes left of colon (speaker tag) | — | Missing | |
| `+lx` | Do not show the list of changes | (default) | Done | chatter operates silently. |
| `+q` | Clean up tiers (add tabs after colons, remove blank spaces) | — | Missing | Tier-cleanup pass. |
| `+q1` | Clean up tiers for CORELEX | — | Missing | |
| `+sS S` | Inline find/replace pair | — | Missing | All replacements must come via `--changes`. |
| `-w` | String-oriented search and replacement | (default) | Done | chatter's word-leaf replacement IS string-oriented. |
| `+x` | Interpret `*`, `_`, `\` as literal characters | — | Missing | chatter's matcher does not yet expose wildcard-vs-literal switching. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 3 |
| Partial | 1 |
| Missing | 6 |

CHSTRING is intentionally a thin transform in chatter — the
typed-AST design eliminates several CLAN flags by construction
(`+b`, `-c` inside-codes guard, `+lx`). The remaining gaps are
mostly orthogonal niceties (`+q` tier-cleanup, `+sS` inline pair,
`+x` literal-character mode); none change correctness of the
default file → file transform.

## Changes File Format

The changes file contains alternating lines of find and replace strings:

```text
find_text1
replace_text1
find_text2
replace_text2
```

The file must have an even number of non-empty lines. CLAN looks for `changes.cut` in the current directory by default; `chatter clan chstring` requires the path to be passed explicitly via `--changes`.

## Behavior

For each utterance in the file, the transform walks all word nodes on the main tier -- including words inside annotated groups, replacement forms, and bracketed groups -- and applies find/replace substitutions from the changes file.

## Differences from CLAN

- Operates on the parsed AST rather than raw text, ensuring structural integrity of the CHAT file after substitution.
- Does not support CLAN's regex-based pattern matching in the changes file.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
- **Golden test parity**: Verified against CLAN C binary output.
