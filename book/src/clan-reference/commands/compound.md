# COMPOUND -- Normalize Compound Word Formatting

**Status:** Current
**Last updated:** 2026-05-22 12:55 EDT

## Purpose

Reimplements CLAN's COMPOUND command, which normalizes compound word notation in CHAT files. In CHAT, compound words are joined with `+` (e.g., `ice+cream`). This command converts dash-joined compounds to the canonical plus notation.

## Usage

```bash
chatter clan compound file.cha
```

## Options

COMPOUND has no command-specific flags — only the input `path`
positional and the optional shared `-o`/`--output` (default: stdout).
The dash→plus normalization is unconditional: there is no
`--dash-to-plus` switch in the current CLI.

## CLAN `+`-flag coverage audit

COMPOUND is a **transform**. CLAN's `compound.cpp::usage` is an
empty stub (`void usage() {}`), and the command-specific
`getflag` is `#pragma unused (f, f1, i)` — i.e. CLAN itself
exposes **no `+`-flags** for this command. Surface byte-parity is
trivially complete.

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| _(none)_ | — | default transform | Done |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 |
| Missing | 0 |

## Behavior

The transform walks all main-tier word nodes and converts dash-joined compounds to plus notation (e.g., `ice-cream` becomes `ice+cream`).

Operations performed:

- Normalize dash-joined compounds to plus notation: `ice-cream` -> `ice+cream`
- Preserves filler prefixes (`&-uh`) and omission prefixes (`0word`)
- Only converts when all parts are purely alphabetic

The transform recurses into annotated words, replacement forms, groups, and annotated groups.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
