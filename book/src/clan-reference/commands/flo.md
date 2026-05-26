# FLO -- Simplified Fluent Output

**Status:** Current
**Last updated:** 2026-05-26 11:43 EDT

## Purpose

Reimplements CLAN's `flo` command, which generates a `%flo:` dependent tier containing a simplified, "fluent" version of each utterance's main line.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409312) for the original command documentation.

## Usage

```bash
chatter clan flo file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

FLO is a **transform** — input CHAT in, output CHAT (with the
new `%flo:` tier) out. No banner; no shared `CommonAnalysisArgs`.

Sources: `OSX-CLAN/src/clan/flo.cpp::usage`,
`crates/talkbank-clan/src/transforms/flo.rs`.

### FLO-specific `+`-flags (from `flo.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Do not strip `@…` suffixes (default strips `word@s` → `word`) | — | Missing | chatter currently always strips. |
| `+cm` | Filter main tier the way `mor` does | — | Missing | Mor-grade pre-cleanup. |
| `+cr` | Filter main tier + remove speaker codes and delimiters | — | Missing | Stripped output. |
| `+cjC` | Filter main tier for JSON comparison (`C-`/`p-`/`s-` strip variants) | — | Missing | Pipeline-internal mode. |
| `+ca` | Create output for MFA aligner | — | Missing | Niche format. |
| `+cb` | Keep bullets in output | — | Missing | Bullet-preservation toggle. |
| `+d` | Replace main tier with simplified `%flo` in output | — | Missing | `OSX-CLAN/src/clan/flo.cpp:197`: bare `+d` (or `+d0`) sets `substitute_flag = 1` (flo line replaces main line). chatter emits `%flo:` as a new dependent tier alongside the original main tier — no substitute-flag consumer. Per-FLO rewriter arm in `clan_args.rs` passes the token through so clap reports the literal `+d` argument rather than the misleading `--display-mode` rewrite. |
| `+d1` | FAVE-formatted output | — | Missing | Same `flo.cpp:197` switch sets `substitute_flag = 2`. Per-FLO rewriter arm passes the token through. |
| `+d2` | AntConc-formatted `.txt` with BOM | — | Missing | Same `flo.cpp:197` switch — `+d2` is an empty branch (no `substitute_flag` change). Per-FLO rewriter arm passes the token through. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 0 |
| Partial | 0 |
| Missing | 9 |

FLO has the **largest "Missing" bucket relative to surface size** of
any audited command — the chatter implementation supports only the
default `flo file.cha` invocation (emit `%flo` tier, keep main).
Every variant flag is missing. Most of the variants are niche
output-format selectors (FAVE, MFA, AntConc) rather than analysis-
shape changes; defer until a researcher requests one.

## Behavior

Processing steps:

1. Strips all header lines (no `@UTF8`, `@Begin`, `@End`, etc.)
2. Adds a `%flo:` dependent tier to each utterance containing the simplified main line: just countable words plus the terminator
3. Strips retrace targets (words/groups before `[/]`, `[//]`, `[///]`, `[/-]` — the four `RetraceKind` variants per `crates/talkbank-model/src/model/content/retrace.rs`)
4. Strips non-countable words (`xxx`/`yyy`/`www`, `0word`, `&~frag`, `&-um`)
5. Strips events (`&=thing`) and pauses
6. For replaced words (`[: form]`), uses the replacement (corrected form)
7. Keeps existing dependent tiers (`%mor`, `%gra`, etc.)

The `%flo:` tier is inserted at position 0 (before other dependent tiers).

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
