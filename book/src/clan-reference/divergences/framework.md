# Framework-Level Divergences

**Status:** Current
**Last updated:** 2026-05-12 21:21 EDT

These differences apply across all commands, not to specific ones.

## CLI syntax

| Aspect | Legacy CLAN | Rust CLAN |
|--------|-------------|-----------|
| Flag style | `+flag` / `-flag` | `--flag` (both styles accepted) |
| Invocation | `freq file.cha` | `chatter clan freq file.cha` |
| Directory | Manual file listing or `*.cha` | Recursive directory traversal |
| Output format | Text only | Text, JSON, CSV |
| Error output | Simple messages to stderr | Rich diagnostics with file location and context |

Both flag styles are accepted — `+t*CHI` and `--speaker CHI` are equivalent. The automatic rewriting happens in `clan_args::rewrite_clan_args()` before argument parsing.

## Output formatting

- **File headers**: Our headers include the full file path; CLAN uses basename only in some commands
- **Speaker ordering**: CLAN outputs speakers in reverse encounter order (C linked-list prepend pattern). We replicate this behavior for parity.
- **Encoding**: UTF-8 only. CLAN supports legacy encodings via CP2UTF.
- **Line wrapping**: Our text output does not wrap long lines. CLAN wraps at ~80 characters in some commands.

## Counting semantics

These counting rules were discovered during golden test parity verification and apply across multiple commands:

| Rule | Commands affected | Details |
|------|-------------------|---------|
| Population SD | MLU, MLT | Uses `/ n`, not `/ n-1` (sample SD) |
| Brown's morphemes | MLU, WDLEN | Only 7 suffix strings count: `PL`, `PAST`, `Past`, `POSS`, `PASTP`, `Pastp`, `PRESP` |
| Fusional features | VOCD, MLU | `&PRES`, `&INF` etc. are part of the lemma; strip with `split('&')` |
| Apostrophe stripping | WDLEN | Characters counted after removing apostrophes |
| Turn = utterance | DIST | Every utterance is its own turn (no speaker-continuity grouping) |

## AST-based processing

The most fundamental divergence: CLAN processes CHAT files as text using string pattern matching. The Rust reimplementation parses files into a typed AST and operates on structured data. This means:

- Word classification uses typed fields (`word.category`, `word.untranscribed()`) instead of string prefixes (`starts_with('&')`)
- Annotations are structured objects, not regex-extracted substrings
- Transform commands modify AST nodes, then serialize — guaranteeing structural validity

This eliminates classes of bugs where string patterns match unintended content, but occasionally produces different whitespace or formatting in edge cases.
