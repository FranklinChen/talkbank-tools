# LONGTIER -- Remove Line Continuation Wrapping

**Status:** Current
**Last updated:** 2026-05-22 13:06 EDT

## Purpose

Removes physical continuation-line wrapping so each logical tier occupies one line in the output file. The legacy manual describes `LONGTIER` as removing line wraps on continuation lines so that each main tier and each dependent tier is on one long line.

In CHAT format, long tiers are conventionally wrapped with a newline followed by a tab. `LONGTIER` folds those continuations back into a single line.

## Usage

```bash
chatter clan longtier file.cha
chatter clan longtier file.cha -o unwrapped.cha
```

## Behavior

Folds any line starting with a tab character into the preceding tier line, replacing the newline+tab with a single space. The result has one line per tier with no continuation wrapping.

## Differences from CLAN

- **Manual intent**: `LONGTIER` is below the AST layer; it is about physical line wrapping, not CHAT semantics.
- Operates on raw text rather than partial parsing, making it robust against malformed files that might not parse cleanly.
- Normalizes all newlines to `\n` (handles `\r\n` and `\r`).

## CLAN `+`-flag coverage audit

LONGTIER is a **transform**. CLAN's `longtier.cpp::usage` accepts
`[o ...]` (the `+o` extra-output-tier flag, inherited) but
documents **no command-specific `+`-flags**. chatter's surface is
likewise minimal.

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| (none documented) | default unwrap continuations | default | Done |

Audit summary: 1 Done, 0 Missing. Byte-parity complete.
- Multiple leading tabs on continuation lines are all consumed.
