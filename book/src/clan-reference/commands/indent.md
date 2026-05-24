# INDENT -- Align CA Overlap Markers

**Status:** Current
**Last updated:** 2026-05-22 13:03 EDT

## Purpose

Aligns overlap markers in Conversation Analysis (CA) transcripts. The legacy manual describes `INDENT` simply as a program for realigning overlap marks in CA files, and notes that the files must use a fixed-width font such as CAFont.

`talkbank-clan` aligns closing overlap markers (`⌊`, U+230A) by column position with their matching opening overlap markers (`⌈`, U+2308) on a preceding speaker tier.

## Usage

```bash
chatter clan indent file.cha
chatter clan indent file.cha -o aligned.cha
```

## CLAN `+`-flag coverage audit

INDENT is a **transform**. Sources:
`OSX-CLAN/src/clan/indent.cpp::usage`,
`crates/talkbank-clan/src/transforms/indent.rs`.

CLAN's `indent.cpp::usage` exposes **no command-specific `+`-flags**
— the surface is just the inherited general flag set (none of
which apply to a pure CA-overlap aligner).

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default align) |
| Missing | 0 |

INDENT is byte-parity complete by virtue of having no
command-specific surface. The only legitimate concern is the
fixed-width-font dependency CLAN mentions — chatter doesn't care
about font metrics since it aligns by column count, not visual
width.

## Algorithm

1. Parse the file into tiers (speaker prefix + content text)
2. For each main tier (`*SPK:`), scan for opening overlap markers `⌈` and record their column positions and optional numeric suffixes
3. Scan up to 30 subsequent tiers from *different* speakers for closing overlap markers `⌊`
4. Match open/close pairs by numeric suffix (or sequentially if unnumbered)
5. Insert or remove spaces before the closing marker to align columns
6. Report unmatched markers as warnings

## Example

Before:
```text
*CHI:	I want ⌈ cookies ⌉ .
*MOT:	⌊ yeah ⌋ okay .
```

After:
```text
*CHI:	I want ⌈ cookies ⌉ .
*MOT:	       ⌊ yeah ⌋ okay .
```

Numbered overlaps (`⌈1`, `⌈2`, etc.) are matched by their numeric suffix, allowing multiple simultaneous overlaps to be aligned independently.

## Differences from CLAN

- **Manual intent**: `INDENT` is a layout command, not a semantic CHAT analysis command.
- Operates on UTF-8 text using Rust's `char`-based column counting rather than C byte-level scanning.
- Uses the text-based transform pattern (no AST round-trip) to preserve original formatting outside of overlap alignment.
- Maximum 10 alignment passes (CLAN's `goto beginAgain` loop has no bound, causing infinite loops on some inputs).
- Column counting treats each Unicode scalar value as width 1.
