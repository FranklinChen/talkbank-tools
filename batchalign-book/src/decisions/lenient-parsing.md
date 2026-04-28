# Lenient Parsing

**Status:** Current decision and current parser behavior  
**Last verified:** 2026-03-05

## Decision

Batchalign keeps parsing and validation as distinct concerns:

- the parser should recover as much CHAT structure as it safely can
- validation should decide whether the recovered structure is acceptable for a
  given command or workflow

This keeps messy real-world CHAT files processable without pretending malformed
content is valid.

## Current behavior

The parser currently has three resilience layers:

1. tree-sitter grammar catch-alls for unknown headers and unsupported lines
2. Rust-side recovery/reporting for localized parse failures that escape the
   grammar
3. strict versus lenient entrypoints, depending on caller needs

## Current parse/validate split

### Strict parsing

Used where the command expects structurally clean CHAT input or is checking its
own output:

- structural extraction paths
- morphosyntax writeback paths
- round-trip or post-write validation

### Lenient parsing

Used where the command must accept messy corpus input and preserve as much
signal as possible:

- alignment-oriented paths
- translation and other workflows that may receive imperfect source CHAT
- pipeline entrypoints that need best-effort recovery

## Current resilience summary

### Works well now

- unknown `@Header:` lines recover as structured unknown headers
- junk non-CHAT lines are localized instead of causing broad parse collapse
- many malformed dependent tiers stay localized to the affected utterance/tier

### Known current limits

- missing `@UTF8` or `@Begin` are still structurally significant because they
  sit deep in the grammar shape
- malformed content can still taint the affected utterance or dependent tier
  even when the rest of the file parses
- lenient parsing is recovery-oriented, not a promise that every malformed file
  will become fully usable for every command

## Why this remains the right boundary

This decision supports the broader BA3 architecture:

- parse broadly enough to preserve recoverable structure
- validate explicitly at command boundaries
- avoid flattening CHAT to strings and trying to reconstruct intent later

For migration history or branch-by-branch parser evolution, use the migration
book. This page describes the current rule and the current parser boundary
only.
