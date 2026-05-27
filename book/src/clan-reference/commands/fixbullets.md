# FIXBULLETS -- Fix Timing Bullet Consistency

**Status:** Current
**Last updated:** 2026-05-27 10:12 EDT

## Purpose

Repairs timing bullets that link CHAT to audio or video. The legacy manual describes `FIXBULLETS` more broadly: converting old-format bullets to new format, inserting `@Media`, merging multiple bullets, adding language tags, and shifting global timing offsets.

`talkbank-clan` now supports the AST-safe subset of that behavior: main-tier monotonic bullet repair, global millisecond offsets on parsed bullet timings, and tier-scoped bullet repair using parsed tier kinds.

## Usage

```bash
chatter clan fixbullets file.cha
chatter clan fixbullets file.cha --offset 800
chatter clan fixbullets file.cha --tier cod
chatter clan fixbullets file.cha --exclude-tier com
```

## Options

- `--offset N`
  Shift parsed bullet timings by `N` milliseconds. Negative offsets fail if they would move a parsed bullet before `0`.
- `--tier S`
  Restrict processing to selected tier kinds such as `cod`, `%cod`, or `*` for main tiers.
- `--exclude-tier S`
  Exclude selected tier kinds from processing.

## CLAN `+`-flag coverage audit

FIXBULLETS is a **transform**. Sources:
`OSX-CLAN/src/clan/fixbullets.cpp::usage`,
`crates/talkbank-clan/src/transforms/fixbullets.rs`.

### FIXBULLETS-specific `+`-flags (from `fixbullets.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b` | Merge multiple bullets per line into one bullet per tier | (default) | Partial | chatter's monotonic-bullet repair already collapses overlapping ranges on the main tier. The semantics differ slightly — file a follow-up if a round-trip test surfaces a discrepancy. |
| `+g` | Zero out first bullet and offset the rest (per-file gem) | — | Missing | Gem-based offset reset. |
| `+m` | Merge all files into one file with offsets progressively offset | — | Missing | Multi-file merge with timing chaining. |
| `+oN` | Add N ms to all bullet timings | `--offset=N` | Done | Direct mapping. Rewriter routes `+oN` → `--offset=N` (`=` syntax, symmetric with the negative form below). The numeric-only guard prevents `+oS` non-numeric from accidentally matching. |
| `-oN` | Subtract N ms | `--offset=-N` (negative) | Done | Negative-value form. Rewriter routes `-oN` → `--offset=-N` (`=` syntax mandatory: clap parses a free-standing `-N` as a short-flag attempt and rejects it before reading it as the value of `--offset`). Subprocess regression guard: `legacy_fixbullets_negative_offset_runs_via_subprocess`. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 2 |
| Partial | 1 |
| Missing | 2 |

FIXBULLETS's `+g`/`+m` multi-file workflows are out of scope for
the AST-safe subset; the audit notes them as legacy
batch-processing workflows. The `+oN` ↔ `--offset N` mapping
should get a rewriter route (same shape as the recent
MAXWD `+cN` → `--limit N` fix); filed as a Phase 1.7 follow-up.

## Behavior

The transform iterates through the parsed AST and:

1. Shifts parsed bullet timings on supported AST locations: main-tier terminal bullets, main-tier inline word/internal bullets, bullet-content dependent tiers such as `%act/%cod/%com`, and `%wor` inline bullets.
2. Enforces non-overlapping, monotonic timing windows on main-tier terminal bullets.
3. Preserves duration when repairing an overlapping main-tier terminal bullet, using a minimum duration of 1 ms.

Utterances without main-tier terminal bullets are skipped for monotonic tracking.

## Differences from CLAN

- **Current supported scope**: `FIXBULLETS` supports global offsets and tier-scoped AST-native bullet repair on parsed bullet locations, including bullet-bearing `@Comment` headers.
- **Scope reduction remains:** the
  [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) documents a
  broader repair tool than the current implementation provides.
- **Not yet implemented:** old-to-new bullet conversion, `@Media` insertion, multi-bullet merge (`+b`), and `+l` language-tag insertion described in the manual.
- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
