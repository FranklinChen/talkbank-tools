# TIERORDER -- Reorder Dependent Tiers to Canonical Order

**Status:** Current
**Last updated:** 2026-05-22 13:04 EDT

## Purpose

Reorders dependent tiers into a consistent order. The legacy manual describes `TIERORDER` as putting dependent tiers into a consistent alphabetical order, with `/lib/fixes/tierorder.cut` able to control the order.

`talkbank-clan` sorts dependent tiers on each utterance according to its built-in canonical ordering.

## Usage

```bash
chatter clan tierorder file.cha
```

## Options

This command has no command-specific flags beyond the shared
`-o, --output <PATH>` (default: stdout). See
[Output Formats](../user-guide/output-formats.md#transform-commands--o---output)
for the transform output flag.

## CLAN `+`-flag coverage audit

TIERORDER is a **transform**. Sources:
`OSX-CLAN/src/clan/tierorder.cpp::usage`,
`crates/talkbank-clan/src/transforms/tierorder.rs`.

### TIERORDER-specific `+`-flags (from `tierorder.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b` | Do not move bullets to a new speaker tier | — | Missing | Bullet-handling toggle. |
| `+cF` | Dictionary file (default `tierorder.cut`) controlling the canonical order | — | Missing | chatter hard-codes the canonical priority list in `tierorder.rs::tier_order()`. The `+cF` form would let researchers override the order. |
| `+q` | Sort only header tiers | — | Missing | Restrict scope to header reordering. |
| `+tS` | Switch speaker tier `S` with the main speaker tier | — | Missing | Speaker-tier promotion. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 (default canonical-order sort) |
| Missing | 4 |

TIERORDER's gap profile is dominated by **customization** flags
— `+cF` for user-supplied tier-order files is the most likely
request from researchers with non-standard tier conventions.
chatter's built-in `tier_order()` priority list is sufficient
for the canonical TalkBank/CHILDES corpus but doesn't let users
override.

## Behavior

Dependent tiers are sorted into the following canonical order
(per the `tier_order()` priority function at
`crates/talkbank-clan/src/transforms/tierorder.rs:57-100`):

1. **Linguistic analysis tiers** (priorities 0-5):
   `%mor` → `%gra` → `%pho` → `%mod` → `%wor` → `%sin`

2. **Phon project syllabification/alignment tiers** (priorities 6-8):
   `%xmodsyl` → `%xphosyl` → `%xphoaln`

3. **Behavioral/descriptive tiers** (priorities 10-18):
   `%act` → `%cod` → `%com` → `%spa` → `%gpx` → `%sit` → `%exp` → `%int` → `%add`

4. **Simple text tiers** (priorities 20-30):
   `%alt` → `%coh` → `%def` → `%eng` → `%err` → `%fac` → `%flo` → `%gls` → `%ort` → `%par` → `%tim`

5. **User-defined tiers** (priority 100):
   `%x*` (anything beyond the standard set)

6. **Unsupported tiers** (priority 101):
   tiers that hit the grammar's catch-all but aren't recognized CHAT tiers

Utterances with zero or one dependent tier are left unchanged.

## Differences from CLAN

- **Manual configurability not yet mirrored**: The legacy manual describes `tierorder.cut` as controlling tier order. The current implementation uses a built-in ordering instead.
- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
