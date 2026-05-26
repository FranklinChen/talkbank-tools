# COMBTIER -- Combine Duplicate Dependent Tiers

**Status:** Current
**Last updated:** 2026-05-26 09:15 EDT

## Purpose

Combines duplicate dependent tiers within an utterance. The legacy manual describes `COMBTIER` narrowly: it corrects the case where transcribers create several `%com` lines, combining two `%com` lines into one by removing the second header and moving its material onto the first `%com` line.

`talkbank-clan` generalizes that behavior to any selected dependent-tier label, while preserving the underlying tier variant whenever the selected tier has a supported AST model.

This is useful for cleaning up files where duplicate tiers were introduced during manual editing or automated annotation.

## Usage

```bash
chatter clan combtier --tier com file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tier` | string | *(required)* | The tier label to combine (e.g., `com` for `%com:`, `spa` for `%spa:`) |
| `--separator` | string | `" "` | Separator between combined tier contents |

## CLAN `+`-flag coverage audit

COMBTIER is a **transform**. Sources:
`OSX-CLAN/src/clan/combtier.cpp::usage`,
`crates/talkbank-clan/src/transforms/combtier.rs`.

### COMBTIER-specific `+`-flags (from `combtier.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+tS` | Tier to combine (required) | `--tier <NAME>` (required; rewriter intercept) | Done | Same required-flag refusal shape — CLAN exits with "Please specify tier to combine with +t option." Rewriter routes both `+tcom` (bare prefix) and `+t%com` (percent prefix) to `--tier com` via a per-Combtier intercept in `clan_args.rs` (the generic `+t<bareword> → --speaker <bareword>` rewrite would otherwise produce `--speaker com`, which combtier rejects since its required field is `--tier`). Prior to 2026-05-26 the `Combtier` variant was missing from `ClanSubcommandKind`, so the intercept could not fire and `combtier +tcom` failed clap with `error: unexpected argument '--speaker' found`. |
| `--separator` (chatter extension) | Separator between combined contents | `--separator` | Chatter-only | No CLAN analog — CLAN's default is a space, hard-coded. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 |
| Chatter extension | 1 |
| Missing | 0 |

COMBTIER's surface is byte-parity complete. `--separator` is a
chatter convenience.

## Behavior

For each utterance, the transform finds all dependent tiers matching the specified label. If two or more matching tiers exist, their text content is concatenated using the configured separator and the result replaces the first occurrence. All subsequent duplicates are removed.

Utterances with zero or one matching tier are left unchanged.

## Differences from CLAN

- **Manual intent is narrower**: the legacy manual only documents `%com` combination explicitly. `talkbank-clan` supports other selected tier labels too.
- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
- **Tier-kind preservation**: Combining bullet/text dependent tiers such as `%cod:` and `%com:` preserves their actual tier variants instead of degrading them to user-defined tiers.
