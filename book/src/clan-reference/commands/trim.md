# TRIM — Remove Dependent Tiers

**Status:** Current
**Last updated:** 2026-05-22 12:55 EDT

## Purpose

Removes selected dependent tiers from a CHAT file while preserving headers,
main tiers, and all other file structure. The
[CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) describes `TRIM`
as a shorthand for removing coding tiers, such as `%mor`, without changing
anything else in the transcript.

## Usage

```bash
chatter clan trim file.cha --exclude-tier mor
chatter clan trim file.cha --exclude-tier '*'
chatter clan trim file.cha --tier cod
```

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--tier <NAME>` | `+t%NAME` | Keep only selected dependent tier(s) |
| `--exclude-tier <NAME>` | `-t%NAME` | Remove selected dependent tier(s) |

## CLAN `+`-flag coverage audit

TRIM has **no dedicated `*.cpp` in the CLAN source tree** — CLAN
documents it as a `KWAL`-style invocation that produces text
output rather than as a discrete command. chatter exposes
`trim` as a first-class subcommand operating on the typed AST.

* Inherited flags applicable here: `+t%NAME` / `-t%NAME` for
  tier selection (both routed via `clan_args::rewrite_tier_speaker`
  to `--tier` / `--exclude-tier`).
* `*` wildcard for "all dependent tiers" is a chatter extension.

### Audit summary

| Bucket | Count |
|---|---|
| Done | 3 (default, `+t%X`, `-t%X`) |
| Chatter extension | 1 (`*` wildcard) |
| Missing | 0 |

TRIM is a chatter-first interpretation of CLAN's legacy
workaround. Parity is "as documented" rather than "as
implemented in CLAN" — by design.

## Differences from CLAN

- **Legacy intent preserved**: `TRIM` follows the tier-removal behavior described in `CLAN.html`, rather than extracting utterance or gem ranges.
- Operates on the typed AST rather than the `KWAL`-style text-output workaround shown in the legacy manual.
- `--tier` / `--exclude-tier` operate on dependent-tier labels only. Headers and main tiers are always preserved.
- Supports `*` as a wildcard dependent-tier selector and normalizes `%trn` to `%mor` and `%grt` to `%gra`.
