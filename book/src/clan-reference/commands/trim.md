# TRIM — Remove Dependent Tiers

**Status:** Current
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

## Differences from CLAN

- **Legacy intent preserved**: `TRIM` follows the tier-removal behavior described in `CLAN.html`, rather than extracting utterance or gem ranges.
- Operates on the typed AST rather than the `KWAL`-style text-output workaround shown in the legacy manual.
- `--tier` / `--exclude-tier` operate on dependent-tier labels only. Headers and main tiers are always preserved.
- Supports `*` as a wildcard dependent-tier selector and normalizes `%trn` to `%mor` and `%grt` to `%gra`.
