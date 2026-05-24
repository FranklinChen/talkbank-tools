# Migrating from CLAN

**Status:** Current
**Last updated:** 2026-05-23 18:52 EDT

This guide helps users of the original CLAN C binaries transition to `chatter clan`.

## Command mapping

Legacy CLAN commands map directly:

| Legacy | New | Notes |
|--------|-----|-------|
| `freq file.cha` | `chatter clan freq file.cha` | |
| `mlu +t*CHI file.cha` | `chatter clan mlu --speaker CHI file.cha` | |
| `combo +s"the" file.cha` | `chatter clan combo --include-word "the" file.cha` | |
| `flo file.cha` | `chatter clan flo file.cha` | |

## Flag translation

Legacy CLAN uses `+flag`/`-flag` syntax. Both styles work — the CLI automatically translates:

| CLAN flag | Modern equivalent | Meaning |
|-----------|-------------------|---------|
| `+t*CHI` | `--speaker CHI` | Include speaker |
| `-t*CHI` | `--exclude-speaker CHI` | Exclude speaker |
| `+t%mor` | `--tier mor` | Include dependent tier |
| `-t%gra` | `--exclude-tier gra` | Exclude dependent tier |
| `+t@ID="..."` | `--id-filter "..."` | Filter by @ID fields |
| `+s<word>` | `--include-word <word>` | Include word |
| `-s<word>` | `--exclude-word <word>` | Exclude word |
| `+g<label>` | `--gem <label>` | Include gem |
| `-g<label>` | `--exclude-gem <label>` | Exclude gem |
| `+z25-125` | `--range 25-125` | Utterance range |
| `+r6` | `--include-retracings` | Count retraced material |
| `+u` | *(default)* | Merge speakers (always on) |
| `+dN` | `--display-mode N` *(generic)* / per-command typed flag | Display mode — **partially landed.** Some `+dN` values have specific typed flags per command (FREQ `+d1`/`+d2`/`+d3`/`+d4`, COOCCUR `+d`, FREQPOS `+d`); the rest still fall through to the placeholder. See [`flag-translation.md`](flag-translation.md). |
| `+k` | `--case-sensitive` | Case-sensitive matching — **fully landed** across the search/frequency family (FREQ, KWAL, VOCD, COMBO, FREQPOS, DIST, MAXWD). See [`flag-translation.md`](flag-translation.md) and each command's audit page for the per-command implementation details. |
| ~~`+fEXT`~~ | ~~`--output-ext EXT`~~ | Output file extension — **currently non-functional**, see [`flag-translation.md`](flag-translation.md) |

You can continue using `+t*CHI` syntax for the functional flags above if you prefer — it works identically.

## New capabilities

Features not available in legacy CLAN:

- **JSON and CSV output** — `--format json` or `--format csv` for programmatic processing
- **Recursive directories** — pass a directory path to process all `.cha` files
- **Unified CLI** — all commands under one `chatter clan` namespace
- **Structured errors** — rich diagnostics with file location and context

## Known differences

See [Why We Diverge](../divergences/philosophy.md) and [Per-Command Divergences](../divergences/per-command.md) for a complete list. Key points:

- Output formatting may differ slightly (spacing, alignment)
- Some legacy CLAN C bugs are fixed rather than replicated
- Commands not ported are documented in [Commands Not Ported](../divergences/not-ported.md)
