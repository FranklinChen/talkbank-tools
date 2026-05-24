# ORT -- Orthographic Conversion via Dictionary Lookup

**Status:** Current
**Last updated:** 2026-05-22 13:07 EDT

## Purpose

Reimplements CLAN's CONVORT command, which applies orthographic conversion rules from a dictionary file to main-tier words. When a word is modified, the original main-tier text is preserved on a `%ort:` dependent tier for reference.

## Usage

```bash
chatter clan ort --dictionary ort.cut file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-d`, `--dictionary` | path | *(required)* | Path to the orthographic conversion dictionary |
| `-o`, `--output` | path | stdout | Output CHAT file path |

## CLAN `+`-flag coverage audit

ORT is a **transform**. Sources:
`OSX-CLAN/src/clan/ort.cpp::usage`,
`crates/talkbank-clan/src/transforms/ort.rs`.

### ORT-specific `+`-flags (from `ort.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+cF` | Homons table file name (default `0canhomo.cut`) | `-d` / `--dictionary <PATH>` | Done | Direct mapping; chatter requires the path explicitly (no default path search). Rewriter routes `+cF` → `--dictionary F` (since 2026-05-22). |

Audit summary: 1 Done, 0 Missing.

## External Data

Requires an orthographic conversion dictionary. CLAN uses `ort.cut`
from its `lib/` directory by default; `chatter clan ort` does not
bundle a dictionary, so `--dictionary` must be passed explicitly.
Format: `from_word  to_word` (one pair per line, tab or space
separated). Lines starting with `#` or `;` are treated as
comments. Lookups are case-insensitive.

## Behavior

For each utterance, the transform:

1. Serializes the original main tier content for preservation.
2. Applies dictionary-based word substitutions on the main tier.
3. If any words were modified, inserts a `%ort:` dependent tier containing the original (pre-conversion) main-tier text.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
