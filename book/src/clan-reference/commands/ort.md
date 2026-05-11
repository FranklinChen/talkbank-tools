# ORT -- Orthographic Conversion via Dictionary Lookup

**Status:** Current
## Purpose

Reimplements CLAN's CONVORT command, which applies orthographic conversion rules from a dictionary file to main-tier words. When a word is modified, the original main-tier text is preserved on a `%ort:` dependent tier for reference.

## Usage

```bash
chatter clan ort --dictionary ort.cut file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dictionary` | path | `ort.cut` | Path to the orthographic conversion dictionary |

## External Data

Requires an orthographic conversion dictionary (default: `ort.cut`). Format: `from_word  to_word` (one pair per line, tab or space separated). Lines starting with `#` or `;` are treated as comments. Lookups are case-insensitive.

## Behavior

For each utterance, the transform:

1. Serializes the original main tier content for preservation.
2. Applies dictionary-based word substitutions on the main tier.
3. If any words were modified, inserts a `%ort:` dependent tier containing the original (pre-conversion) main-tier text.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
