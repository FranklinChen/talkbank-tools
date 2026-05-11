# FIXIT -- Normalize CHAT Formatting

**Status:** Current
## Purpose

Normalizes CHAT file formatting by re-serializing through the parser. Fixes inconsistent spacing, malformed tier prefixes, and other formatting issues.

## Usage

```bash
chatter clan fixit file.cha
chatter clan fixit file.cha -o normalized.cha
```

## Behavior

Since the parse-serialize pipeline produces canonically formatted output, FIXIT is effectively a roundtrip: parse the file, then serialize the resulting AST. Any formatting inconsistencies are corrected during serialization.

## Differences from CLAN

- Uses full AST roundtrip rather than heuristic text manipulation.
- Files that fail to parse produce an error rather than attempting partial text-level fixes.
- Output is the canonical CHAT serialization, which may reorder some whitespace or normalize header formatting.
