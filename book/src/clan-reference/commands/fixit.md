# FIXIT -- Normalize CHAT Formatting

**Status:** Current
**Last updated:** 2026-05-22 13:05 EDT

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

## CLAN `+`-flag coverage audit

FIXIT is a **transform**. CLAN's `fixit.cpp::usage` shows `[c
...]` in its `Usage:` shape but documents no `+`-flag entries.
chatter's FIXIT exposes only the input path + `-o` output.

| CLAN flag | Meaning | Chatter | Status |
|---|---|---|---|
| (none documented) | default re-serialize | default | Done |

Audit summary: 1 Done, 0 Missing. Byte-parity complete for the
documented surface.
- Output is the canonical CHAT serialization, which may reorder some whitespace or normalize header formatting.
