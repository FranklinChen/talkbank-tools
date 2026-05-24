# LINES -- Add or Remove Sequential Line Numbers

**Status:** Current
**Last updated:** 2026-05-22 13:05 EDT

## Purpose

Adds or removes line numbers for display. The legacy manual describes `LINES` as inserting line numbers based on CLAN's "Show Line Numbers" display and using `+n` to remove them.

Since line numbering is a display concern rather than a structural CHAT operation, the AST transform is a no-op; the actual logic operates on serialized CHAT text via `add_line_numbers()`, `remove_line_numbers()`, and the end-to-end `run_lines()` function.

Line numbers are formatted as 5-character right-aligned integers prefixed to non-header lines. Header lines (`@Begin`, `@Languages`, etc.) are not numbered.

## Usage

```bash
chatter clan lines file.cha
chatter clan lines --remove file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--remove` | bool | `false` | Remove existing line numbers instead of adding them |

## CLAN `+`-flag coverage audit

LINES is a **transform**. Sources:
`OSX-CLAN/src/clan/lines.cpp::usage`,
`crates/talkbank-clan/src/transforms/lines.rs`.

### LINES-specific `+`-flags (from `lines.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+n` | Remove all line/tier numbers | `--remove` | Done | Direct mapping; rewriter routes `+n` → `--remove` (since 2026-05-22). |

Audit summary: 1 Done (add-numbers default), 1 Done (`+n`
removal mapped). 0 Missing.

## Behavior

**Adding line numbers:** Each non-header line receives a 5-character right-aligned sequential number prefix (e.g., `    1 *CHI: ...`). Header lines (`@`-prefixed) are not numbered.

**Removing line numbers:** Strips the first 6 characters (5-digit number + space) from non-header lines.

## Differences from CLAN

- **Manual intent**: `LINES` is a display/layout command, so text-level processing is intentional here.
- Operates on serialized text (post-parse) rather than raw input, since line numbering is a display concern.
- Uses the framework transform pipeline (parse -> transform -> serialize -> text fixups -> write).
