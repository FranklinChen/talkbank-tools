# ROLES — Rename Speakers

**Status:** Current
**Last updated:** 2026-05-22 13:08 EDT

## Purpose

Renames speaker codes throughout a CHAT file: in `@Participants`, `@ID` headers, and all main-tier speaker prefixes. Used to standardize speaker codes across a corpus.

## Usage

```bash
chatter clan roles --rename "EXP=INV" file.cha
chatter clan roles --rename "Child=CHI" --rename "Mother=MOT" file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `-r`, `--rename "OLD=NEW"` | Rename speaker OLD to NEW (required, can be repeated). Splits on the first `=`; see `crates/talkbank-cli/src/commands/clan/transforms.rs:172` for the parser. |
| `-o`, `--output` | Output CHAT file path (default: stdout). |

## Behavior

Speaker codes are renamed in all structural locations:
- `@Participants` header entries
- `@ID` header speaker fields
- Main-tier speaker prefixes (`*OLD:` becomes `*NEW:`)

## Differences from CLAN

- Operates on the typed AST rather than raw text.
- Speaker codes are renamed in all structural locations via AST manipulation.

## CLAN `+`-flag coverage audit

ROLES is a **transform**. Sources:
`OSX-CLAN/src/clan/roles.cpp::usage`,
`crates/talkbank-clan/src/transforms/roles.rs`.

### ROLES-specific `+`-flags (from `roles.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+cF` | Dictionary file (`original_code speaker_code speaker_role`) | `--rename "OLD=NEW"` (inline only) | Partial | chatter accepts inline `OLD=NEW` mappings as a repeatable flag; CLAN reads a dictionary file with three-column lines. Adding `--rename-file <PATH>` would close the file-list form. |

Audit summary: 1 Partial (file-list form missing), 0 Missing.
The semantic is also slightly narrower in chatter: CLAN's
dictionary maps `OLD → (NEW_CODE, NEW_ROLE)`; chatter only
remaps codes, not the participant role. Filed as a follow-up.
