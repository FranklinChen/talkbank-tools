# POSTMORTEM -- Pattern-Matching Rules for %mor Post-Processing

**Status:** Current
**Last updated:** 2026-05-22 13:11 EDT

## Purpose

Reimplements CLAN's POSTMORTEM command, which applies pattern-matching and replacement rules to dependent tiers (typically `%mor:`). Rules are applied sequentially, and wildcard tokens (`*`) match any single token. The replacement side uses `$-` to copy the matched wildcard text.

The [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) does not
appear to contain a standalone `POSTMORTEM` command section. It is mentioned
indirectly as part of the `mor *.cha` pipeline that runs `MOR`, `PREPOST`,
`POST`, `POSTMORTEM`, and `MEGRASP` to produce `%mor` and `%gra`.

## Usage

```bash
chatter clan postmortem --rules postmortem.cut file.cha
chatter clan postmortem --rules rules.cut --target-tier spa file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-r`, `--rules` | path | *(required)* | Path to the rules file |
| `--target-tier` | string | `"mor"` | Target tier label to apply rules to |
| `-o`, `--output` | path | stdout | Output CHAT file path |

## CLAN `+`-flag coverage audit

POSTMORTEM is a **transform**. Sources:
`OSX-CLAN/src/clan/postmortem.cpp::usage`,
`crates/talkbank-clan/src/transforms/postmortem.rs`.

### POSTMORTEM-specific `+`-flags (from `postmortem.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Create files with ambiguous results (default: no ambiguous) | — | Missing | Ambiguity-output policy. |
| `+a1` | Interactive disambiguation (non-UNX only) | — | Missing | Interactive mode; out of scope for a CLI tool. |
| `+a2` | Mark all changes (with color on non-UNX) | — | Missing | Diff-style output. |
| `+cF` | Dictionary/rules file (default `postmortem.cut`) | `--rules <PATH>` | Done | Direct mapping. chatter requires explicit path. |
| `+MF` | Path of mor lib folder (UNX only) | — | Missing | Library lookup. |
| `+p1` | Look for `postmortem.cut` in working directory first | — | Missing | Path-search heuristic. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 1 |
| Missing | 5 |

POSTMORTEM's `+a` variants are output-policy switches (ambiguity
handling, interactive disambiguation, change-marking). chatter
implements only the default "non-ambiguous, no marking" behaviour.
Filed as Phase 1.7 follow-ups; not load-bearing for the typical
rule-application workflow.

## External Data

Requires a rules file. CLAN uses `postmortem.cut` from its `lib/`
directory by default; `chatter clan postmortem` does not bundle a
rules file, so `--rules` must be passed explicitly. Format:
`from_pattern => to_replacement` (one rule per line, using `=>` or
`==>` as the separator). Lines starting with `#` or `;` are
comments.

Wildcards: `*` in the pattern matches any single token. `$-` in the replacement copies the matched wildcard text.

## Behavior

For each utterance, the transform:

1. Finds the target dependent tier (default: `%mor:`).
2. If the target is a user-defined text tier, tokenizes its content.
3. Applies each rule sequentially, matching patterns and performing substitutions.
4. Stores the modified result back on the tier.

## Differences from CLAN

- **Manual coverage gap**: the
  [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) only mentions
  `POSTMORTEM` indirectly through the MOR pipeline, so this chapter cannot yet
  rely on a standalone legacy command spec.
- **Typed `%mor` safety**: If a rule would change a parsed `%mor` tier, `POSTMORTEM` fails explicitly until an AST-based `%mor` rewrite exists, rather than degrading typed morphology into user-defined text.
- User-defined target tiers are still supported as text rewrite targets.
- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
