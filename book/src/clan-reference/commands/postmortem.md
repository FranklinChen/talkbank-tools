# POSTMORTEM -- Pattern-Matching Rules for %mor Post-Processing

**Status:** Current
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
| `--rules` | path | `postmortem.cut` | Path to the rules file |
| `--target-tier` | string | `"mor"` | Target tier label to apply rules to |

## External Data

Requires a rules file (default: `postmortem.cut`). Format: `from_pattern => to_replacement` (one rule per line, using `=>` or `==>` as the separator). Lines starting with `#` or `;` are comments.

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
