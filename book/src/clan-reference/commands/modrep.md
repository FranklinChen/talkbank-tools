# MODREP — Model/Replica Comparison

**Status:** Current
**Last updated:** 2026-05-26 10:20 EDT

## Purpose

Compares the model (target) pronunciation on the `%mod` tier with the actual (replica) pronunciation on the `%pho` tier, tracking word-by-word mappings between model forms and replica forms. This is used in phonological analysis to assess how closely a speaker's productions match the adult target forms.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409226) for the original MODREP command specification.

## Usage

```bash
chatter clan modrep file.cha
chatter clan modrep file.cha --speaker CHI
```

## Algorithm

1. For each utterance with both `%mod` and `%pho` tiers:
   - Extract word lists from both tiers (flattening groups)
   - Pair words positionally (model word N <-> replica word N)
   - Record each (model, replica) pair in a frequency map per speaker
2. Report per-speaker tables of model words with their replica variants and frequency counts

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <code>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <code>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--format <fmt>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

### MODREP-specific `+`-flags (from `modrep.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+a` | Sort output by descending frequency | — | Missing | CLAN's `modrep.cpp:1431` sets `isSort = TRUE`, gating a frequency-sorted output path at line 659. chatter's `finalize` step at `commands/modrep.rs:174-203` iterates `BTreeMap<String, ModelWordData>` in alphabetical-ascending key order; the result struct has no sort mode. A prior version of this row claimed Done-by-default; that was a mis-classification surfaced by the 2026-05-26 audit-vs-runtime sweep. Real Tier-4 feature work: add a sort-mode enum to `ModrepConfig`, the rewriter arm, and a frequency-sorted output path. |
| `+bS` | Set model tier name to `S` (e.g. `+b*CHI` or `+b%hes`); `S = "*"` to use `+t@ID=` | — | Missing | chatter hard-codes `%mod` as the model tier. |
| `+cS` | Set replica tier name to `S` | — | Missing | chatter hard-codes `%pho` as the replica tier. |
| `+d` | Spreadsheet output | — | Missing | Per `OSX-CLAN/src/clan/modrep.cpp:1492`, CLAN's `+d` is a no-arg toggle (`no_arg_option(f)`; `isExcel = TRUE`). chatter has no `--format csv` for MODREP. Per-MODREP arm in `clan_args.rs` passes the token through so clap reports the literal `+d` argument. |
| `+nS` / `+n@F` | Word `S` (or file `@F`) included in output associated with `+c` | — | Missing | Replica-side word allowlist. |
| `+oS` / `+o@F` | Word `S` (or file `@F`) included in output associated with `+b` | — | Missing | Model-side word allowlist. |
| `+o3` | Combine selected speakers per file | partial via `--per-file` inverse | Partial | |
| `+sS` / `-sS` | Word `S` included/excluded from input | `--include-word` / `--exclude-word` | Done | |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 5 |
| Partial | 1 |
| Rewriter only | 3 |
| Missing | 7 |

MODREP's biggest gap is the **tier-name customization** (`+bS`,
`+cS`): researchers using non-default model/replica tier names
cannot redirect chatter's MODREP. The `+nS` / `+oS` word
allowlists are also missing, which limits MODREP to whole-file
comparisons rather than targeted lexical subsets.

## Output

Per-speaker listing of model words, each with its set of replica variants and their frequency counts, sorted alphabetically by model word.

## Differences from CLAN

- Model and replica extraction uses parsed `%mod` and `%pho` tier structures from the AST rather than raw text line parsing
- Word pairing operates on typed `PhoWord` content rather than string splitting
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
