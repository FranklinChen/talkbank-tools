# compare

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Compare CHAT transcripts against gold-standard references to compute word
error rate (WER) and produce annotated output. For each primary `.cha` input,
the command first looks for a `FILE.gold.cha` companion in the same directory.
If that is absent, it falls back to `template.gold.cha` in the same directory.

Outputs two files per input:
1. A projected reference `.cha` — the gold transcript with `%xsrep` /
   `%xsmor` annotation tiers showing substitutions, insertions, and deletions
2. A `.compare.csv` sidecar with aggregate and per-POS metrics

Text-only — no audio involved.

---

## Quick start

```bash
# Compare a corpus (each FILE.cha must have FILE.gold.cha alongside it)
batchalign3 compare corpus/ -o compared/

# Override language
batchalign3 compare corpus/ -o out/ --lang eng

# Use the fleet server
batchalign3 --server http://net:8001 compare corpus/ -o out/
```

---

## Pipeline

```mermaid
flowchart TD
    start([compare invoked]) --> discover[Discover primary .cha files\nskip *.gold.cha companions]
    discover --> pair[Pair FILE.cha with FILE.gold.cha]
    pair --> found{Gold companion or template found?}
    found -->|No| fail[Report file error]
    found -->|Yes| morph[process_morphosyntax\nmain transcript only]
    pair --> parse_gold[parse_lenient raw gold\n→ gold AST]
    morph --> parse_main[parse_lenient morphotagged main\n→ main AST]
    parse_main --> bundle[compare()\nconform + local window search + local DP\nComparisonBundle: main view, gold view,\nstructural word matches, metrics]
    parse_gold --> bundle
    bundle --> released[GoldProjectedCompareMaterializer\nproject_gold_structurally()]
    bundle --> internal_main[MainAnnotatedCompareMaterializer (internal/benchmark)\ninject %xsrep / %xsmor on main]
    released --> safe{Exact structural match?}
    safe -->|Yes| copy[Copy %mor / %gra / %wor]
    safe -->|No, full gold coverage| mor_only[Project %mor only]
    safe -->|No, partial or unsafe| keep[Keep gold dependent tiers unchanged]
    copy --> goldannot[Inject %xsrep / %xsmor on gold]
    mor_only --> goldannot
    keep --> goldannot
    goldannot --> merge_check
    internal_main --> internal_done([Internal main-annotated view])
    merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| metrics[Write .compare.csv]
    merge --> metrics
    metrics --> done([Output .cha + .compare.csv])
```

---

## Options

### Path options

| Option | Meaning |
| --- | --- |
| `PATHS...` | Input `.cha` files or directories (`.gold.cha` companions are auto-skipped) |
| `-o`, `--output DIR` | Output directory |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Explicit in-place flag |

### compare options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | `eng` | 3-letter ISO language code |
| `-n`, `--num-speakers N` | `2` | Number of speakers |
| `--merge-abbrev` | off | Merge abbreviations in the output |

Use `--override-media-cache` (global flag) when you need to force fresh
morphosyntax on the main transcript before scoring.

---

## Gold file convention

For each `FILE.cha` input, compare first looks for `FILE.gold.cha` in the
**same directory**. If that companion is absent, it falls back to
`template.gold.cha` in the same directory. Files ending in `.gold.cha` are
automatically treated as companions and skipped as primary inputs. If neither
gold file is found, the file is reported as failed.

---

## Output: `%xsrep` and `%xsmor` tiers

The projected reference `.cha` output uses:
- `%xsrep:` — word-level comparison: unchanged words, `+word` insertions in
  gold, `-word` deletions from hypothesis
- `%xsmor:` — same alignment with POS tags (`NOUN`, `+ADJ`, `-?`)

The output is the **projected reference transcript**, not the main hypothesis.
The gold transcript's structure is preserved; morphosyntactic information from
the main transcript is projected onto it structurally where safe.

## Output: `.compare.csv`

A companion `.compare.csv` is written alongside each output `.cha` file. It
contains:
- Aggregate metrics row: WER, accuracy, match/insertion/deletion counts, total
  words
- Per-POS breakdown rows

---

## Gotchas

**`compare` outputs the gold transcript, not the hypothesis.** The released
command materializes the projected-reference view. The main-annotated view
(hypothesis CHAT with comparison annotations) is an internal path used by
`benchmark`.

**Gold files are never modified.** Only the primary `.cha` and the output
`.cha` and `.compare.csv` are written.

---

## Related documentation

- [Command I/O: compare](../../reference/command-io.md#8-compare) — I/O patterns, gold file convention, output shapes
- [Command Flowcharts: compare](../../architecture/command-flowcharts.md#compare) — full architecture flowchart
- [Benchmarks](../../reference/benchmarks.md) — WER metrics and evaluation methodology
