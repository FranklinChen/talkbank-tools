# compare

**Status:** Current
**Last updated:** 2026-09-16 22:04 EDT

Compare CHAT transcripts against gold-standard references to compute word
error rate (WER) and produce annotated output. For each primary `.cha` input,
the command first looks for a `FILE.gold.cha` companion in the same directory.
If that is absent, it falls back to `template.gold.cha` in the same directory.

Outputs two files per input:
1. A projected reference `.cha`: the gold transcript with `%xsrep` /
   `%xsmor` annotation tiers showing substitutions, insertions, and deletions
2. A `.compare.csv` sidecar with aggregate and per-POS metrics

Text-only, no audio involved.

---

## Quick start

```bash
# Compare a corpus (each FILE.cha must have FILE.gold.cha alongside it)
batchalign3 compare corpus/ -o compared/

# Override language
batchalign3 compare corpus/ -o out/ --lang eng

# Use the remote server
batchalign3 --server http://your-server:8001 compare corpus/ -o out/
```

---

## Pipeline

```mermaid
flowchart TD
    start([compare invoked]) --> discover[Discover primary .cha files\nskip *.gold.cha companions]
    discover --> pair[Pair FILE.cha with FILE.gold.cha]
    pair --> found{Gold companion or template found?}
    found -->|No| fail[Report file error]
    found -->|Yes| morph[process_morphosyntax\nmain transcript only\n→ validated document, carried as a proof]
    pair --> parse_gold[parse_lenient raw gold\n→ gold AST]
    morph --> bundle[compare()\nconform + one whole-file alignment\nComparisonBundle: main view, gold view,\nstructural word matches, metrics]
    parse_gold --> bundle
    bundle --> released[materialize_released\nproject_gold_structurally]
    bundle --> internal_main[materialize_main_annotated\ninternal/benchmark\ninject %xsrep / %xsmor on main]
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
| `--num-speakers N` | `2` | Number of speakers |
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
- `%xsrep:`: word-level comparison: unchanged words, `+word` insertions in
  gold, `-word` deletions from hypothesis
- `%xsmor:`: same alignment with POS tags (`NOUN`, `+ADJ`, `-?`)

The output is the **projected reference transcript**, not the main hypothesis.
The gold transcript's structure is preserved; morphosyntactic information from
the main transcript is projected onto it structurally where safe.

### Where the part of speech comes from

compare morphotags the main transcript itself and reads the gold companion off
disk as it is, so the usual pairing is a tagged main side and an untagged gold
one. Which transcript a tag is read from is decided once per file, from whether
the gold companion carries `%mor` at all:

| Gold companion | Matches report | Insertions report | Deletions report |
| --- | --- | --- | --- |
| Carries `%mor` | the gold tag | the main tag | the gold tag |
| Carries no `%mor` | the main tag | the main tag | `?` |

With a tagged gold companion the gold tag is what a reviewer needs: when the two
transcripts disagree about a word they both contain, the gold-standard tag is
the point of running compare.

With an untagged one there is no gold tag to report. A match means both sides
hold the same word, and the main side was morphotagged by this very run, so its
tag describes that word and is the only tag in existence for it. A deletion is a
gold word the main transcript does not contain, so nothing tagged it anywhere
and it reports `?`.

**This differs from batchalign2 on purpose.** batchalign2 attributes the gold
form's tag to every match and falls back to the literal `?` per form, with no
notion of whether the gold side is tagged at all. Against the untagged gold
companion that is the normal case, that makes every matched word `?` and
collapses the whole per-POS breakdown in `.compare.csv` into a single `?` row.
BA3 reports the tag it actually has.

One consequence is worth knowing. Punctuation is excluded from the comparison by
its surface form or by a `PUNCT` tag, and the second test can only fire on a
document that has tags. On an untagged gold companion, a token that is
punctuation only by its tag is therefore aligned as an ordinary word. Nothing
can recover a tag from a document that has none; the tagged-versus-untagged
decision is what makes that a stated property of an untagged companion rather
than an accident.

## Output: `.compare.csv`

A companion `.compare.csv` is written alongside each output `.cha` file. It
contains:
- Aggregate metrics row: WER, accuracy, match/insertion/deletion counts, total
  words
- Per-POS breakdown rows
- Substitution-paired WER and scores by language: per-language error counts,
  utterance and word language agreement, and code-switch precision and recall.
  See [Benchmarks: scoring by language](../../reference/benchmarks.md#scoring-by-language).

---

## Gotchas

**`compare` outputs the gold transcript, not the hypothesis.** The released
command materializes the projected-reference view. The main-annotated view
(hypothesis CHAT with comparison annotations) is an internal path used by
`benchmark`.

**Gold files are never modified.** Only the primary `.cha` and the output
`.cha` and `.compare.csv` are written.

**The main transcript is never re-read from text.** compare morphotags the main
transcript and then compares the document that morphotag produced, which is the
document its own validation gate judged. The main side used to be serialized
and parsed again with the lenient parser before being compared, so the compared
document was the parser's recovery of those bytes and a parse failure on that
path was a log line rather than a failure. The gold companion is still parsed
leniently, because it is read off disk as it is and nothing vouched for it.

---

## Related documentation

- [Command I/O: compare](../../reference/command-io.md#8-compare), I/O patterns, gold file convention, output shapes
- [Command Flowcharts: compare](../../architecture/command-flowcharts.md#compare), full architecture flowchart
- [Benchmarks](../../reference/benchmarks.md), WER metrics and evaluation methodology
