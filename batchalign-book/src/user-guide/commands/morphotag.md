# morphotag

**Status:** Current
**Last updated:** 2026-04-22 20:55 EDT

Add morphosyntactic analysis (`%mor` POS/lemma tiers and `%gra` dependency
tiers) to existing CHAT transcripts. Text-only — no audio involved.

---

## Quick start

```bash
# Tag one file in place
batchalign3 morphotag file.cha

# Tag a corpus directory
batchalign3 morphotag corpus/ -o tagged/

# Tag with language override (file has wrong or missing @Languages header)
batchalign3 morphotag file.cha --lang spa

# Retokenize main lines to match UD tokenization (expands contractions)
batchalign3 morphotag corpus/ -o out/ --retokenize

# Use fleet server
batchalign3 --server http://net:8001 morphotag corpus/ -o out/
```

---

## Pipeline

All files are batched together into a single GPU pass. Utterances that hit
the cache are injected immediately without touching the worker.

```mermaid
flowchart TD
    start([morphotag invoked]) --> parse[Parse all files → ASTs]
    parse --> clear[Clear existing %mor/%gra tiers]
    clear --> collect[collect_payloads\nPer-utterance word lists with language metadata]

    collect --> retok_check{--retokenize?}
    retok_check -->|Yes: --retokenize| stanza_retok[TokenizationMode::StanzaRetokenize\nStanza may split/merge words]
    retok_check -->|No: --keeptokens| preserve[TokenizationMode::Preserve\nKeep original tokenization]

    stanza_retok --> lang_check
    preserve --> lang_check

    lang_check{--skipmultilang?}
    lang_check -->|Yes| skip_non_primary[MultilingualPolicy::SkipNonPrimary\nSkip utterances in non-primary language]
    lang_check -->|No: --multilang| process_all[MultilingualPolicy::ProcessAll\nProcess all utterances regardless of language]

    skip_non_primary --> cache
    process_all --> cache

    cache[Cache lookup — BLAKE3 keys\nwords + lang + terminator + special forms + engine version]
    cache --> inject_hits[Inject cache hits immediately]
    inject_hits --> worker[execute_v2(task="morphosyntax") misses\nprepared_text batch → Stanza NLP pipeline]
    worker --> repartition[Repartition responses by file]
    repartition --> inject_results[inject_results → insert %mor/%gra tiers]

    inject_results --> before_check{--before path?}
    before_check -->|Yes| incremental[process_morphosyntax_incremental\nSkip NLP for unchanged utterances]
    before_check -->|No| full_inject[Process all utterances]

    incremental --> merge_check
    full_inject --> merge_check

    merge_check{--merge-abbrev?}
    merge_check -->|Yes| merge[merge_abbreviations]
    merge_check -->|No| validate

    merge --> validate[Alignment validation\n%mor word count must match main tier]
    validate --> done([Output .cha files])
```

---

## Options

### Path options

| Option | Meaning |
| --- | --- |
| `PATHS...` | Input `.cha` files or directories |
| `-o`, `--output DIR` | Output directory (omit to overwrite in place) |
| `--file-list FILE` | Read input paths from a text file |
| `--in-place` | Explicit in-place flag |

If you combine `--file-list` with in-place processing on a large corpus, do
not expect the `.cha` files on disk to rewrite one by one during the run.
`morphotag` batches and stages text-NLP work internally; the visible in-place
file updates may land only when the current invocation finishes. For long
repair runs where you want output to appear incrementally, split the file list
into smaller chunks and run those chunks sequentially.

### Morphotag options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lang CODE` | from `@Languages` | 3-letter ISO language code. Overrides the file's `@Languages` header when set |
| `--retokenize` / `--keeptokens` | `--keeptokens` | Retokenize main lines to UD tokenization (may split/merge words), or preserve existing tokenization |
| `--skipmultilang` / `--multilang` | `--multilang` | Skip utterances in non-primary languages, or process all |
| `--lexicon FILE` | — | Comma-separated manual lexicon override file (read on client, injected as typed options) |
| `--merge-abbrev` | off | Merge abbreviations in the output |
| `--no-l2-morphotag` | off | Opt out of L2 dispatch. With this flag, `@s` code-switched words emit `L2\|xxx` placeholders instead of real POS/lemma/deprel annotations (legacy behavior, kept for reproducibility of older analyses) |
| `--no-pos-hints` | off | Opt out of transcriber `$POS` hint respect. By default, after morphotag the pipeline overrides any `%mor` POS that disagrees with the CLAN→UD-mapped hint on main-tier words carrying `$POS` suffixes. Lemma and features from Stanza are preserved. Pass `--no-pos-hints` to skip the override pass and keep Stanza's POS as-is. See [Transcriber `$POS` Hints](../../reference/pos-hints.md) for the mechanism and coverage table |
| `--before PATH` | — | Previous version of the file for incremental processing (skip unchanged utterances) |

---

## What changes in the `.cha` file

- `%mor` tier added or replaced with POS tags and lemmas per word
- `%gra` tier added or replaced with dependency relations
- Main tier text may be retokenized when `--retokenize` is set
- Special `@Options: dummy` notation is auto-detected and preserved
- No audio is involved; this is a text-only transform

---

## Language routing

When `--lang` is omitted, the language is read from the CHAT file's
`@Languages` header (first declared language). Individual utterances with
a `[- lang]` precode are routed to the appropriate language-specific Stanza
model regardless of the file-level language.

See [Per-Utterance Language Routing](../../reference/per-utterance-language-routing.md).

## `--retokenize` warning

`--retokenize` allows Stanza to split or merge words on the main tier to match
UD tokenization (e.g. expanding "don't" → "do n't"). This may invalidate
existing `%wor` timing bullets. If the file has already been aligned, re-run
`align` after retokenizing.

## Reading the server log

If you see ``WARN Stripped N Stanza control-token leak(s) ...`` lines
in ``~/.batchalign3/server.log``, those are working-as-designed
signals from a known-upstream-defect workaround firing — not errors.
See the troubleshooting page section
[Stripped N upstream-library warnings](../troubleshooting.md#stripped-n-upstream-library-warnings-in-the-server-log)
for the full explanation and what to do.

## L2 dispatch for code-switched words (default: on)

`@s` (code-switched) words are routed to secondary-language Stanza
models and annotated with real POS tags, lemmas, and dependency
relations — including proper handling of contractions
(`it's@s:eng` → `pron|it~aux|be`) and phrasal verbs
(`wake@s up@s` → `verb|wake part|up` with `COMPOUND-PRT` GRA deprel).
This is the default behavior as of 2026-04-15.

To opt out and emit legacy `L2|xxx` placeholders (e.g. for
reproducibility of older analyses), pass `--no-l2-morphotag`:

```bash
batchalign3 morphotag bilingual.cha --no-l2-morphotag
```

**Validation.** L2 dispatch has been validated at scale —
**99.97% dispatch rate across 19 language pairs** and 17,352
`@s` words in the [2026-04-15 evaluation](../../reference/l2-eval-runs/2026-04-15/summary.md),
with 100% on 18 of 19 pairs. See the
[ungating decision](../../reference/l2-morphotag-ungating-decision.md)
for aggregate numbers and known limitations.

**Example.** German-English code-switching:

```
*EVA:   was ich jetzt machen möchte ist film@s studies@s .
%mor:   ... noun|film noun|study-Plur .     ← default (L2 dispatch on)
%mor:   ... L2|xxx L2|xxx .                 ← with --no-l2-morphotag
```

See also:
- [L2 Morphotag: Per-Word Code-Switching Analysis](../../reference/l2-morphotag.md)
  — full design, merge algorithm, phrasal-verb diagram
- [Ungating Decision](../../reference/l2-morphotag-ungating-decision.md)
  — aggregate numbers and known limitations

---

## Related documentation

- [Morphosyntax Pipeline](../../reference/morphosyntax.md) — %mor/%gra format, Stanza model details
- [Per-Utterance Language Routing](../../reference/per-utterance-language-routing.md) — `[- lang]` precodes
- [L2 & Language Switching](../../reference/l2-handling.md) — `@s` annotation, code-switching
- [Multi-Word Tokens](../../reference/mwt-handling.md) — MWT expansion and `--retokenize`
- [Command I/O: morphotag](../../reference/command-io.md#4-morphotag) — I/O patterns and mutation behavior
- [Command Flowcharts: morphotag](../../architecture/command-flowcharts.md#morphotag) — full architecture flowchart
- [Incremental Processing](../../architecture/incremental-processing.md) — `--before` flag mechanics
