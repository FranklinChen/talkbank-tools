# morphotag

**Status:** Current
**Last updated:** 2026-05-23 09:20 EDT

Add morphosyntactic analysis (`%mor` POS/lemma tiers and `%gra` dependency
tiers) to existing CHAT transcripts. Text-only — no audio involved.

---

## Language is per-file, not job-level

Morphotag has **no `--lang` flag**. Every input file's processing language
is read from that file's own `@Languages:` header at the start of the
per-file pipeline (`pipeline/morphosyntax.rs::resolve_per_file_lang`).
A single morphotag invocation can therefore process a heterogeneous
corpus — English files routed to Stanza English, Spanish files to Stanza
Spanish, Cantonese files to Stanza Chinese with the PyCantonese POS
overlay, etc. — all from one command. The job's wire-level language
spec is `LanguageSpec::PerFile`, surfaced on the dashboard and JSON API
as `"per-file"`. No English placeholder is ever stored.

If a file's `@Languages:` header is missing, malformed, or names a
language that Stanza does not support, morphotag does **not** silently
fall back to English. The file is reported in the job's status with a
typed error and returned unchanged.

---

## Quick start

```bash
# Tag one file in place — language is read from the file's @Languages header
batchalign3 morphotag file.cha

# Tag a corpus directory
batchalign3 morphotag corpus/ -o tagged/

# Retokenize main lines to match UD tokenization (expands contractions)
batchalign3 morphotag corpus/ -o out/ --retokenize

# Use remote server
batchalign3 --server http://your-server:8001 morphotag corpus/ -o out/
```

To "override" the language, edit the file's `@Languages:` line. There is
no CLI shortcut — and there cannot be, because a single command may span
many languages.

---

## Pipeline

All files are batched together through the batched-text-infer pool
(`crates/batchalign/src/runner/dispatch/infer_batched.rs` handles the
recipe-driven dispatch family; `ReleasedCommand::Morphotag` is the
discriminant used by the planner at
`crates/batchalign/src/runner/dispatch/plan.rs`).
Utterances are pooled across all files, grouped by language, and
dispatched to a Stanza worker per language group with semaphore-bounded
concurrency. Repeated `morphotag` runs on the same input run the full
Stanza pipeline again — text-NLP results are not cached
(`CacheTaskName` at `crates/batchalign/src/chat_ops/cache_key.rs:58`
covers only `ForcedAlignment` and `UtrAsr`).

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

    skip_non_primary --> worker
    process_all --> worker

    worker[execute_v2(task='morphosyntax')\nprepared_text batch → Stanza NLP pipeline\nper-language semaphore-bounded dispatch]
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

There is **no `--lang` flag**. Each file's processing language is read
from its own `@Languages:` header. Passing `--lang` to morphotag is a
clap parse error — the CLI surface deliberately rejects it. See the
"Language is per-file, not job-level" section above for the rationale.

| Option | Default | Meaning |
| --- | --- | --- |
| `--retokenize` / `--keeptokens` | `--keeptokens` | Retokenize main lines to UD tokenization (may split/merge words), or preserve existing tokenization |
| `--skipmultilang` / `--multilang` | `--multilang` | Skip utterances in non-primary languages, or process all |
| `--lexicon FILE` | — | Comma-separated manual lexicon override file (read on client, injected as typed options) |
| `--merge-abbrev` | off | Merge abbreviations in the output |
| `--no-l2-morphotag` | off | Opt out of L2 dispatch. With this flag, `@s` code-switched words emit `L2\|xxx` placeholders instead of real POS/lemma/deprel annotations (legacy behavior, kept for reproducibility of older analyses) |
| `--no-pos-hints` | off | Opt out of transcriber `$POS` hint respect. By default, after morphotag the pipeline overrides any `%mor` POS that disagrees with the CLAN→UD-mapped hint on main-tier words carrying `$POS` suffixes. Lemma and features from Stanza are preserved. Pass `--no-pos-hints` to skip the override pass and keep Stanza's POS as-is. See [Transcriber `$POS` Hints](../../reference/pos-hints.md) for the mechanism and coverage table |
| `--before PATH` | — | Previous version of the file for incremental processing (skip unchanged utterances) |

## `@Options: CA` files are passed through

Files whose header declares `@Options: CA` (Conversation Analysis mode)
are passed through morphotag unchanged. The pipeline parses the file,
detects the option, and serializes it back as-is — no `%mor` / `%gra`
tiers are added, and any pre-existing `%mor` / `%gra` tiers are
preserved verbatim. Provenance comments are not injected for these
files.

This mirrors how `align` skips files with `@Options: NoAlign`. The
mechanism is the option header alone; per-utterance content (CA
prosody markers, pauses, `&=` events, etc.) does not influence the
decision.

---

## What changes in the `.cha` file

- `%mor` tier added or replaced with POS tags and lemmas per word
- `%gra` tier added or replaced with dependency relations
- Main tier text may be retokenized when `--retokenize` is set
- Special `@Options: dummy` notation is auto-detected and preserved
- No audio is involved; this is a text-only transform

---

## Language routing

The language for each file is read from its `@Languages:` header (first
declared language). Individual utterances tagged with a `[- lang]` precode
are routed to the appropriate language-specific Stanza model regardless of
the file-level language. There is no CLI override — see the "Language is
per-file, not job-level" section at the top of this page.

For Cantonese, files declared with primary `@Languages: yue` route to
Stanza's Chinese (`zh`) pipeline with a PyCantonese POS overlay applied
after Stanza finishes (Stanza zh scores ~50% on Cantonese vocabulary;
PyCantonese ~94% — only `upos` is replaced; lemma and dependency parse
from Stanza are preserved). Mandarin files (`zho` / `cmn`) use Stanza zh
without the PyCantonese overlay. See
[Cantonese language details](../../reference/languages/cantonese.md) and
[Mandarin](../../reference/languages/mandarin.md).

See [Language Routing](../../../architecture/language-and-multilingual/language-routing.md#per-utterance-routing-into-stanza).

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
This is the default behavior. Mandarin-marked words (`@s:cmn` /
`@s:zho`) route through the Chinese `zh` morphosyntax path, and
Cantonese-marked words (`@s:yue`) use the same secondary-dispatch
surface as other supported Stanza languages. Unresolved or unsupported
targets still fall back to `L2|xxx`.

To opt out and emit legacy `L2|xxx` placeholders (e.g. for
reproducibility of older analyses), pass `--no-l2-morphotag`:

```bash
batchalign3 morphotag bilingual.cha --no-l2-morphotag
```

**Validation.** L2 dispatch has been validated at scale: across 19
language pairs and ~17K `@s` words, well above 99% dispatch to a
secondary-language Stanza model on most pairs, with 100% dispatch on
the majority of evaluated language pairs. The remaining cases fall
back to `L2|xxx`.

### Unsupported non-primary languages

`morphotag` only requires the **primary** `@Languages` code to be
Stanza-supported. Files whose primary is not Stanza-supported are
skipped with a typed diagnostic and never enter the pipeline.

When the primary IS supported, **non-primary content in any language
that Stanza does not support is processed cleanly with an `L2|xxx`
fallback**:

- `[- UNSUPPORTEDLANG]` whole-utterance precodes — the entire utterance
  is grouped under `UNSUPPORTEDLANG`, the worker partition routes that
  group to the fallback bucket (no Stanza dispatch), and every word in
  the utterance receives `L2|xxx` in `%mor` with no `%gra` relation
  emitted for those positions.
- `@s:UNSUPPORTEDLANG` per-word markers — the secondary L2 dispatch
  path for that span is short-circuited the same way; the host primary
  analysis is preserved and the `@s` token's slot stays as `L2|xxx`.

Both fallbacks are *graceful*: the worker never crashes on an
unsupported secondary, and other utterances in the same file (or other
spans in the same utterance) that target supported languages continue
to receive real morphology. The mechanism is a partition step in
`infer_batch` (`partition_groups_by_stanza_support`) that splits each
batch's language groups into "dispatchable" and "fallback" before
calling Stanza.

**Example.** German-English code-switching:

```text
*EVA:   was ich jetzt machen möchte ist film@s studies@s .
%mor:   ... noun|film noun|study-Plur .     ← default (L2 dispatch on)
%mor:   ... L2|xxx L2|xxx .                 ← with --no-l2-morphotag
```

See also:
- [L2 Morphotag: Per-Word Code-Switching Analysis](../../reference/l2-morphotag.md)
  — full design, merge algorithm, phrasal-verb diagram

## Validation and repair for `@s` input

- Whole-utterance same-language runs written as `word@s word@s ...` are
  rejected by pre-validation (E255). The canonical CHAT form is
  `[- lang]`, and `chatter debug fix-s` rewrites the qualifying
  whole-utterance pattern in place.
- Explicit `@s:LANG` words still dispatch to `LANG` even if `LANG` is
  missing from `@Languages`, but validation emits warn-only E254 so the
  header drift is visible. `chatter debug fix-s` appends those missing
  explicit languages to `@Languages`.
- `chatter debug fix-s` is a true no-op on already-correct files: it
  only rewrites a file when it can prove a `[- lang]` conversion or
  `@Languages` repair is needed.

### When `fix-s` will and will not rewrite

The rewrite predicate is conservative on purpose: an incorrect
`[- LANG]` insertion silently changes the language scope of an entire
utterance, including fillers and nonwords. The predicate only fires
when **every** word-bearing item in the utterance — words, fillers
(`&~`, `&-`, `&+`), nonwords, AND retraced material — carries an
explicit language attribution that resolves to the **same** target
language. A single unmarked token (e.g. a filler `&~dang3` with no
`@s:` marker) blocks the rewrite, even if every other word would
qualify.

When the rewrite fires, `fix-s` clears bare `@s` shortcuts from
fillers and nonwords as well as from regular words. This is critical:
a bare `@s` resolves relative to the surrounding tier language, so
adding a `[- LANG]` precode without clearing the shortcut would *flip*
the filler's resolved language to the precode target. (A previous
version of the tool skipped fillers and corrupted a corpus this way;
the fix-s predicate now walks all word-bearing items.)

---

## Failure modes

morphotag fails fast on engine failures rather than emitting partial
output. When the Stanza worker reports a per-utterance error (model
runtime error, Stanza output parse failure, protocol violation), the
affected file is marked failed with a typed `ItemErrors` message
naming the first few offending items and the total count. Other
files in the same batch continue normally — one bad file does not
poison the rest (BA2-parity multi-file semantics). The output `.cha`
for a failed file is **not** written; there is no silent path where
the job appears successful but the `%mor` tier is missing.

Note: items in languages Stanza does not support (code-switches into
`@s:<lang>` for unsupported languages) are an intentional fallback,
not a failure — those items keep their `L2|xxx` placeholders in
`%mor` and the file still succeeds.

---

## Related documentation

- [Morphosyntax Pipeline](../../reference/morphosyntax.md) — %mor/%gra format, Stanza model details
- [Language Routing](../../../architecture/language-and-multilingual/language-routing.md) — `[- lang]` precodes, auto-detection, per-word routing limits
- [L2 & Language Switching](../../reference/l2-handling.md) — `@s` annotation, code-switching
- [Multi-Word Tokens](../../reference/mwt-handling.md) — MWT expansion and `--retokenize`
- [Command I/O: morphotag](../../reference/command-io.md#4-morphotag) — I/O patterns and mutation behavior
- [Command Flowcharts: morphotag](../../architecture/command-flowcharts.md#morphotag) — full architecture flowchart
- [Incremental Processing](../../architecture/incremental-processing.md) — `--before` flag mechanics
