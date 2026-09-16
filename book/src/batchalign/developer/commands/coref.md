# coref: Developer Reference

**Status:** Current
**Last updated:** 2026-09-15 20:20 EDT

Implementation guide for the `coref` command. For user-facing documentation,
see [User Guide: coref](../../user-guide/commands/coref.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `CorefArgs` | merge-abbrev only; coref has no `--lang` |
| Language shape | `crates/batchalign/src/dispatch_language.rs` | Declares coref per-file, and mints the proof a job-level dispatcher needs |
| Coref dispatch | `crates/batchalign/src/execution/coref.rs`: `dispatch_coref_job` | Plans, tracks, batches across files, writes; takes no language |
| Catalog entry | `crates/batchalign/src/recipe_runner/catalog.rs` | the `CatalogEntry` for `coref` |
| Stage recipe | `crates/batchalign/src/recipe_runner/recipes.rs` | `COREF_RECIPE` |
| Coref orchestration | `crates/batchalign/src/coref.rs` | Full-document context assembly, worker dispatch, sparse injection |
| Injection | `crates/batchalign/src/coref.rs` | Writes sparse `%xcoref:` tiers |
| Worker IPC | `batchalign/inference/coref.py`: `batch_infer_coref()` | Loads Stanza coref model, returns chain structures |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## No caching

`coref` intentionally bypasses the utterance cache. Coreference chains span
the entire document, the same utterance has different coreference in different
document contexts, making per-utterance BLAKE3 keys meaningless. Every `coref`
invocation always calls the worker.

This is a deliberate architectural decision, not an oversight. Annotated in
the `coref.rs` orchestration module.

---

## Sparse output

`%xcoref:` tiers are only written on utterances that contain at least one
mention participating in a coreference chain. Most utterances in a file are
untouched. This makes `coref` output stable under incremental edits, adding
or removing utterances that don't participate in chains doesn't disturb the
existing annotations.

---

## Worker IPC: coref task

```text
request item (one per document):
{ "sentences": [["hello", "world"], ["she", "said"]] }

CorefResultV2 items, one per document:
{ "kind": "resolved",
  "annotations": [ { "sentence_idx": 0, "words": [[{"chain_id": 0, "is_start": true, "is_end": true}], []] } ],
  "engine": "stanza-<version>/<coref package>" }
{ "kind": "no_sentences" }
{ "kind": "failed", "error": "Coref failed: ..." }
```

Each annotation gives, per word of one sentence, the chain references that
start or end at it. A worker exception is a `failed` item, never an empty
`resolved` one. The PyO3 bridge parses each host item through the Rust wire
type; an item that does not parse becomes that item's `failed` outcome.

## Engine identity and provenance

The engine names the model that resolved the chains, not only the library:
`stanza-<version>/<coref package>`, for example
`stanza-1.10.1/ontonotes-singletons_roberta-large-lora`, built by
`batchalign/inference/coref.py::coref_engine` from the installed Stanza release
and the same package constant the pipeline is constructed with.

The engine comes from the results, not from the worker's capability report.
`coref.rs` admits each document into `ResolvedCoref`
(`Resolved { response, engine }` or `NoSentences`), and the batch path stamps
each eligible file with `[fc-ba3 coref | engine=... ; lang=eng | ...]`, naming
the engine that file's own resolved result named, through the builder
translate shares (`result_named_provenance` with `ResultNamedCommand::Coref`),
which joins distinct names with `+` in text order and cannot fail. The stamp's
language is the constant `eng`, never a job-level value (see the 2026-05-03
incident). A file the worker resolved nothing for gets no stamp and the run
says why; a file that was never eligible (dummy, or not English) had no coref
run, so no stamp question arises for it. Because nothing is read from the
report, a coref job is never refused for a worker that has not named its
coreference engine; that pre-dispatch refusal was removed.

There is one implementation of the coref lifecycle. The per-file entry point
(`process_coref` / `run_coref_impl`) was deleted with the workflow trait that
reached it: it duplicated parse, gate, English check, dispatch, injection and
stamping, and the two copies had already drifted apart on whether a batch file
records its provenance.

---

## Per-file language dispatch, and why coref takes no language at all

Coref is a **per-file** command in the sense `dispatch_language.rs` defines: it
has no `--lang` on the CLI, so submission validation requires every coref job
to arrive as `LanguageSpec::PerFile`, and the command owns its own inference
language, the constant `eng`.

`dispatch_coref_job` therefore takes no language parameter, and neither does
`WorkerGateway::coref_batch`. That is not a simplification, it is the fix for a
defect that made the command unrunnable. Coref used to dispatch through a
shared "simple batched text" path that began by demanding
`job.dispatch.lang.as_resolved()`. On a per-file job that is always `None`, so
**every coref job ever submitted was refused before any work was dispatched**,
with a message telling the operator to pass a `--lang` flag coref does not
have. The language it demanded was then discarded unread by the batch, which
hardcodes `eng`.

Two spellings of one rule are what allowed this: submission validation listed
the per-file commands in a `matches!`, while each dispatcher re-asked the same
question of the job's `LanguageSpec`, and the two answers disagreed.
`dispatch_language::language_source` is now the single owner, read by both.
A per-file command's dispatcher takes no language; a job-level command's
(utseg, compare) takes a `JobLanguage`, whose inner code is private, so the
only way to obtain one is a resolution that refuses to mint it for a per-file
command. The absent case those dispatchers used to check for at runtime no
longer exists for the compiler to be reminded of.

The shared path had exactly one caller, so it was deleted with the defect
rather than made conditional.

## English-only restriction

Stanza's coreference model is English-only. The per-file English gate lives
inside the batch (`coref.rs::file_has_english`), which reads each file's own
`@Languages:` header; files with no header fall back to `eng`. Non-English
files pass through with no `%xcoref` tiers written and no error reported. There
is no job-level language to check, and no stage that checks one.

---

## Testing

```bash
make test
cargo test -p batchalign coref::
# Requires Stanza coref model
cargo test -p batchalign --features ml-golden --test ml_golden coref::golden
```

---

## Related developer documentation

- [Command Flowcharts: coref](../../architecture/command-flowcharts.md#coref)
