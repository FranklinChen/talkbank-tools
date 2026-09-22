# benchmark: Developer Reference

**Status:** Current
**Last updated:** 2026-07-29 18:27 EDT

Implementation guide for the `benchmark` command. For user-facing
documentation, see [User Guide: benchmark](../../user-guide/commands/benchmark.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `BenchmarkArgs` | asr-engine, lang, num-speakers, wor/nowor |
| Catalog entry | `crates/batchalign/src/recipe_runner/catalog.rs` | the `CatalogEntry` for `benchmark` |
| Stage recipe | `crates/batchalign/src/recipe_runner/recipes.rs` | `BENCHMARK_RECIPE` |
| Benchmark pipeline | `crates/batchalign/src/runner/dispatch/benchmark_pipeline.rs` | Orchestrates transcribe → compare → materialize |
| Benchmark composition | `crates/batchalign/src/benchmark.rs`: `process_benchmark()` | Calls process_transcribe(), then process_compare_main_annotated() |

---

## Composite architecture

`benchmark` is the canonical `Composite` command. It calls two sub-workflows
in sequence using their shared internal dispatch helpers:

1. `transcribe_pipeline.rs`: produces the hypothesis `ChatFile`
2. `compare.rs`: produces `ComparisonBundle` from hypothesis + gold

The materializer for `benchmark` is `materialize_main_annotated()` function
(injects comparison annotations on the main/hypothesis side), which is the
**opposite** of the released `compare` command's `materialize_released()` function.
They share the same `ComparisonBundle` type but use different output views.

---

## Gold file discovery

Gold files (`FILE.cha`) are expected alongside the audio (`FILE.mp3`) with the
same stem. If the gold file is missing, the audio file is reported as failed
with a typed `GoldFileMissing` error.

The pairing is DERIVED, never submitted. `plan_benchmark_pairs`
(`recipe_runner/planner.rs`) takes each discovered input as the recording and
builds the gold path from it by replacing the extension.

Two things enforce that, because the planner used to accept anything it was
given:

- **Sources are classified at submission.** `JobSubmission::validate` refuses a
  CHAT source for a command whose planner is `PlannerKind::BenchmarkPairs`,
  naming the file. Without this, a submitted `.cha` became an audio work unit
  whose gold was itself, and `prepare_asr_media_input` handed the transcript to
  `ensure_wav`, which hands it to ffmpeg.

  The planner is the right key for exactly this case, and only this case: it is
  the planner that derives the gold companion by replacing the source's
  extension, so a `.cha` under it is its own gold. It is NOT a proxy for "takes
  audio": `align` and `speaker_identify` are `PlannerKind::AudioInputs` yet
  consume CHAT (`CommandIoProfile::ResolvedAudio`).

### Known gap: the same shape is still open for the media commands

`transcribe`, `opensmile`, `avqi` and `diarize` have the identical exposure. A
`.cha` submitted to any of them is accepted and reaches `ensure_wav`, which
hands it to ffmpeg; `runner/policy.rs` deliberately leaves `MediaAnalysisV2`
outside the dispatch-time CHAT gate, so nothing classifies the source in
between.

Generalizing the submission check to every command whose I/O profile is
`MediaInput` was implemented and then withdrawn, because it
is not sound against this repository's own tests: the server suites drive
`transcribe` with CHAT fixtures against the test-echo worker, in both content
mode (`FilePayload { filename: "test.cha" }`) and paths mode
(`source_paths: [..._test.cha]`), since test-echo echoes whatever it is given.
The generalized rule failed 45 tests in `cli_integration_suite`.

Closing it properly therefore needs a decision this change did not make: either
the test harness stops using CHAT as a stand-in for media, or the check
distinguishes a real submission from a test-echo one. Whether a command's
sources are transcripts or recordings is stated by its `CommandIoProfile` in
`recipe_runner/catalog.rs` (`MediaInput` means recordings), so the fact is
there wherever that decision is taken.
- **`BenchmarkWorkUnit` has private fields and a `pub(super)` constructor**, so
  `plan_benchmark_pairs` is the only place one is built. `benchmark_pipeline.rs`
  and `planning/` read it through `audio()` and `gold_chat()` accessors and
  cannot mint one, which is what stops a second route from pairing two
  unrelated files or putting a transcript in the audio slot.

---

## Testing

```bash
make test
# Full ML golden test (ASR + compare, only on Fleet/Large-tier hosts)
cargo test -p batchalign --features ml-golden --test ml_golden benchmark::golden
```

---

## Related developer documentation

- [Command Flowcharts: benchmark](../../architecture/command-flowcharts.md#benchmark)
- [compare developer reference](compare.md)
- [transcribe developer reference](transcribe.md)
- [Adding Commands](../adding-commands.md), use `benchmark` as the reference for `Composite`
