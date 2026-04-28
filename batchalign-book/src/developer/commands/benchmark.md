# benchmark — Developer Reference

**Status:** Current
**Last updated:** 2026-04-08 07:40 EDT

Implementation guide for the `benchmark` command. For user-facing
documentation, see [User Guide: benchmark](../../user-guide/commands/benchmark.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign-cli/src/args/commands.rs` — `BenchmarkArgs` | asr-engine, lang, num-speakers, wor/nowor |
| Command definition | `crates/batchalign-app/src/commands/benchmark.rs` | `CommandDefinition` impl |
| Benchmark pipeline | `crates/batchalign-app/src/runner/dispatch/benchmark_pipeline.rs` | Orchestrates transcribe → compare → materialize |
| Compare (internal) | `crates/batchalign-app/src/compare.rs` — `MainAnnotatedCompareMaterializer` | Injects `%xsrep`/`%xsmor` on hypothesis, not gold |

---

## Composite architecture

`benchmark` is the canonical `Composite` command. It calls two sub-workflows
in sequence using their shared internal dispatch helpers:

1. `transcribe_pipeline.rs` — produces the hypothesis `ChatFile`
2. `compare.rs` — produces `ComparisonBundle` from hypothesis + gold

The released materializer for `benchmark` is `MainAnnotatedCompareMaterializer`
(injects comparison annotations on the main/hypothesis side), which is the
**opposite** of the released `compare` command's
`GoldProjectedCompareMaterializer`. They share the same `ComparisonBundle`
type but use different output views.

---

## Gold file discovery

Gold files (`FILE.cha`) are expected alongside the audio (`FILE.mp3`) with the
same stem. If the gold file is missing, the audio file is reported as failed
with a typed `GoldFileMissing` error.

---

## Testing

```bash
make test
# Full ML golden test (ASR + compare — only on net)
cargo nextest run --profile ml -E 'test(benchmark::golden)'
```

---

## Related developer documentation

- [Command Flowcharts: benchmark](../architecture/command-flowcharts.md#benchmark)
- [compare developer reference](compare.md)
- [transcribe developer reference](transcribe.md)
- [Adding Commands](adding-commands.md) — use `benchmark` as the reference for `Composite`
