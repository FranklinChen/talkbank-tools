# compare — Developer Reference

**Status:** Current
**Last updated:** 2026-04-14 17:35 EDT

Implementation guide for the `compare` command. For user-facing
documentation, see [User Guide: compare](../../user-guide/commands/compare.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign-cli/src/args/commands.rs` — `CompareArgs` | lang, num-speakers |
| Command definition | `crates/batchalign-app/src/commands/compare.rs` | `CommandDefinition` impl, gold-file discovery |
| Compare library | `crates/batchalign-app/src/compare.rs` | `compare()` — produces `ComparisonBundle` |
| Gold materializer | `crates/batchalign-app/src/compare.rs` — `GoldProjectedCompareMaterializer` | Projects %mor/%gra/%wor from main to gold, injects `%xsrep`/`%xsmor` |
| Main materializer | `crates/batchalign-app/src/compare.rs` — `MainAnnotatedCompareMaterializer` | Used internally by `benchmark` only |
| CSV writer | `crates/batchalign-app/src/compare.rs` — `write_compare_csv()` | Typed metrics model → CSV via Rust `csv` crate |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. Compare derives `FILE.gold.cha` first and falls back to
`template.gold.cha` at execution time inside the same directory.

---

## ComparisonBundle

The central typed model. Produced by `compare()`:

```rust
pub struct ComparisonBundle {
    pub main_view:    ChatFile,   // morphotagged hypothesis
    pub gold_view:    ChatFile,   // raw gold transcript
    pub word_matches: Vec<UtteranceAlignment>,  // structural per-utterance matches
    pub metrics:      CompareMetrics,           // aggregate WER + per-POS breakdown
}
```

Two materializers consume this bundle:
- `GoldProjectedCompareMaterializer` — released command output (gold side, with %xsrep/%xsmor)
- `MainAnnotatedCompareMaterializer` — internal path used by `benchmark`

**Never add logic that drives output from already-serialized CHAT text.** All
projection operates on the typed `ComparisonBundle` AST. This is the invariant
that prevents the BA2 string-surgery approach from re-entering.

---

## Gold projection semantics

`project_gold_structurally()` walks `word_matches` utterance by utterance:

| Match type | Action |
|------------|--------|
| Exact structural match | Copy `%mor` / `%gra` / `%wor` from main to gold |
| Full gold coverage, partial match | Project `%mor` only |
| Partial or unsafe | Keep gold dependent tiers unchanged |

Then `%xsrep` and `%xsmor` tiers are injected on the gold view using the
typed `CompareTierModel` — not string concatenation.

---

## CSV output model

```rust
pub struct CompareMetricsRow {
    pub label:       MetricLabel,    // "aggregate" or POS string
    pub wer:         f64,
    pub accuracy:    f64,
    pub matches:     u32,
    pub insertions:  u32,
    pub deletions:   u32,
    pub total_words: u32,
}
```

Written once at the serialization boundary via `csv::Writer`. No ad-hoc string
assembly.

---

## Testing

```bash
# Unit tests (no ML models)
make test
cargo nextest run -p batchalign-app -E 'test(compare::)'

# Golden tests (real Stanza for morphotag step — only on net)
cargo nextest run --profile ml -E 'test(compare::golden)'
```

---

## Related developer documentation

- [Command Flowcharts: compare](../architecture/command-flowcharts.md#compare)
- [BA2 Compare Migration](../../migration/ba2-compare-migration.md) — how compare was re-architected from BA2
- [Adding Commands](adding-commands.md) — use `compare` as the reference for `ReferenceProjection`
- [benchmark developer reference](benchmark.md) — composite command that calls compare internally
