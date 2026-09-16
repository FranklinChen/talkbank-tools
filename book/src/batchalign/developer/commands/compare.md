# compare: Developer Reference

**Status:** Current
**Last updated:** 2026-05-19 22:58 EDT

Implementation guide for the `compare` command. For user-facing
documentation, see [User Guide: compare](../../user-guide/commands/compare.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign/src/cli/args/commands.rs`: `CompareArgs` | lang, num-speakers |
| Command definition | `crates/batchalign/src/commands/compare.rs` | `CommandDefinition` impl, gold-file discovery |
| Compare engine | `crates/batchalign-transform/src/compare/engine.rs` | `compare()`: produces `ComparisonBundle` |
| Compare orchestration | `crates/batchalign/src/compare.rs` | Pairs each main transcript with its gold companion, runs the comparison, drives the materializers |
| Part-of-speech evidence | `crates/batchalign-transform/src/compare/pos.rs` | `GoldPos`: whether a document carries `%mor` tags at all, decided ONCE per file from the gold side. `GoldTag` and `MainTag` are separate types over the same primitive so the two sides cannot be passed the wrong way round |
| Comparison states | `crates/batchalign/src/compare.rs`: private `artifacts` module | `MorphotaggedMain` and `ComparisonArtifacts`, with private fields and one constructor each |
| Released materializer | `crates/batchalign/src/compare.rs`: `materialize_released()` | Projects %mor/%gra/%wor from main to gold, injects `%xsrep`/`%xsmor` |
| Benchmark materializer | `crates/batchalign/src/compare.rs`: `materialize_main_annotated()` | Annotates main transcript with `%xsrep`/`%xsmor` (internal to benchmark) |
| CSV writer | `crates/batchalign-transform/src/compare/metrics.rs`: `format_metrics_csv()` | Typed metrics model → CSV output |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. Compare derives `FILE.gold.cha` first and falls back to
`template.gold.cha` at execution time inside the same directory.

---

## ComparisonBundle

The central typed model. Produced by `compare()` in `talkbank_transform::compare`:

```rust,ignore
pub struct ComparisonBundle {
    pub main_utterances: Vec<UtteranceComparison>,  // per-utterance main-side comparisons
    pub gold_utterances: Vec<UtteranceComparison>,  // per-utterance gold-side comparisons
    pub gold_word_matches: Vec<GoldWordMatch>,      // structural word matches (gold → main)
    pub metrics: CompareMetrics,                     // aggregate WER + per-POS breakdown
}
```

Each `UtteranceComparison` contains:
- `utterance_index`: position in file
- `speaker`: speaker code
- `tokens`: comparison tokens (status: Match/ExtraMain/ExtraGold, with optional POS)

Each `GoldWordMatch` maps one word position in a gold utterance to one position in a main
utterance, establishing the structural alignment used by projection.

Two materializer functions consume this bundle:
- `materialize_released()` → gold-projected output (for released compare command)
- `materialize_main_annotated()` → main-annotated output (internal path, used by benchmark)

---

## The main side arrives as a proof, not as text

`crates/batchalign/src/compare.rs` owns a private `artifacts` module holding two
states and the only transitions between them:

- `MorphotaggedMain`, whose single constructor is `from_proof(PostValidated)`.
  It consumes morphotag's own post-validation proof and continues in the
  document that proof carries.
- `ComparisonArtifacts`, whose single constructor is
  `build(MorphotaggedMain, ChatFile)`. It RUNS the comparison rather than
  accepting a bundle, so the bundle a materializer reads is always the
  comparison of the two documents beside it.

Both types keep their fields private to that module, so there is no route into a
comparison that begins with a `String`, in production or in this crate's own
tests. That is the point of the shape rather than a side effect of it. Compare
used to serialize the morphotagged main transcript and parse it back with
`parse_lenient`, so the document it compared was the parser's recovery of our
own bytes rather than the document the gate had judged, and a parse failure
there was a `warn!` and a continue rather than an answer. Deleting that parse
without closing the route would leave nothing to stop it growing back.

`WorkerGateway::morphotag_for_compare` therefore returns `PostValidated`, and
`process_compare_morphotagged_main` takes one. The execution kernel carries the
proof between its `Morphosyntax` and `CompareAlign` stages and has no text to
offer in its place.

`PostValidated::into_judged_document()` is the transition that hands back the
model. It succeeds for a gated proof and for a declined one, because a CA main
transcript that morphotag declines to analyze still has to be compared, and it
refuses a pass-through, which carries the input's own bytes and no output model
at all.

The gold companion is still parsed leniently, and its parse errors are still
reported rather than refused, because nothing admits a gold companion at any
validity level: `gate_comparison_output` judges compare's output against what
that companion HAD, so refusing it for its own faults would refuse a document
compare never damaged.

---

## Which side's tags a comparison reports

`%mor` is a per-utterance tier, so "this word has no tag" and "this document
tags nothing at all" reach a consumer as the same `None`. `compare/pos.rs`
separates them once per file: `GoldPos::of` returns `Tagged` only when some
`%mor` item exists somewhere in the document and `Untagged` otherwise, and the
field behind `Tagged` is private, so the variant cannot be spelled for a
document that tags nothing.

compare morphotags the main transcript itself and reads the gold companion off
disk as it is, so the ordinary case is a tagged main beside an untagged gold.
A tagged gold reports its own tag for a matched pair, which is the point of
running compare at all. An untagged gold reports the MAIN tag, which is the
only tag that word has, and a deletion still reports nothing because no side
tagged it. Until 2026-09-16 every matched word took its part of speech from the
untagged gold form and reported the literal `?`, so `%xsmor` came out as a row
of question marks and the whole per-POS breakdown landed in one `?` bucket.

The decision reaches past the reported column. Punctuation is recognized by
surface OR by tag, and the tag half can only fire on a document that has tags,
so `GoldPos::excludes_from_comparison` falls back to the surface rule alone on
an untagged side, where a token that is punctuation only by its tag enters the
alignment as an ordinary word. Nothing can recover a tag a document does not
have; what the file-level decision buys is that the weaker rule is a stated
property of an untagged document rather than an accident of a per-form `None`.

`GoldTag` and `MainTag` wrap the same `Option<&str>` deliberately: passing them
the wrong way round is the one mistake `pos_for_match`'s call site could make,
and the compiler now refuses it instead of a reviewer catching it.

---

## Known defect: the projection census and the alignment filter disagree

Recorded 2026-09-16, not fixed. It belongs to neither change made that day: it
sits in a third seam, the compare engine's own word universe.

`materialize.rs::compared_word_counts` filters words by SURFACE only, through
`is_punct_or_filler`. The alignment's `engine.rs::flatten_side` filters through
`GoldPos::excludes_from_comparison`, which on a tagged document is the
disjunction "the surface is punctuation OR the word's `%mor` tag is `PUNCT`".
The two therefore disagree about any word that is punctuation by TAG but not by
surface, and `compared_word_counts` is exactly what `exact_projection_source`
compares its match count against.

Measured with a differential probe rather than argued: two documents identical
except for the tag on one word, `*CHI: hello comma .` carrying
`%mor: intj|hello PUNCT|comma .` on both sides. The comparison itself is
perfect, 1 match with 0 insertions and 0 deletions, both sides having excluded
the tagged word. `project_gold_structurally` nevertheless copied no `%mor`,
`%gra` or `%wor` at all. With that word tagged `noun` instead, the exact
projection fired and `%wor` was copied. The tag is the only difference between
the two runs.

The failure is a silent DEGRADATION rather than wrong output. The exact
projection path is refused, and the partial `%mor` reconstruction beneath it is
refused too, because both measure gold words with the census while the matches
they count came from the filter. An exactly matching utterance can therefore
project nothing.

The fix is one owner for "does this word take part in the comparison". The
alignment already computes that set in `FlattenedSide`, so the repair is for the
projection to read that set rather than recompute a different one, which also
removes the parallel-count shape that let the two drift apart.

---

## Gold projection semantics

The gold projection process (`project_gold_structurally()`) iterates each gold utterance
and determines whether to copy or reconstruct comparison tiers.

For each gold utterance, a six-condition check (`exact_projection_source()`) determines if
all gold words have **perfect structural alignment** to a single main utterance:

1. **Match completeness:** Every gold word position must have exactly one match
2. **Uniqueness:** Matches must map to distinct gold positions
3. **Mono-utterance:** All matches must originate from a single main utterance
4. **Word parity:** Compared word counts must match between gold and the source main utterance
5. **Alignable parity:** Alignable word counts (same universe) must match
6. **No errors:** All tokens must have `Match` status (no insertions/deletions)

**If exact match found (all 6 conditions pass):**
Copy `%mor`, `%gra`, `%wor` tiers directly from the source main utterance to the gold
utterance. This is the safest projection path.

**Otherwise (any condition fails):**
Reconstruct a projected `%mor` tier from the partial matches in `gold_word_matches`.
This handles cases where words are reordered, inserted, or deleted. The projected
`%mor` is built directly from the bundle's typed data, never from serialized text.

---

## CSV output model

```rust,ignore
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
cargo test -p batchalign compare::

# Golden tests (real Stanza for morphotag step, only on Fleet/Large-tier hosts)
cargo test -p batchalign --features ml-golden --test ml_golden compare::golden
```

---

## Related developer documentation

- [Command Flowcharts: compare](../../architecture/command-flowcharts.md#compare)
- [BA2 Compare Migration](../../migration/ba2-compare-migration.md), how compare was re-architected from BA2
- [Adding Commands](../adding-commands.md), use `compare` as the reference for `ReferenceProjection`
- [benchmark developer reference](benchmark.md), composite command that calls compare internally
