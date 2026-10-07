# `compare-runs`

**Last modified:** 2026-10-06 19:25 EDT

`compare-runs` is an offline comparator for two immutable, already-produced
artifact sets. It does not run Batchalign, contact a server, or treat either
side as gold. The existing [`compare`](compare.md) command remains the
primary-versus-`.gold.cha` workflow.

## Author manifests

```text
batchalign3 compare-runs manifest machine \
  --artifacts ours/ --output ours.manifest.json --run-id ours-2026 \
  --source-id session-17 --implementation batchalign3 \
  --command transcribe --build git-identity

batchalign3 compare-runs manifest human \
  --artifacts review/ --output review.manifest.json --run-id review-2026 \
  --source-id session-17 --protocol review-v1 --cohort reviewed
```

Manifests hash every regular file with BLAKE3. Roots must contain regular
files and may not contain symlinks; an existing identical manifest is a
no-op, while conflicting output is rejected.

## Plan and execution

Paths in the TOML plan are relative to the plan file. Artifact pair paths are
relative to their verified roots. A run-wide `speaker_map` may be overridden
per pair. Transcription permits a partial map and reports omitted speakers
visibly unmatched. Morphotag and alignment require a complete one-to-one map
covering every utterance speaker on both sides; a partial map is unpairable
with explicit unmatched-speaker lists, never a partial metric disguised as a
complete comparison. Pairs can be held out of aggregates with a required reason.

```toml
schema_version = 1
pairing = "same_source_chat"
output = "comparison-output"
exclusion_tokens = ["xxx", "yyy"]

[left]
manifest = "ours.manifest.json"
artifacts = "ours"

[right]
manifest = "review.manifest.json"
artifacts = "review"

[[pairs]]
left = "session.cha"
right = "session.cha"

[pairs.aggregate]
status = "included"
```

Run one typed mode:

```text
batchalign3 compare-runs transcribe --plan comparison.toml
batchalign3 compare-runs morphotag --plan comparison.toml
batchalign3 compare-runs align --plan comparison.toml
```

Transcription reports agreement WER/cWER, never accuracy, and count excluded
tokens separately. Morphotag reports tokenization, lemma, POS, feature-set,
clitic/chunk, dependency-head, and relation differences. Alignment first
requires identical normalized token identities, then reports each token's
timing state, absolute deltas, distributions, and independent order violations.

Every retained artifact must pass full Chatter admission, including dependent
tier alignment, before a mode computes metrics. Comparison does not regenerate
tiers and therefore has no corrupt-tier exemption. Invalid CHAT is unpairable;
a validator/producer failure is reported separately, not as CHAT invalidity.

## Morphology coverage and agreement

Each main-tier position carries `left_annotation` and `right_annotation`:
`no_main_token`, `absent`, `morphology_only`, or `complete`. All word chunks,
including post-clitics, contribute lemma, POS, feature and dependency differences.
Heads identify local item/chunk positions, not speaker-code spelling, so an
explicit speaker mapping does not itself create dependency differences. Chunk
order remains meaningful: a surface-first item and a governing-head-first item
can differ without either losing linguistic content. No side is presumed gold.

`compared_tokens` counts examined positions, whereas `fully_annotated_tokens`
counts positions with complete morphology and dependencies on both sides.
`analysis_agreement` is true/false only for those complete pairs; otherwise it
is null (an empty CSV cell). Two missing annotations are not an agreement.
`summary.csv` includes `left_annotation_state`, `right_annotation_state` and
`analysis_agreement` beside the individual difference axes. The rows and counts
are producer-sealed in the Rust API; counts and difference subsets derive from
the same rows rather than independently writable fields.

Report schema and comparison algorithm version 3 introduce these guarantees.
Earlier cached reports neither establish annotation completeness nor compare
all clitic analyses; they remain historical evidence and are never reused by
version 3.

Comparison algorithm version 4 additionally requires source-bound, complete
speaker correspondence for morphology and alignment. Earlier partial-map
results cannot be reused by that version. The report schema is unchanged.

## Alignment timing states

Every alignment token carries a timing STATE rather than a timing that may be
absent, so a token with no timing says why it has none:

| State | Meaning |
| --- | --- |
| `timed` | the `%wor` tier was corroborated and times this word; `start_ms` and `end_ms` sit beside the state |
| `unaligned` | the tier was corroborated and simply carries no bullet for this word |
| `no_wor_tier` | the utterance has no `%wor` tier, so no word in it is timed |
| `wor_tier_drifted` | the tier's slot count disagrees with the main tier's (`wor_slots`, `main_words`), which is what an edit made after alignment ran looks like |
| `wor_tier_uncorroborated` | the counts agree but `mismatches` display tokens do not match the words they would time, so the bullets describe a different reading of the utterance |

The three failure states are not interchangeable: a missing tier means
no timing tier is available, a drifted one means the transcript changed after it ran,
and an uncorroborated one means the tier belongs to different words than the
ones beside it. They were one empty value until 2026-09-16.

The timing-state types retain these distinctions for library callers. The
version-3 CLI refuses a Chatter-invalid drifted or contradictory retained tier
at admission rather than treating it as an algorithm input.

`summary.csv` carries `left_timing_state` and `right_timing_state` beside the
millisecond columns. A delta is reported only where both sides are `timed`.

Results are written under `OUTPUT/runs/COMPARISON_ID/`: complete
`report.json`, `summary.csv`, content-addressed `pairs/PAIR_ID.json`, and
evidence-only `review/PAIR_ID.json`. Pair caches are reused by default;
`--recompute` regenerates them. The algorithm version is part of the
comparison identity, so a change to what a comparison computes lands under a
new `COMPARISON_ID` and rows cached by an earlier version are never reused. Unpairable or unparsable pairs are recorded,
all pairs continue, and the command exits 2 after materialization. Differences
are evidence for human review, not automatic winner selection or golden-fixture
creation.
