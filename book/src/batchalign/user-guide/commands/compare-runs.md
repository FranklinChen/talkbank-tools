# `compare-runs`

**Last modified:** 2026-08-03 12:42 EDT

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
  --source-id session-17 --protocol iisrp-v1 --cohort reviewed
```

Manifests hash every regular file with BLAKE3. Roots must contain regular
files and may not contain symlinks; an existing identical manifest is a
no-op, while conflicting output is rejected.

## Plan and execution

Paths in the TOML plan are relative to the plan file. Artifact pair paths are
relative to their verified roots. A run-wide `speaker_map` may be overridden
per pair; a partial map is valid and leaves omitted speakers visibly
unmatched. Pairs can be held out of aggregates with a required reason.

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
alignment never ran, a drifted one means the transcript changed after it ran,
and an uncorroborated one means the tier belongs to different words than the
ones beside it. They were one empty value until 2026-09-16.

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
