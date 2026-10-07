# Incremental Processing

**Status:** Current
**Last updated:** 2026-10-03 21:37 EDT

Incremental processing allows batchalign to reprocess only the utterances that
changed after a user edits a CHAT file, preserving cached dependent tiers
(`%mor`, `%gra`, `%wor`, bullets) for unchanged content. This is the key
enabler for the **transcribe → manual review → re-align** workflow.

## Motivation

The standard workflow is:

1. Run `transcribe` on audio → initial CHAT with ASR output
2. User reviews: fixes words, splits/merges utterances, corrects speakers
3. Run `morphotag` and/or `align` on the edited file

Without incremental processing, step 3 reprocesses every utterance even if
only 3 out of 50 changed. For morphosyntax, this means redundant Stanza
inference calls. For forced alignment, this means re-aligning audio for groups
whose words and existing timing are still structurally trustworthy.

The diff engine solves this by comparing the "before" version (pre-edit, with
existing dependent tiers) against the "after" version (post-edit) and computing
a precise per-utterance change classification.

## Architecture

```text
Before CHAT (with %mor/%gra/%wor/bullets)
   │
   ├── complete retained-source admission → admitted prior
   │
After CHAT (user-edited)
   │
   ├── source-bound command-plan admission → admitted working document
   │
   ▼
diff_chat(before, after) → Vec<UtteranceDelta>
   │
   ├── Unchanged    → copy dependent tiers from before
   ├── SpeakerChanged → copy dependent tiers (words identical)
   ├── TimingOnly   → copy %mor/%gra, re-align FA group
   ├── WordsChanged → reprocess NLP, re-align FA group if timing changed
   ├── Inserted     → process from scratch
   └── Deleted      → absent from output
```

### Layer 1: Diff Engine (`crates/batchalign-transform/src/diff/`)

The diff engine lives in `batchalign-transform` and has no server
dependencies. It operates purely on `ChatFile` ASTs.

**Algorithm:**

1. Extract Mor-domain words per utterance from both files using
   `extract_words()`.
2. Compute fingerprints: space-joined cleaned word text per utterance.
3. Run Hirschberg DP alignment on the fingerprint sequences (reusing the
   existing `dp_align::align()` infrastructure).
4. Post-process the alignment to detect substitution pairs (adjacent
   `ExtraPayload` + `ExtraReference` from the DP aligner).
5. Classify each result into an `UtteranceDelta`.

For matched pairs (same fingerprint), the classifier checks speaker codes and
bullet timing to distinguish `Unchanged`, `TimingOnly`, and `SpeakerChanged`.

**Files:**

| File | Purpose |
|------|---------|
| `diff/types.rs` | `UtteranceDelta` enum, `DiffSummary` |
| `diff/classify.rs` | `diff_chat()`: DP alignment + classification |
| `diff/preserve.rs` | `copy_dependent_tiers()`: tier transfer between files |

### Layer 2: Selective Orchestrators (`batchalign`)

Each incremental orchestrator receives an admitted prior and command-admitted
working input, runs the typed diff, and selectively reprocesses. A prior's
dependent tiers have no replacement exemption: they may be copied into output.

#### Morphosyntax (`process_morphosyntax_incremental`)

```text
1. Admit complete prior and command-planned working input
2. diff_chat(before, after) → deltas
3. For Unchanged/SpeakerChanged/TimingOnly:
     bind a complete prior %mor/%gra pair to compatible morphology inputs,
     then copy the pair; otherwise leave analysis owed
4. Collect all remaining analyzable utterances:
     changed/inserted content and unchanged content lacking complete analysis
5. Infer, inject, and admit complete checked output before serialization
```

Only utterances with outstanding analysis are sent to the Stanza worker.
Unchanged words are not sufficient to reuse an absent or partial analysis.
Nor does word equality establish equivalent analysis inputs: the canonical
`MorphologyInput` producer is shared with fresh payload collection. Reuse
requires matching resolved utterance language, typed terminator, ordered word
texts and roles (including special forms and resolved own/span code-switches),
and transcriber POS hints and isolation evidence used by grammatical rewrites.
Changing a language declaration, precode, form marker, question terminator or meaningful
pause can therefore require inference without changing the cleaned words.
The transfer permission borrows the exact destination and is consumed when
installing the pair; it cannot be redirected after compatibility admission.
Speaker-only and timing-only changes retain their compatible fast paths.
Text NLP is not cached; complete admitted prior tiers provide the incremental
fast path. A partial prior pair is not copied, since an isolated `%mor` would
cause the canonical collector to consider the utterance already analyzed.

#### Forced Alignment (`process_fa_incremental`)

FA operates on groups (time windows containing multiple utterances), but the
incremental path now preserves stable utterance-level timing before it decides
which groups need worker or cache work:

```text
1. Admit complete prior and adaptive word-timing working input before UTR
2. diff_chat(before, after) → deltas
3. For Unchanged / SpeakerChanged / TimingOnly utterances:
     copy %wor from before → after
     refresh main-tier word timing and utterance bullet from %wor
4. Group utterances in the refreshed "after" file
5. For each group:
     if every utterance in the group was refreshed successfully:
       → reuse current main-tier timing directly
     else:
       → check cache, then send misses to the FA worker
6. Inject remaining timings and serialize
```

This gives `align --before` three tiers of reuse:

1. full-file `%wor` refresh when the whole file is already reusable
2. per-utterance `%wor` preservation for unchanged regions in an edited file
3. cache lookup and worker FA only for the remaining changed groups

A single changed utterance still causes its containing FA group to be
re-aligned when that group cannot be reconstructed from preserved timing. But
stable groups no longer have to go back through audio alignment just because
the file contains edits elsewhere.

### Layer 3: Dispatch Integration (`runner/dispatch/`)

The dispatch layer reads optional `before_paths` from the job. Active alignment
admits a declared prior once before media preparation or UTR, retains it across
retries, and refuses missing or invalid declared input rather than silently
falling back. A declared no-op does not consume an unused optional prior.

```text
source parse → source-bound working disposition
active disposition + optional admitted prior → attempt
attempt + prior → process_fa_incremental
attempt without prior → run_fa_from_ast
processed structure → complete construction admission → output
```

For morphosyntax, the batched dispatch similarly checks `before_texts`:

```text
if !before_texts.is_empty() {
    // Per-file incremental path
    process_morphosyntax_incremental(before, after, services, &params).await
} else {
    // Batch path
    process_morphosyntax_batch(&files, services, &params).await
}
```

## `UtteranceDelta` Type

```rust,ignore
pub enum UtteranceDelta {
    Unchanged     { before_idx, after_idx },
    WordsChanged  { before_idx, after_idx, timing_changed: bool },
    TimingOnly    { before_idx, after_idx },
    SpeakerChanged { before_idx, after_idx },
    Inserted      { after_idx },
    Deleted       { before_idx },
}
```

Helper methods:

| Method | Returns `true` for |
|--------|--------------------|
| `needs_nlp_reprocessing()` | `WordsChanged`, `Inserted` |
| `affects_timing()` | `WordsChanged` (with timing), `TimingOnly`, `Inserted`, `Deleted` |
| `before_idx()` | All except `Inserted` |
| `after_idx()` | All except `Deleted` |

## Dependent Tier Preservation

`copy_dependent_tiers()` in `diff/preserve.rs` transfers specified tiers from
a "before" utterance to an "after" utterance using the existing
`replace_or_add_tier()` injection function. It's idempotent, safe to call
multiple times.

```rust,ignore
copy_dependent_tiers(
    &before_file, before_idx,
    &mut after_file, after_idx,
    &[TierKind::Mor, TierKind::Gra],
);
```

## "Before" File Sources

| Context | Before source | After source |
|---------|--------------|--------------|
| `--in-place` CLI | Current file on disk | Same file (pre-edit is the "before") |
| `--before` flag | Explicit path | Input file |
| REST API | `before_text` field | File content |
| First run | None → full processing | Input file |

When no "before" is available, the orchestrator falls back to full processing
automatically.

## Fallback Behavior

The incremental path falls back to full processing when:

- No "before" text is provided (first run)
- All utterances changed (`summary.unchanged == 0 && summary.speaker_changed == 0`)
- The diff engine cannot establish any correspondence

This ensures incremental processing is always safe, worst case, it does the
same work as batch processing, while the best common rerun case avoids both
cache lookup misses and worker FA for stable regions.

## Command Applicability

| Command | Incremental? | Granularity | Behavior |
|---------|-------------|-------------|----------|
| `morphotag` | Yes | Per-utterance | Skip unchanged; reprocess changed words |
| `align` | Yes | Per-group | Any changed utterance in a group → re-align group |
| `utseg` | Not yet | Per-utterance | Would skip unchanged utterances |
| `translate` | Not yet | Per-utterance | Would skip unchanged utterances |
| `transcribe` | N/A | Whole-file | Creates from scratch (audio → text) |
| `coref` | N/A | Whole-document | Context-dependent, must reprocess entirely |

## Cache Interaction

The per-utterance BLAKE3 cache complements the diff engine:

- **Cache hit on unchanged utterance:** Diff engine preserves tiers directly
  from "before", cache isn't even consulted for these.
- **Cache hit on changed utterance:** The new content might match a previous
  cache entry (e.g., fixing a typo back to the original). Cache is checked
  for all utterances that need reprocessing.
- **Cache miss on changed utterance:** Normal path, infer and cache the result.

The diff engine adds value beyond caching by preserving dependent tier
*alignment* (the cache stores NLP results, but tier injection requires the
full AST context) and by enabling FA group-level optimization (cache is
per-utterance, but FA operates per-group).

## Performance Impact

For a file with 50 utterances where 3 were edited:

| Path | Worker calls | Cache checks |
|------|-------------|-------------|
| Batch | 50 utterance payloads | 50 |
| Incremental | 3 utterance payloads | 3 |

For FA with 8 groups where 1 contains a changed utterance:

| Path | FA groups aligned |
|------|-------------------|
| Batch | 8 |
| Incremental | 1 |

## Files

| File | Crate | Purpose |
|------|-------|---------|
| `diff/types.rs` | talkbank-transform | `UtteranceDelta`, `DiffSummary` |
| `diff/classify.rs` | talkbank-transform | `diff_chat()` algorithm |
| `diff/preserve.rs` | talkbank-transform | `copy_dependent_tiers()` |
| `morphosyntax/` | batchalign | `process_morphosyntax_incremental()` |
| `fa/` | batchalign | `process_fa_incremental()` |
| `runner/dispatch/infer_batched.rs`, `fa_pipeline.rs` | batchalign | Dispatch routing for incremental paths |
