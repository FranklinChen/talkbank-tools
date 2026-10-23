# Workflow Contributor Guide

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

This is the shortest path for adding a new command, workflow family, or engine
without fighting the refactor stream.

If you read code before prose, start at
`crates/batchalign/src/recipe_runner/catalog.rs`. That table is the
contributor-facing map: every released command is one entry in it. From there,
jump to:

- `recipe_runner/recipes.rs` for the ordered stages each entry points at
- `recipe_runner/command_spec.rs` for what a `CommandFamily` implies
- `crates/batchalign/src/recipe_runner/command_spec.rs` for the workflow-family metadata
- `crates/batchalign/src/text_batch.rs` for reusable text-family helpers
- the family-specific runner/dispatch modules only when the command shape is
  genuinely new

(`crates/batchalign/src/commands/` no longer exists. It held a per-command
authoring layer that produced values nothing read, deleted 2026-07-28, and a
catalog of one-line delegations, deleted 2026-07-29.)

For batch-oriented text commands, the important typed seams are:

- `TextBatchFileInput` for one named file plus its owned CHAT payload
- `TextBatchFileResults` for one batch's named file outcomes
- `TextWorkflowFileError` for a file-scoped failure that keeps the message
  separate from file identity

## Choose A Family

The catalog already assigns released commands to one of these families, so the
first question is usually "which family is my command reusing?"

The families are the variants of `CommandFamily`
(`recipe_runner/command_spec.rs`), and the choice is load-bearing: the family
implies seven runtime policies through `const fn`s on the enum, so read those
before picking.

- `AudioSequential` for one media file at a time (GPU lane, bounded file-level parallelism).
- `BatchedText` when work is pooled across files into shared infer batches (CPU lane, one dispatch per job).
- `ReferenceProjection` when two artifacts are jointly primary (paired inputs).
- `MediaAnalysis` when the output is not CHAT (IO lane).
- `Composite` when you are composing existing command flows (all policy delegated to children).
- Use `text_batch.rs` and typed materializers when the hard part is output shape rather than dispatch shape.

An earlier `WorkflowFamily` enum offered a coarser 4-way version of this choice
and was deleted on 2026-07-29; it merged the audio and media-analysis families
into one, and nothing read it.

## Current Examples

Every one is a `CatalogEntry` in `crates/batchalign/src/recipe_runner/catalog.rs`
plus a `Recipe` in `recipe_runner/recipes.rs`:

- `transcribe`: `AudioSequential`, `TRANSCRIBE_RECIPE`
- `align`: `AudioSequential`, `ALIGN_RECIPE`
- `morphotag`: `BatchedText`, `MORPHOTAG_RECIPE`
- `compare`: `ReferenceProjection`, `COMPARE_RECIPE`
- `benchmark`: `Composite`, `BENCHMARK_RECIPE`

The first three are the simplest command-owned wrappers over shared runner
families. `compare` is the reference-projection example, and `benchmark` is the
composite example that chains shared kernels while still keeping output
materialization in Rust rather than CLI glue.

Today `compare` is also the clearest example of "output shape is the hard
part":

- `build_comparison_artifacts()` morphotags only the main transcript and parses the gold
  companion raw
- `ComparisonBundle` is the compare IR: main/gold utterance views, structural
  gold-to-main word matches, and metrics
- the released materializer writes projected-reference `%xsrep` / `%xsmor`
  through typed tier-content models, then lowers once to
  `UserDefinedDependentTier`
- the internal benchmark/main materializer is separate from the compare command
- projection must stay AST-first rather than rebuilding tiers from `%xsrep` /
  `%xsmor` strings
- `.compare.csv` comes from a typed row/table model, not handwritten CSV text

`transcribe_s` is the same per-file family as `transcribe`, but surfaced as the
diarized variant in the catalog.

## Add A New Command

1. Add the stage recipe to `crates/batchalign/src/recipe_runner/recipes.rs`.
2. Declare one `CatalogEntry` in `crates/batchalign/src/recipe_runner/catalog.rs`. That is the whole registration; there is no second place.
3. Reuse an existing runner family when possible; only widen `runner/dispatch/` when the command shape is genuinely new.
4. Keep the command-specific orchestration in Rust helper modules, not in `pyo3`.
5. Keep runner/dispatch code focused on job lifecycle, resource policy, and shared execution mechanics.

The step-by-step version, including which tests fail until each step is done, is
[Adding a New Command](./adding-commands.md).

If the command batches text across files, prefer the
`TextBatchFileInput`/`TextBatchFileResults` seam over raw tuples at the
text-family boundary, and keep any file-local error detail in
`TextWorkflowFileError` rather than stringly return values.

If the command emits structured output, add a typed pre-serialization model in
the owning crate before you add serializer code. New semantic strings should be
newtyped, CHAT tier payloads should flow through `WriteChat`, and CSV should be
rendered from structured row/table types via `csv`.

## Add A New Engine

1. Keep provider selection at the control-plane boundary.
2. Keep engine-specific transport or worker protocol code in the provider or
   worker layer.
3. Add new typed payloads in a shared crate before widening the command-owned Rust API.

## Practical Rule

If a change makes `commands/*` more obvious and keeps `runner/dispatch/*`
reusable, it is probably a real improvement. If it pushes orchestration back
into `pyo3`, `cli`, or scattered dispatch tables, it is probably the wrong
direction.
