# Workflow Contributor Guide

**Status:** Current
**Last updated:** 2026-03-26 14:05 EDT

This is the shortest path for adding a new command, workflow family, or engine
without fighting the refactor stream.

If you read code before prose, start at `crates/batchalign-app/src/commands/`.
That tree is now the contributor-facing map. From there, jump to:

- `crates/batchalign-app/src/command_family.rs` for the released-command family metadata
- `crates/batchalign-app/src/text_batch.rs` for reusable text-family helpers
- `crates/batchalign-app/src/commands/catalog.rs` for the released command catalog
- the family-specific runner/dispatch modules only when the command shape is
  genuinely new

For batch-oriented text commands, the important typed seams are:

- `TextBatchFileInput` for one named file plus its owned CHAT payload
- `TextBatchFileResults` for one batch's named file outcomes
- `TextWorkflowFileError` for a file-scoped failure that keeps the message
  separate from file identity

## Choose A Family

The command-owned catalog already assigns released commands to one of these
families, so the first question is usually "which family is my command reusing?"

- Use `WorkflowFamily::PerFileTransform` for a single-file transform.
- Use `WorkflowFamily::CrossFileBatchTransform` when work is pooled across files.
- Use `WorkflowFamily::ReferenceProjection` when two artifacts are jointly primary.
- Use `WorkflowFamily::Composite` when you are composing existing command flows.
- Use `text_batch.rs` and typed materializers when the hard part is output shape rather than dispatch shape.

## Current Examples

- `transcribe`: `crates/batchalign-app/src/commands/transcribe.rs`
- `align`: `crates/batchalign-app/src/commands/align.rs`
- `morphotag`: `crates/batchalign-app/src/commands/morphotag.rs`
- `compare`: `crates/batchalign-app/src/commands/compare.rs`
- `benchmark`: `crates/batchalign-app/src/commands/benchmark.rs`

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

1. Add or extend the command spec in `crates/batchalign-app/src/commands/<name>.rs`.
2. Register/export it via `crates/batchalign-app/src/commands/catalog.rs` and `commands/mod.rs`.
3. Reuse an existing runner family when possible; only widen `runner/dispatch/` when the command shape is genuinely new.
4. Keep the command-specific orchestration in the command module and Rust helpers, not in `pyo3`.
5. Keep runner/dispatch code focused on job lifecycle, resource policy, and shared execution mechanics.

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
