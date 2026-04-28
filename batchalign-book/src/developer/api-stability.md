# API Stability

**Status:** Current
**Last updated:** 2026-04-02 07:26 EDT

This page defines the current compatibility stance for Batchalign's Python
surfaces.

## Locked de-Pythonization target

For current cleanup work, the repository finish line is explicit:

- keep subprocess workers;
- keep Python only at direct ML/library boundaries and thin adapters around
  them;
- keep config, orchestration, payload preparation, cache policy, result
  normalization, validation, and CHAT semantics in Rust;
- keep already-landed BA2 compatibility shims out of scope for this wave.

The detailed surface inventory lives in
[Python/Rust Interface](../architecture/python-rust-interface.md#locked-de-pythonization-target).

## Current Stance

**The project is in beta.** API stability is a goal but is not yet guaranteed.
Breaking changes may occur during the pre-release stabilization period. The
public API surface will be formally defined before the 1.0 release.

The current compatibility stance is intentionally architecture-first. That means:

- internal ergonomics matter more than compatibility
- cross-language boundaries may be redesigned aggressively
- old convenience helpers should not be preserved if they get in the way of a
  cleaner Rust-primary architecture
- breaking changes will be documented but may land without a deprecation period
  while the project remains in beta

The Python surfaces that exist today are still the recommended import points
inside this repository:

- `batchalign.pipeline_api`
- `batchalign.providers`

But they should be treated as current convenience layers, not frozen public
contracts. Neither surface is guaranteed stable until the 1.0 release.

## Stability Levels

### `batchalign.pipeline_api`

Status: repo-internal direct Python facade over Rust-owned CHAT operations.

Compatibility expectation:

- prefer the cleanest operation/provider shape, not compatibility shims
- remove Python orchestration layers when Rust can own them instead
- keep this layer as thin callback glue, not as a second document-runtime
  implementation
- update docs and tests in the same change

### `batchalign.providers`

Status: narrow stable public API.

This module currently re-exports worker wire types such as:

- `BatchInferRequest`
- `BatchInferResponse`
- `InferResponse`
- `InferTask`
- `WorkerJSONValue`

Compatibility expectation:

- keep this thin and explicit
- widen only when the worker wire contract truly needs it
- do not let it grow into a plugin-descriptor or discovery SDK

### `batchalign.compat`

Status: migration-only BA2 compatibility shim.

Compatibility expectation:

- keep it working as a transition aid while it exists
- do not use it to justify new non-ML Python logic
- do not treat it as the target architecture for de-Pythonization work

## Internal Surfaces

The following should be treated as internal implementation details:

- `batchalign.worker._types`
- `batchalign.worker._infer`
- `batchalign.worker._main`
- direct `batchalign_core.ParsedChat` callback ABI for extension authors

These modules may change whenever runtime, architecture, or performance work
needs them to change.

## Change Policy

When changing one of the current Python-facing surfaces:

1. prefer the simplest architecture over compatibility
2. update the docs in the same change
3. update HK/Cantonese engine wiring when provider behavior changes
4. add or update tests that pin the new intended contract

## Practical Rule

If you are writing repo-local CHAT-aware Python code:

- import from `batchalign.pipeline_api`

If you are writing provider-style code that needs the worker request/response
models:

- import the wire types from `batchalign.providers`

If you are reaching into `batchalign.worker._*`, you are probably depending on
the wrong layer.
