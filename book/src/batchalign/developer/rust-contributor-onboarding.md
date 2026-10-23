# Rust Contributor Onboarding

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

This page is the shortest path to productive work on the Rust side of Batchalign3.

## Start Here

1. Read the user-facing [CLI reference](../user-guide/cli-reference.md).
2. Read the [Rust workspace map](rust-workspace-map.md).
3. Read the [Rust CLI and Server](rust-cli-and-server.md) for dispatch architecture and command-creation checklist.
4. Read the [migration book](../migration/index.md) if you need historical context from Batchalign2.
5. Run the root workspace tests before changing behavior.

## Current Rust Surfaces

The batchalign side of the workspace is split across three crates:

- `crates/batchalign-types/`: shared domain and worker-boundary types
  (worker protocol, language/domain scalars, wire-facing identifiers).
  No filesystem, no network, no model loading.
- `crates/batchalign/`: the application: CLI, HTTP server, worker
  pool, cache, daemon lifecycle, command dispatch. Depends on
  `batchalign-types` and on the `talkbank-*` crates for CHAT
  parsing/validation/transform.
- `crates/batchalign-pyo3/`: the PyO3 bridge, building the
  `batchalign_core` Python module that the worker processes import.
  Worker-runtime-only surface (ASR / FA / media / cantonese-asr
  adapters); no morphosyntax orchestration.

## Setup

```bash
make sync
make build
cargo check --workspace
cargo test --workspace
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
```

`make sync` is the normal setup path even for Cantonese/provider work.
Cantonese ASR engines are part of the main package surface, not a separate
extra or plugin tier.

Rebuild rule of thumb while iterating:

- CLI/server-only changes: `cargo build -p batchalign` or `make build-rust`
- `batchalign` or `crates/batchalign-pyo3/` changes: `make build-python`
- the fast contributor loop: run `cargo build -p batchalign` once, then
  `uv run batchalign3 ...` will use the repo CLI fallback in a source checkout
  after a slim `make build-python`

## Where To Work

- CLI flags, args parsing, cache, daemon, dispatch: `crates/batchalign/src/cli/`, `crates/batchalign/src/cache/`, `crates/batchalign/src/cli/daemon.rs`
- Server routes, jobs, persistence, OpenAPI: `crates/batchalign/src/server.rs`, `crates/batchalign/src/routes/`, `crates/batchalign/src/openapi.rs`
- Worker pool, IPC, daemon spawn: `crates/batchalign/src/worker/`
- Shared CHAT transformations and morphosyntax / FA / UTR / mapping logic: `crates/batchalign-transform/` (and `crates/batchalign/src/chat_ops/` for the batchalign-side adapters that route through it)
- Worker-boundary types and wire-facing scalars: `crates/batchalign-types/`
- Python extension boundary: `crates/batchalign-pyo3/`

## Expectations

- add or update tests before large behavioral changes
- keep public docs in sync with the actual CLI and server surface
- do not introduce maintainer-local filesystem paths into public docs
- treat migration notes as historical context, not as the current API contract

## Useful Commands

```bash
cargo build -p batchalign
make build-python
cargo test -p batchalign --test cli_integration_suite cli::
cargo test -p batchalign --test cli_integration_suite e2e::
cargo test -p batchalign --test cli_integration_suite integration::
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
cargo run -q -p batchalign -- openapi --check --output openapi.json
```
