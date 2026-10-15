# Rust Workspace Map

**Status:** Current
**Last updated:** 2026-10-15 13:28 EDT

The Batchalign code lives inside the `talkbank-tools` Cargo workspace
as sibling crates under `crates/`. Generic CHAT parser/model/transform crates
come from the pinned published Chatter dependency; Batchalign-owned crates
build in this workspace. Do not replace those pins with local path overrides.

```mermaid
flowchart TD
    subgraph batchalign["Batchalign crates"]
        ba["batchalign\n(CLI binary, axum server, job store,\nworker pool, CHAT ops, FA, ASR post-processing,\nRev.AI client)"]
        types["batchalign-types\n(domain types, IPC contracts, newtypes)"]
        pyo3["batchalign-pyo3\n(batchalign_core Python extension)"]

        ba --> types
        pyo3 --> types
    end

    subgraph tt["talkbank-tools core"]
        model["talkbank-model"]
        parser["talkbank-parser"]
        transform["talkbank-transform"]
    end

    ba --> model & parser & transform
    pyo3 --> transform
```

## Crates

| Crate | Responsibility |
| --- | --- |
| `batchalign` | the `batchalign3` binary plus the axum server, dispatch, daemon lifecycle, worker pool, job store, cache, CHAT extraction/injection/mapping, FA, ASR post-processing, and the Rev.AI client. The bulk of the runtime. |
| `batchalign-types` | shared domain, protocol, and scheduling types (newtypes, worker IPC contracts), the small low-dependency crate everything else builds on. |
| `batchalign-pyo3` | builds the `batchalign_core` Python extension module via maturin. Workspace deps are `batchalign-types` + `talkbank-transform`; it does NOT depend on the runtime `batchalign` crate. |
| `xtask` | repo-local automation: affected-check selection, install/build smoke tests, repository policy checks. |

## Internal organization of the `batchalign` crate

Top-level modules under `crates/batchalign/src/`:

- `cli/`, `commands/`: CLI argument parsing, released-command specs,
  dispatch.
- `routes/`, `db/`, `pipeline/`, `planning/`, `worker/`, `execution/`,
  axum HTTP server, job store, worker orchestration.
- `chat_ops/`: CHAT extraction and injection.
- `fa/`: forced alignment and review-tier handling.
- `morphosyntax/`: `%mor`/`%gra` injection and Stanza interaction.
- `revai/`: Rev.AI ASR client.
- `host_facts/`, `host_memory.rs`, `host_policy.rs`: host-aware
  runtime configuration.
- `cache/`, `provenance.rs`, `compare.rs`, `benchmark.rs`: caching,
  provenance metadata, and compare/benchmark utilities.

## Typical commands

```bash
cargo build -p batchalign
cargo check --workspace
cargo test --workspace
cargo test -p batchalign --test cli_integration_suite cli::
cargo test -p batchalign --test cli_integration_suite integration::
cargo xtask affected-rust packages

# Python extension build
make batchalign-develop
# or, for a full wheel install into the dev env:
#   make batchalign-build-wheel && make batchalign-python-prepare
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
```

## Where to make changes

- CLI behavior, server APIs, job execution, logs, cache handling,
  worker orchestration, CHAT ops, FA, morphosyntax, Rev.AI →
  `crates/batchalign/`.
- Shared newtypes or worker IPC contracts → `crates/batchalign-types/`.
- Python extension surface (`batchalign_core`) →
  `crates/batchalign-pyo3/`.
- Parser, data model, validation, or transform behavior consumed by
  Batchalign → the corresponding `talkbank-*` crate in the same
  workspace.

## First files to read

1. `crates/batchalign/src/cli/mod.rs:251`: `run_command()`, the
   canonical command router.
2. `crates/batchalign/src/cli/`: CLI dispatch (explicit server vs.
   auto-daemon).
3. `crates/batchalign/src/execution/`: server-side task routing and
   dispatch shapes.
4. `crates/batchalign/src/routes/`: Axum HTTP routes.
5. `crates/batchalign/src/worker/`: worker pool and IPC.
6. `crates/batchalign/src/commands/`: released-command specs and
   command-owned wrappers.
7. `crates/batchalign-pyo3/src/lib.rs`: PyO3 module organization and
   entry points.

See also: [Rust CLI and Server](rust-cli-and-server.md) for detailed
dispatch documentation and the checklist for adding new commands.
