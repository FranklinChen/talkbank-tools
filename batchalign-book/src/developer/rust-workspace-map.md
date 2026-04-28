# Rust Workspace Map

**Status:** Current
**Last modified:** 2026-03-26 14:05 EDT

Batchalign3 currently has three active Rust surfaces plus one repo-local
automation surface. The following diagram shows the crate dependency
graph, including cross-repo path dependencies to `talkbank-tools`.

```mermaid
flowchart TD
    subgraph batchalign3["batchalign3 workspace"]
        cli["batchalign-cli\n(CLI binary, dispatch,\ndaemon lifecycle)"]
        app["batchalign-app\n(axum server, job store,\ncache, worker pool)"]
        chatops["batchalign-chat-ops\n(CHAT parsing, injection,\nFA, ASR post-processing)"]
        types["batchalign-types\n(domain types, IPC contracts,\nnewtypes)"]
        revai["batchalign-revai\n(Rev.AI ASR client)"]
        xtask["xtask\n(repo automation)"]

        cli --> app
        app --> chatops
        app --> types
        chatops --> types
        revai --> types
        cli --> types
    end

    subgraph pyo3_ws["pyo3/ (separate, excluded from workspace)"]
        pyo3["batchalign-pyo3\n(batchalign_core\nPython extension)"]
    end

    pyo3 --> chatops
    pyo3 --> types

    subgraph talkbank["talkbank-tools (cross-repo path deps)"]
        model["talkbank-model\n(typed AST, validation)"]
        parser["talkbank-parser\n(tree-sitter parser)"]
    end

    chatops -->|"path dep"| model & parser
    pyo3 -.->|"path dep\n(editable only)"| model
```

## 1. Root Cargo Workspace

The main Rust control plane lives at the repository root in [Cargo.toml](../../../Cargo.toml).

Active crates:

| Crate | Responsibility |
| --- | --- |
| `batchalign-chat-ops` | shared CHAT extraction, injection, mapping, validation-adjacent operations |
| `batchalign-app` | server routes, job lifecycle, worker orchestration, OpenAPI, persistence |
| `batchalign-cli` | `batchalign3` binary, argument parsing, dispatch, daemon/log/cache commands |
| `batchalign-types` | shared domain, protocol, and scheduling types |
| `xtask` | repo-local automation and affected-check orchestration |
| `batchalign-app/src/commands/` | released-command specs, family metadata usage, and command-owned wrappers |

Typical commands:

```bash
cargo build -p batchalign-cli
make build-rust
cargo check --workspace
cargo nextest run --workspace
cargo nextest run -p batchalign-cli --test cli
cargo nextest run -p batchalign-app --test integration
cargo xtask affected-rust packages
```

## 2. PyO3 Bridge

The Python extension lives under `pyo3/`.

| Crate | Responsibility |
| --- | --- |
| `batchalign-pyo3` | builds the `batchalign_core` extension module used by Python |

Typical commands:

```bash
make build-python
cargo nextest run --manifest-path pyo3/Cargo.toml
```

## 3. Repo Automation

`xtask/` is the home for repo-local developer automation:

- affected-check selection
- local verification presets
- install/build smoke tests
- repository policy checks that need typed access to workspace metadata

Use `cargo xtask ...` when a task understands the repo graph better than a
shell script does. Keep shell wrappers thin.

## Ownership Boundary

- use the root workspace when the change affects CLI behavior, server APIs, job execution, logs, cache handling, or worker orchestration
- use `crates/batchalign-app/src/commands/` plus the owning modules (`compare.rs`, `benchmark.rs`, `transcribe/`, `fa/`, `morphosyntax/`, `text_batch.rs`) when the change affects command semantics, composition, or output materialization
- use `pyo3/` when the change affects the Python extension surface exposed as `batchalign_core`
- remember that `batchalign-chat-ops` is shared by both surfaces: rebuild
  `batchalign_core` with `make build-python`, and if you plan to run the
  standalone CLI directly after a shared crate change, rebuild the CLI too or
  use `cargo run -p batchalign-cli -- ...`
- use `talkbank-tools` for parser/model/validator behavior that Batchalign consumes as a dependency

## First Files to Read

1. `crates/batchalign-cli/src/lib.rs` — `run_command()`, the single canonical command router
2. `crates/batchalign-cli/src/dispatch/` — CLI dispatch (explicit server vs. auto-daemon)
3. `crates/batchalign-app/src/runner/` — server-side task routing and dispatch shapes
4. `crates/batchalign-app/src/routes/` — Axum HTTP routes
5. `crates/batchalign-app/src/worker/` — worker pool and IPC
6. `crates/batchalign-app/src/commands/` — released-command specs and command-owned wrappers
7. `crates/batchalign-app/src/{compare.rs, benchmark.rs, text_batch.rs, command_family.rs}` — key shared command helpers and reference/composite seams
8. `pyo3/src/lib.rs` — PyO3 module organization and entry points

See also: [Rust CLI and Server](rust-cli-and-server.md) for detailed dispatch
documentation and the checklist for adding new commands.
