# Workspace Crate Dependency Contract

**Status:** Current
**Last updated:** 2026-10-01 15:39 EDT

## Overview

The batchalign source lives inside the `talkbank-tools` Cargo
workspace as sibling crates under `crates/`. The standalone
`batchalign3` repo was decommissioned on 2026-04-28 and folded in;
there is no longer a cross-repo path-dependency relationship to
maintain. This page summarises the resulting dependency contract
and points readers at the broader release plan.

## Current Dependency Wiring

Inside `crates/batchalign/Cargo.toml`, the runtime crate consumes
the shared talkbank-* crates as workspace dependencies:

```toml
# crates/batchalign/Cargo.toml
talkbank-model = { workspace = true }
talkbank-parser = { workspace = true }
talkbank-transform = { workspace = true }
batchalign-types = { workspace = true }
```

The PyO3 worker-runtime crate at `crates/batchalign-pyo3/Cargo.toml`
has a narrower set of workspace deps, it does **not** consume the
runtime `batchalign` crate:

```toml
# crates/batchalign-pyo3/Cargo.toml
batchalign-types = { workspace = true }
talkbank-transform = { workspace = true }
```

All path resolution is handled by the workspace `Cargo.toml` at the
repo root; there are no `path = "../../../..."` fragments anywhere in
the tree.

## Consumed Crate Surface

| Crate | Consumed by | Purpose |
|-------|-------------|---------|
| `talkbank-model` | `batchalign` (runtime) | CHAT data model, validation, alignment types |
| `talkbank-parser` | `batchalign` (runtime) | CHAT parsing via tree-sitter |
| `talkbank-transform` | `batchalign`, `batchalign-pyo3` | Pipelines, CHAT↔JSON, alignment, morphosyntax, Cantonese normalisation, ASR post-processing, tokenizer realignment |
| `batchalign-types` | `batchalign`, `batchalign-pyo3`, `batchalign-transform` | Shared domain newtypes + V2 worker IPC contracts |

## Compatibility Rules

1. **One workspace, one CI gate.** Changes that cross talkbank-*
   and batchalign- crate boundaries land in the same PR; CI runs
   the whole workspace.
2. **Single-source build commands.** PyO3 rebuilds go through
   `uv run maturin develop -m crates/batchalign-pyo3/Cargo.toml
   -F pyo3/extension-module` or the
   `make batchalign-build-wheel` → `make batchalign-python-prepare`
   chain. The standalone Rust CLI is `cargo build -p batchalign`
   (or `make build` for the dashboard-embedded release path).
3. **Cross-language IPC drift is gated by tests, not docs.** The
   IPC contract is enforced by `crates/batchalign/tests/worker_protocol_v2_compat.rs`
   on the Rust side and
   `batchalign/tests/test_worker_protocol_v2_types.py` on the
   Python side, plus the `scripts/check_ipc_type_drift.sh` CI gate.

## Release Boundary

The talkbank-* crates target eventual publication to crates.io with
stable versioned APIs. Until that happens, the workspace deps shown
above are the single source of truth. See
[Release Contract](../developer/release-contract.md#workspace-dependency)
for the longer-term plan.

## Release Manifest

Each batchalign3 release records:

- The single git SHA (the workspace HEAD) used for the build, since
  there is no longer a second repo to pin separately.
- The build date and CI run URL.
- License metadata: BSD-3-Clause for everything in this workspace
  (see `LICENSE` and `pyproject.toml`).

The release workflow generates the manifest automatically and
attaches it to the GitHub Release notes.
