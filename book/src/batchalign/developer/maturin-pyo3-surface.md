# Maturin Build and PyO3 Dependency Surface

**Status:** Current
**Last updated:** 2026-10-15 13:28 EDT

## Overview

The `batchalign_core` Python extension is built by **maturin** from `crates/batchalign-pyo3/Cargo.toml`.
The crate has exactly one feature gate: `extension-module` (required by PyO3 for
cdylib linking). No other features exist, the extension is always slim.

## Dependency Graph

```text
batchalign-pyo3 (the .so)
  |
  +-- batchalign-types       (newtypes, worker IPC types)
  +-- talkbank-transform     (Cantonese ASR projection, Cantonese normalization,
  |                           tokenizer realignment, asr_postprocess,
  |                           morphosyntax, text task normalization)
  +-- pyo3, numpy, serde, serde_json, tracing, tracing-subscriber
```

That's it. ~319 crates in the full dependency tree. No server, no CLI, no
Rev.AI, no talkbank-model, no talkbank-parser.

### Why each dependency exists

| Crate | Used by pyo3 for | Could be removed? |
|-------|-------------------|-------------------|
| `batchalign-types` | Domain newtypes, worker IPC types (`ExecuteRequestV2`, etc.) | No, core shared types |
| `talkbank-transform` | Cantonese ASR projection, Cantonese normalization, tokenizer realignment, coref types, text result normalization, morphosyntax sentence mapping (post-crate-split home of all the formerly-`batchalign`-side worker logic) | No, worker-side Rust logic |
| `pyo3` / `numpy` | PyO3 bridge, NumPy array handling for audio | No, fundamental |
| `serde` / `serde_json` | JSON serialization for IPC | No, fundamental |
| `tracing` / `tracing-subscriber` | Worker-process logging (env-filtered) | No, required for diagnostics |

### What was removed

| Removed dep | Why |
|-------------|-----|
| `batchalign` (+ transitive `batchalign`) | CLI binary shipped as package data instead of compiled into .so |
| `batchalign-revai` | Dead code, server uses Rev.AI directly |
| `talkbank-model` | Only used by deleted ParsedChat class |
| `talkbank-parser` | Only used by deleted parse helpers |
| `indexmap`, `thiserror` | Only used by deleted standalone functions |

## CLI Binary Distribution

The `batchalign3` CLI is a standalone Rust binary (`crates/batchalign`).
It is **not** compiled into the .so extension. Instead:

- **GitHub Release wheels**: The binary is pre-built and included as package
  data at `batchalign/_bin/batchalign3`. The console-script entry point
  (`batchalign/_cli.py`) finds and execs it. BA3 itself is not published to
  PyPI.
- **Dev checkout**: `_cli.py` falls back to `target/debug/batchalign3` or
  `cargo run -p batchalign`.

This eliminates the old `cli-entry` feature gate that dragged 741 extra crates
(the entire server stack) into the extension build.

The wrapper is intentionally thin, but it does carry two load-bearing runtime
handoffs into the Rust binary:

- `BATCHALIGN_PYTHON`: preserve the interpreter/venv that owns the installed
  worker package
- `BATCHALIGN_SELF_EXE`: preserve the actual packaged Rust binary path so
  server/daemon re-exec paths do not have to infer it from the Python
  console-script launcher

## Build Commands

```bash
# Development rebuild (debug, fast, incremental)
make batchalign-develop
# Or via the Makefile target chain (build wheel + install into the dev env):
make batchalign-python-prepare

# Release wheel for deployment
cargo build --release -p batchalign --bin batchalign3
cp target/release/batchalign3 batchalign/_bin/batchalign3
uv run maturin build --release \
    -m crates/batchalign-pyo3/Cargo.toml \
    -F pyo3/extension-module --out dist/

# Check compilation without building wheel
cargo check --manifest-path crates/batchalign-pyo3/Cargo.toml
```

## What NOT to do

- **Do not add server deps to pyo3.** The extension is for the worker process.
  If the server needs Rust functionality, use `batchalign` or
  `batchalign` directly, not through pyo3.

- **Do not vendor types.** Use path dependencies. `batchalign-types` is the
  single source of truth for domain newtypes and worker IPC types.

- **Do not add feature gates.** The extension should always build the same way.
  If something is optional, it probably doesn't belong in pyo3.

- **Do not compile the CLI into the .so.** The binary is shipped as package
  data. If you need to change how the CLI is invoked, modify `_cli.py`.

- **Do not move orchestration into `_cli.py`.** The wrapper may pass runtime
  hints into Rust, but the actual CLI/server behavior still belongs to the Rust
  binary.

## Verification checklist

Development and wheel qualification are different modes. See the
[development install procedure](rust-core.md#development-installs).
A bare `uv run` is not evidence that edited Rust was rebuilt.

After any dependency change to `crates/batchalign-pyo3/Cargo.toml`:

```bash
cargo check --manifest-path crates/batchalign-pyo3/Cargo.toml
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
make batchalign-develop
make _batchalign-test-python BATCHALIGN_PYTEST_ARGS="batchalign/tests/test_worker_fa_v2.py -n 0"
```
