# Rust Server Migration

**Status:** Landed  
**Last verified:** 2026-03-09

This page records how the Python-server to Rust-control-plane migration ended
up in the current public repo. It is not a future-state plan.

## What changed

The active release architecture is:

- Rust CLI in `crates/batchalign-cli`
- Rust server in `crates/batchalign-app`
- shared CHAT operations in `crates/batchalign-chat-ops`
- Python workers in `batchalign/worker`

The old migration-era labels `rust/`, the nested Rust workspace, `batchalign-server`,
and `batchalign-types` are no longer the current public layout.

## What Rust owns now

Rust owns the control plane and the CHAT lifecycle for server-managed command
paths:

- HTTP API and OpenAPI
- job staging and persistence
- worker pool management
- dispatch routing and daemon lifecycle
- CHAT parsing, validation, extraction, injection, and serialization
- utterance cache access

## What Python still owns

Python still owns model loading and inference:

- Stanza morphosyntax / coref / utseg inference
- ASR and forced-alignment engine backends
- speaker diarization
- optional engine extras such as HK/Cantonese backends

Those capabilities are exposed through the worker protocol, not through a
separate Python server.

## Current repo mapping

| Historical name | Current location |
|-----------------|------------------|
| nested Rust workspace | root Cargo workspace |
| `batchalign-server` crate | `crates/batchalign-app` |
| `batchalign-types` crate | `crates/batchalign-app/src/types/` |
| `batchalign-worker` crate | `crates/batchalign-app/src/worker/` |
| Python server entry point | removed; replaced by `batchalign3 serve` |

## User-visible migration outcome

For public release users, the important outcomes are:

- the documented CLI is `batchalign3`
- `batchalign3 serve` starts the Rust server, not a separate Python web server
- the CLI can talk to an explicit remote server or auto-start a local daemon
- Python is still required for local processing workloads because workers run
  Python inference code

## Developer migration outcome

If you are updating older internal notes or pre-release docs:

- replace nested-workspace paths with root-workspace paths
- replace `batchalign-server` references with `batchalign-app`
- replace `batchalign-types` references with `batchalign-app/src/types/...`
- treat plugin-system notes as historical; engines are now built in-tree

The migration is complete enough that new public docs should describe the
current workspace directly, not the transitional repository split.
