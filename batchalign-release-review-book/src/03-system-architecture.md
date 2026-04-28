# Chapter 3: System Architecture

## What Is Good

The direction of travel is broadly correct:

- Rust owns more orchestration and CHAT semantics
- Python is being pushed toward a narrower model-host role
- protocol typing is much stronger than in a typical academic toolchain
- architecture intent is documented unusually well

Those are real assets. They are the reason a full rewrite is not currently the
best move.

## What Is Weak

The architecture is still carrying too many partially overlapping product
surfaces:

- Python package entrypoint
- standalone Rust CLI
- server/daemon mode
- React dashboard
- Tauri desktop shell
- installer scripts
- sibling dependency workspace

Each surface adds value, but the system-level ownership model is still too
broad. The result is not one clean product architecture; it is several
reasonable architectures superimposed on one another.

## Maintainability Signals

Approximate source size in non-test code:

- `crates/batchalign-app/src`: 56.6k lines
- `crates/batchalign-chat-ops/src`: 29.6k lines
- `crates/batchalign-cli/src`: 14.7k lines
- `batchalign`: 14.5k lines
- `frontend/src`: 10.9k lines

The biggest control-plane files are not normal-sized modules. They are
coordination zones:

- `worker/pool/mod.rs`: 1421 lines
- `runner/util/file_status.rs`: 1395 lines
- `worker/handle/mod.rs`: 1360 lines
- `runner/mod.rs`: 1107 lines
- `store/job/mod.rs`: 1040 lines
- `types/config.rs`: 999 lines

Large files are not automatically bad, but this pattern is a strong signal that
responsibility boundaries are still too wide in the parts of the system most
likely to cause operational failures.

## Structural Diagnosis

The codebase currently has three different kinds of complexity:

1. Necessary complexity:
   linguistic pipelines, workers, multiple engines, multiple product surfaces.
2. Transitional complexity:
   Python-to-Rust migration leftovers, legacy shims, dormant surfaces.
3. Self-inflicted control-plane complexity:
   large orchestration modules, duplicated lifecycle logic, and many optional
   runtime modes that have not yet converged into a small number of stable paths.

The release program should attack category 3 first, not category 1.

## Action Items

- Declare one primary public product path for 1.0.x:
  - likely CLI + local server + documented installer path
- Explicitly demote or quarantine dormant surfaces that are not release-grade.
- Split oversized orchestration modules by responsibility:
  - worker spawn/lifecycle
  - transport/protocol
  - scheduling/policy
  - persistence
  - operator-facing status projection
- Require every large file reduction to preserve behavior with focused
  regression tests rather than speculative refactors.
