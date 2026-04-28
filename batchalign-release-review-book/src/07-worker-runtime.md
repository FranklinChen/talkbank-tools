# Chapter 7: Worker Runtime and Concurrency

## This Subsystem Is Central and Still Too Broad

The worker runtime is one of the most important parts of the system:

- it owns model process lifecycle
- it defines the Rust/Python reliability boundary
- it is where concurrency bugs become user-visible corruption or hangs

It is also still too structurally broad.

Relevant source size signals:

- `worker/pool/mod.rs`: 1421 lines
- `worker/handle/mod.rs`: 1360 lines
- `worker/pool/shared_gpu.rs`: 768 lines

## Observability Currently Favors Non-Blocking Approximation Over Truth

`worker/pool/status.rs` explicitly uses `try_lock()` and relaxed reads for
status projections:

- `worker/pool/status.rs:1-77`

That means the system is willing to report `0` workers or `false` availability
when it cannot immediately acquire the status lock.

This may be a reasonable local optimization, but it is a poor default for
release-grade operator truth. Under contention, the diagnostics become least
reliable precisely when the operator most needs them.

## Unix-Centric Lifecycle Semantics

Worker lifecycle control depends heavily on Unix process-group behavior:

- `worker/handle/mod.rs:191-204`
- `worker/handle/mod.rs:1026-1071`
- similar logic in CLI daemon/server startup paths

That may be correct on Unix. It does not amount to a proven cross-platform
process-management story.

## Internal Panic as Control-Flow Guard

The transcribe pipeline still contains panic-on-invariant logic:

- `pipeline/transcribe.rs:71-87`

Internal invariants matter, but long-lived public software should minimize
panic as a runtime correctness guard in orchestration paths. Panic is a poor
user contract.

## Recommended Direction

Do not throw away the whole worker design. The direction is sound:

- explicit worker processes
- typed requests
- narrower Python responsibilities

But refactor the runtime into smaller ownership units:

- spawn/bootstrap
- transport
- health/readiness
- lifecycle/shutdown
- concurrency/router
- status/observability

Each of those should become independently testable and less likely to accrete
another 500 lines of coupled behavior.

## Action Items

- Split worker runtime modules by ownership boundary, not by historical accretion.
- Replace approximate status reporting with truthful or explicitly degraded
  reporting.
- Add concurrency tests for shared GPU worker routing and failure cleanup.
- Reduce panic-based invariant enforcement in orchestration code; return typed
  internal errors where possible.
- Add platform-specific lifecycle tests for shutdown and cleanup behavior.
