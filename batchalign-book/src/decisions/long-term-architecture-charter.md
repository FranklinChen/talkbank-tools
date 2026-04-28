# Long-Term Architecture Charter (10+ Years)

## Date

February 24, 2026

## Status

Adopted

## Scope

This charter governs long-term decisions for `batchalign3`:

1. server/control plane architecture
2. dashboard architecture
3. CLI architecture

Out of scope:

- linguistic/CHAT-internal semantics

In-scope CHAT boundary:

- interface contracts between `batchalign3` and CHAT processing components

## Core Principle

Architectural choices must optimize for long-term correctness, clarity, and
maintainability over short-term delivery speed. Significant refactors are
acceptable when they improve the long-term architecture.

## Strategic Commitments

### 1. Stable Platform + Extension Model

- `batchalign3` remains the stable host platform.
- domain/provider-specific behaviors (e.g., HK/Cantonese engines) are built-in
  runtime capabilities, not a separate install tier.
- engine contracts are versioned and treated as public compatibility surfaces.

### 2. Typed Contracts First

- keep typed boundaries across CLI <-> server <-> worker and server <-> dashboard.
- schema and protocol evolution must be explicit, versioned, and CI-gated.
- avoid implicit/undocumented payload drift.

### 3. Composable Control Plane

- server architecture should prioritize composable, replaceable modules
  (queueing, scheduling, telemetry, authn/authz, persistence) over monolithic logic.
- operational features should come from mature libraries where possible.

### 4. Dashboard as Operational Interface

- dashboard should be treated as an operational control surface, not a thin demo UI.
- state, retry, realtime updates, and failure modes must be explicit and testable.
- keep typed API clients generated from canonical contracts.

### 5. CLI as Durable Automation Surface

- CLI flags/commands are a long-lived API for scripts and automation.
- compatibility policy (deprecation windows, aliases, migration notices) is mandatory.
- new engine/plugin capabilities should be selectable without brittle hidden behavior.

### 6. Explicit CHAT Boundary Governance

- CHAT internals are outside this charter.
- the `batchalign3` concern is the integration boundary:
  input/output invariants, error contracts, and performance behavior at that boundary.

## Decision Rules

When evaluating alternatives, prefer options that:

1. reduce long-term coupling and merge debt
2. preserve typed/public contracts
3. isolate provider- and region-specific dependencies
4. improve observability and diagnosability
5. keep migration paths explicit and reversible

## Immediate Implications

1. Provider-specific extensions (e.g., HK/Cantonese engines) are built-in
   runtime capabilities inside the main package.
2. server/dashboard/CLI roadmap items should be justified using this charter.
3. any short-term shortcut that violates these principles requires an explicit
   exception note and sunset plan.

The current concrete server evolution path is captured in
[Fleet Evolution Plan](../architecture/fleet-evolution.md).

Current checkpoint:

- typed retry/failure domain in shared Rust types
- durable attempts and deferred scheduling state in the server store/DB
- explicit `QueueBackend` boundary plus local dispatcher
- explicit single-node lease ownership, renewal, and reclaim for queued jobs
