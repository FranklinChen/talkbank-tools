# Fleet Evolution Plan

> **Note:** This is a design direction document, not a user guide. It describes
> the architectural trajectory from today's single-node server toward future
> multi-node capabilities. The current server is fully functional as a
> single-node system; multi-node fleet support is future work.

This document makes the long-term server direction explicit.

It answers one question:

How should `batchalign3` evolve from today's single-node Rust server into a
future fleet-capable system without overengineering too early?

## Current State

Today the server already has several important building blocks:

- durable job metadata in SQLite
- a Rust HTTP control plane
- a Rust-managed Python worker pool
- typed worker IPC
- crash recovery for jobs
- worker health checks and restart of idle workers
- a memory gate before job start

What it does **not** yet have is an explicit long-term control-plane model for:

- retry policy
- deferred execution
- attempt tracking
- multi-node leasing
- fleet capability routing
- queue/backend abstraction

Those are the concepts we should introduce next.

The first step of that work now exists in code as shared domain types in
`crates/batchalign-app/src/types/scheduling.rs`, which defines:

- `WorkUnitKind`
- `FailureCategory`
- `AttemptOutcome`
- `RetryDisposition`
- `RetryPolicy`
- `AttemptRecord`

The current server also consumes the first slice of that model:

- per-file failures now carry typed `FailureCategory` values in the Rust API
  layer instead of ad hoc strings
- `/health` and WebSocket snapshots expose early control-plane counters for
  started attempts, retry-marked attempts, and deferred work units
- runner code increments attempt-start counters at work-unit dispatch points
- SQLite now has a durable `attempts` table, and the runner records attempt
  start/finish transitions for file-backed work units
- per-file worker-backed dispatch now has a first local retry policy for
  transient failures on `process`, forced alignment, and transcribe paths,
  using typed `RetryPolicy` backoff and persisted `next_eligible_at`
  timestamps on file status rows
- batched text infer dispatch now retries transient whole-batch worker
  failures locally before projecting them into per-file terminal results
- deferred jobs now persist a job-level `next_eligible_at`
- job routes no longer spawn runner tasks directly; a queue backend owns
  queued-job signaling and claiming, and a separate local dispatcher launches
  runners from claimed jobs
- memory-gate deferral now re-queues jobs by writing job-level
  `next_eligible_at` and handing control back to the queue layer
- queued jobs now carry explicit single-node lease state (`leased_by_node`,
  heartbeat, expiry) when claimed by the local dispatcher
- `/health` now exposes a concrete `node_id` so lease ownership has an
  explicit local identity
- active local runner tasks now renew their job lease periodically
- the local queue can reclaim expired queued leases instead of waiting for
  normal release only

That is intentionally only a first slice. Durable attempt persistence,
which used to be future work, now exists. Backoff scheduling and retry
execution policy still belong to the next phase of implementation work.

## Design Goal

Keep the existing Rust server/worker architecture as the execution core.

Do **not** replace it with a generic workflow framework now.

Instead, refactor the control plane so it can later support:

- single-node execution today
- multi-node fleet execution later
- a durable external queue/event substrate when justified

## Preconditions Before Fleet Returns

Before reintroducing fleet work, the single-node server should keep moving away
from raw catch-all runtime structs and toward typed projections.

The most important preconditions are:

- decompose `Job` into clearer internal views such as submission payload,
  scheduler state, runner snapshot, and API response
- keep queue code operating on claimable scheduling projections rather than the
  full job aggregate
- keep runner code operating on immutable snapshots instead of reopening the raw
  job map for static configuration
- keep DB rows and HTTP payloads as boundary types rather than letting them
  become the execution model

That work matters even in single-node mode because fleet will otherwise force
the same decomposition later under much higher complexity.

## Architectural Direction

The long-term system should separate into three layers.

### 1. Job Metadata Store

Responsible for:

- job definitions
- file-level state
- attempts
- error classification
- scheduling timestamps
- audit/history

Likely long-term home:

- Postgres in fleet mode
- SQLite remains acceptable for local/single-node mode

### 2. Queue / Dispatch Substrate

Responsible for:

- pending work delivery
- durable acknowledgement
- delayed retry
- backoff
- queue fairness primitives
- fleet fan-out

Likely long-term direction:

- in-process queue today
- external durable queue later
- likely candidate: NATS JetStream

### 3. Node-Local Execution Engine

Responsible for:

- worker lifecycle
- Python runtime management
- health checks
- capability discovery
- local worker reuse
- command/lang-specific execution

This is the current Rust server + worker-pool model and should be preserved.

## Explicit Domain Model To Introduce Now

Before introducing fleet infrastructure, the current server should grow a
clearer internal model.

### Job

Already exists, but should eventually include:

- `priority`
- `submitter_class` or tenancy metadata
- `queue_class`
- `retry_policy_id`
- `next_eligible_at`

### Work Unit

The scheduler should stop thinking only in terms of "job" and "file task".

The unit that gets retried or leased should be explicit:

- per-file process task
- per-file infer task
- per-file FA task
- future multi-stage task variants

This does **not** have to be a public API type yet, but it should be explicit
in the runner/store design.

### Attempt

Add a durable notion of attempt:

- `attempt_id`
- `job_id`
- `work_unit_id`
- `attempt_number`
- `started_at`
- `finished_at`
- `outcome`
- `error_category`
- `retryable`
- `worker_node_id`
- `worker_pid` or runtime identifier

Without this, retries and fleet diagnostics remain guesswork.

### Retry Policy

Retry behavior should become data, not scattered control flow.

At minimum:

- `max_attempts`
- `initial_backoff_ms`
- `max_backoff_ms`
- `backoff_multiplier`
- `retry_on_categories`

### Lease

Fleet mode should use a lease model for claimed work:

- `leased_by_node`
- `lease_expires_at`
- `heartbeat_at`

Single-node mode can implement this trivially at first, but the abstraction
should exist before fleet returns.

### Capability Advertisement

Today capability is mostly command/task support. Fleet mode will need more:

- command support
- infer-task support
- language support
- provider/engine availability
- runtime class (main vs sidecar)
- hardware traits
- media locality / root access

## Error Taxonomy Direction

Retries only work if the system distinguishes retryable from terminal failures.

The server should standardize categories such as:

- `validation`
- `parse_error`
- `input_missing`
- `worker_crash`
- `worker_timeout`
- `worker_protocol`
- `provider_transient`
- `provider_terminal`
- `memory_pressure`
- `cancelled`
- `system`

Then define retry defaults by category instead of per-call ad hoc behavior.

## Phased Plan

## Phase 1: Local Control-Plane Hardening

Goal: make the current single-node server express the concepts fleet mode will
need later.

Add:

- attempt records
- retryable vs terminal error classification
- explicit retry policy for transient failures
- deferred/backoff state (`next_eligible_at`)
- explicit scheduler state instead of only immediate task spawn

Do **not** add:

- distributed queue
- multi-node coordination
- external workflow framework

Success criteria:

- transient worker crashes/timeouts can retry automatically
- memory-gate failures can defer instead of immediately failing
- health endpoints expose attempts/retries/deferred counts

Current checkpoint:

- typed failure taxonomy: started
- health-level attempt/retry/defer counters: started
- durable attempt records: implemented for file-backed work units
- retry/backoff execution policy: implemented for local per-file worker-backed
  dispatch and whole-batch text infer calls, not yet generalized to
  queue-level rescheduling of batch work units
- deferred scheduling timestamps: implemented at file status level via
  `next_eligible_at`, and at job level via job-level `next_eligible_at`
- local queue seam: implemented as an explicit `QueueBackend` boundary plus a
  concrete `LocalQueueBackend` and host-side `QueueDispatcher`
- memory-gate deferral: now re-queues jobs by writing `next_eligible_at` and
  returning them to the local backend, rather than sleeping inside the runner
- node identity and job leases: implemented in first single-node form so queue
  claims are explicit ownership transitions instead of only in-memory flags
- lease renewal and expiry-based reclaim: implemented for the local backend
  so single-node execution already follows the intended lease lifecycle

## Phase 2: Internal Queue Abstraction

Goal: separate job metadata from dispatch transport.

Introduce a queue/backend boundary such as:

- `QueueBackend`
- `LocalQueueBackend`
- future `FleetQueueBackend`

The runner should stop assuming:

- submission immediately means `tokio::spawn`
- the in-memory semaphore is the only scheduler

Success criteria:

- current behavior preserved on top of a local backend
- retry/defer logic no longer lives directly in HTTP route handlers
- queued job eligibility is owned by the backend instead of ad hoc runner
  sleeps
- host-side dispatch launch is separate from backend claim/signaling policy

## Phase 3: Node Identity and Leasing

Goal: make each server instance look like a future fleet node.

Add:

- `node_id`
- node registration / capability snapshot
- lease acquisition for runnable work
- lease expiry / reclaim semantics

Even if there is only one node at first, this step removes hidden
single-process assumptions.

Success criteria:

- work ownership is explicit
- crashed/inactive nodes can have leases reclaimed

## Phase 4: Fleet Reintroduction

Goal: allow multiple Rust server nodes to pull work safely.

Add:

- shared metadata store
- shared queue substrate
- node capability routing
- locality-aware dispatch rules

Keep:

- node-local worker pool
- typed worker IPC
- current execution semantics

Success criteria:

- multiple nodes can process jobs safely
- retry/defer survives node loss
- job status remains coherent across the fleet

## Phase 5: External Durable Queue / Event Substrate

Goal: move queue durability and event fan-out out of process when needed.

Likely target:

- NATS JetStream for queue/event transport

Probable shape:

- Postgres for jobs, attempts, results, audit state
- JetStream for pending work, acknowledgements, redelivery, event streams
- Rust server nodes act as lessors/executors

This phase is only justified once the local abstractions already exist.

## What We Should Not Do Yet

- adopt a full workflow engine now
- rewrite the worker pool around a generic task library
- introduce distributed infrastructure before retries/attempts are explicit
- bake JetStream assumptions directly into current business logic

## Near-Term Implementation Order

1. Move more eligibility and redelivery logic out of runner control flow and
   into the queue/backend layer.
2. Expose attempt/defer/lease state more directly in job detail and operations
   tooling.
3. Add a fleet-capable backend implementation behind the existing
   `QueueBackend` boundary when multi-node execution returns.
4. Introduce explicit multi-node lease conflict/reclaim rules once real fleet
   workers exist.

## Decision Rule

If a change improves:

- retry semantics
- explicit scheduling state
- queue/backend replaceability
- future node leasing

then it is probably aligned with the long-term architecture.

If a change only adds more special-case logic inside today's runner without
creating better abstractions, it is probably short-term debt.
