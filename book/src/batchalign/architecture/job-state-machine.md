# Job State Machine

**Status:** Current
**Last updated:** 2026-11-09 12:34 EST

## Overview

This page documents the job lifecycle state machine, the allowed states,
transitions, and validation rules for jobs managed by the batchalign3 server.

## Current Implementation

Every job status change goes through one gate, and legality is checked
against an explicit table:

| Piece | Where | Role |
|---|---|---|
| `Job::set_status` | `store/job/lifecycle.rs` | The ONLY place `execution.status` is assigned outside test fixtures |
| `JobStatus::can_transition_to` | `types/status.rs` | The transition table below, in code |
| `StatusChange` | `types/status.rs` | Why the change is happening |

The table is keyed by the REASON, not by the from/to pair alone, because the
same pair can be legal or illegal depending on what caused it: `Cancelled ->
Queued` is legal as an operator `Restart` and illegal as a `MemoryRequeue`.
Conflating those is what allowed the memory gate to resurrect a cancelled
job (2026-07-29).

A writer whose transition is refused must persist nothing. Both the registry
wrappers and the store-level helpers propagate the refusal for exactly this
reason; a refused finalize that still wrote its status to SQLite left rows
that startup recovery never revisits.

## Incompatible recovered command/options

A persisted job whose command and typed options disagree is shown as `Failed`
with a recovery explanation. The original job and file rows are left unchanged,
including metadata that ordinary recovery would repair. File statuses remain
available for inspection; result artifacts are not guessed from incompatible
options. This job is not queued automatically.

`Job::prepare_for_restart` is a fallible transition: it admits command/options
compatibility and restart eligibility before mutating files, cancellation,
leases or run generation. An incompatible job cannot be restarted, so the
restart endpoint cannot launch its runner or persist a replacement status.
Resolve the stored mismatch through a reviewed repair or submit a correctly
specified new job; this behavior does not provide an automatic repair command.

## State Diagram

`Running --> Interrupted` is the GRACEFUL path, where `interrupt_for_shutdown`
runs. A hard crash never reaches it, so the row is still `Running` when
recovery finds it and is reconciled straight from there; that is why
`JobStatus::is_recoverable` accepts both.

```mermaid
stateDiagram-v2
    [*] --> Queued: submit job

    Queued --> Running: worker available\nmark_running()
    Running --> Completed: all files done\nfinalize()
    Running --> Failed: any file errored\nor dispatch error\nfinalize()
    Running --> Cancelled: user cancel\nrequest_cancellation()
    Running --> Queued: memory gate\nrequeue_after_memory_gate()

    Queued --> Cancelled: user cancel\nrequest_cancellation()

    Failed --> Queued: restart\nprepare_for_restart()
    Cancelled --> Queued: restart\nprepare_for_restart()
    WritebackFailed --> Queued: restart\nprepare_for_restart()

    Running --> WritebackFailed: remote copy failed\nfinalize()
    Running --> Interrupted: graceful shutdown\ninterrupt_for_shutdown()

    Interrupted --> Queued: reconcile (resumable)
    Interrupted --> Failed: reconcile (any errored)
    Interrupted --> Completed: reconcile (all done)

    Running --> Queued: hard crash, reconcile (resumable)
    Running --> Failed: hard crash, reconcile (any errored)
    Running --> Completed: hard crash, reconcile (all done)

    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]
    WritebackFailed --> [*]
```

## States

| State | Terminal? | Restartable? | Description |
|-------|:---------:|:------------:|-------------|
| `Queued` | No |: | Waiting for worker availability |
| `Running` | No |: | Actively processing files |
| `Completed` | Yes | No | All files succeeded |
| `Failed` | Yes | Yes | One or more files errored |
| `Cancelled` | Yes | Yes | User-initiated cancellation |
| `Interrupted` | Yes | No | Server crash detected (transient, reconciled at startup) |
| `WritebackFailed` | Yes | Yes | Remote results lost during copy-back |

## Status transition gate

`JobStatus::can_transition_to`, `StatusChange` and `Job::set_status` define
which lifecycle transitions are legal. Timestamps, lease clearing and failure
reasons remain owned by the corresponding `Job` method. A declined transition
keeps the status already accepted by cancellation or shutdown.

Runner completion has a narrower owning boundary: the caller supplies its
linear claim and dispatch outcome, never a proposed final job status. The
registry admits the generation, closes unfinished files and derives the final
status in one mutation. See [cancellation and completion](#what-a-cancel-stops).

## Runner Lifecycle

Every queued job must have a scheduled runner or a live attempt handing off
to its replacement. Only one attempt may hold its exclusive execution claim.
Submission and memory-gate requeue preserve that scheduling obligation:

### 1. Submit path

`submit_job()` in `server_backend.rs` spawns `job_task` immediately after
inserting the job:

```rust,ignore
runtime.spawn_detached(job_task(job_id, host.clone()));
```

### 2. Memory-gate requeue path

When the host-memory coordinator rejects a job, `run_hosted_job` returns
`Ok(HostedJobRunOutcome::Requeued { retry_at })`. The `job_task` match arm
catches this outcome and spawns a delayed replacement runner:

```rust,ignore
Ok(HostedJobRunOutcome::Requeued { retry_at }) => {
    // Never negative: a deadline already past is a delay of zero.
    let delay = retry_at.saturating_duration_since(host.store.now());
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        job_task(job_id_retry, host_retry).await;
    });
}
```

The attempt has already consumed its claim and stopped its owned heartbeat
before returning `Requeued`. The replacement task must obtain a new claim
before acquiring the semaphore and memory gate after the backoff. A delayed
task for an already terminal job receives no dispatch authority.

**Why this matters:** if `Requeued` were silently discarded (e.g., swallowed by
`if let Err(e) = ...`), the job would remain `Queued` forever with
`next_eligible_at` set but no runner. Every subsequent submission of the same
files would receive a 409 conflict with no way to resolve it short of a manual
cancel.

### 3. Bootstrap path

When the daemon starts, `bootstrap_local_server_backend` calls
`store.queued_job_ids()` after `load_from_db()` and spawns `job_task` for each
recovered `Queued` job:

```rust,ignore
for job_id in queued_job_ids {
    runtime.spawn_detached(job_task(job_id, host.clone()));
}
```

This handles jobs that were `Queued` when the daemon was last stopped, either
because they survived a crash, or because a requeue runner was lost when the
process exited before the backoff timer fired.

`recover_interrupted()` (also called at startup) handles the complementary
case: jobs that were `Running` at crash time are marked `Interrupted`, then
reconciled to `Queued`/`Failed`/`Completed`. The bootstrap spawn above then
picks them up if they land in `Queued`.

**Invariant:** after bootstrap completes, every `Queued` job in the DB has
exactly one active `job_task` runner.

### Why `job_task` is not an `async fn`

`job_task` is declared as a plain function returning
`Pin<Box<dyn Future<Output=()> + Send + 'static>>` rather than `async fn`.
This is required by the requeue self-spawn:

```rust,ignore
tokio::spawn(async move {
    sleep(delay).await;
    job_task(job_id_retry, host_retry).await;  // recursive call
});
```

With `async fn`, Rust's `Send` inference on the opaque `impl Future` return
type becomes self-referential. The compiler can prove `Send` from the outside
(non-recursive call site) but not from inside the body when the future awaits
itself inside `tokio::spawn`. The explicit boxed return type breaks the cycle:
`job_task(...)` now returns a concrete `Pin<Box<dyn Future + Send>>` whose
`Send`-ness the compiler can verify unconditionally.

## File Locations

| File | Role |
|------|------|
| `types/status.rs` | `JobStatus`, `StatusChange`, `can_transition_to` (the table) |
| `store/job/types.rs` | `Job` struct |
| `store/job/lifecycle.rs` | `Job::set_status` (the gate) and the transition methods |
| `runner/execution.rs` | Claimed dispatch and consuming run completion |
| `routes/jobs/lifecycle.rs` | Cancel/restart requests |
| `store/queries/recovery.rs` | Crash recovery |

## Runner ownership, restart safety, and requeue (canonical invariants)

Every submitted job gets a `job_task` runner: a self-contained async task
that owns the full lifecycle from semaphore acquire through finalization.

**Exclusive-runner invariant (2026-07-10).** Before executing, every runner
claims exclusive ownership via `store.begin_runner()`; a second runner for
the same job (typically a restart racing the cancelled runner's teardown)
WAITS on that claim instead of running concurrently. The claim also
establishes the queue lease, so the per-job heartbeat loop genuinely renews
it (and doubles as a stall alarm: ERROR log when a live runner shows no
file activity for 30 minutes). Every restart bumps the job's
`RunGeneration`; completion consumes the `RunnerClaim` and admits its
cleanup in the same registry mutation as its generation check. A superseded
claim cannot sweep or finalize a restarted population (the failure mode was: restart re-queues files, the stale
runner force-fails them as "did not reach terminal status" and finalizes
the job `failed` over the healthy new run). Contract test:
`tests/restart_handoff.rs`.

`job_task` is a **non-recursive function returning `Pin<Box<dyn Future>>`**
(not `async fn`). This is deliberate: the `Requeued` branch spawns a fresh
`job_task` inside `tokio::spawn`. With `async fn`, Rust's Send inference
becomes circular on the self-referential opaque return type. The explicit
boxed future gives the inner call a concrete, provably-Send type.

**Requeue invariant**: when the memory gate rejects a job:

1. `run_hosted_job` returns `Ok(HostedJobRunOutcome::Requeued { retry_at })`
2. `job_task` catches this in the `match` arm (not `if let Err(...)`)
3. It spawns a new delayed `job_task` via `tokio::spawn(sleep + job_task(...))`
4. The current `job_task` instance finishes: `lease_task.abort()` +
   `release_runner_claim()` run unconditionally at the bottom
5. The new task re-acquires the semaphore and memory gate after the backoff

Without step 3, a requeued job stays `Queued` forever with no runner and
blocks all future submissions of the same files (409 conflict).

**Bootstrap invariant**: queued jobs loaded from the DB at startup:

`bootstrap_local_server_backend` calls `store.queued_job_ids()` immediately
after `load_from_db()` and spawns `job_task` for each recovered `Queued` job
via `runtime.spawn_detached(job_task(job_id, host.clone()))`. This fulfills
the recovery path when the daemon is restarted after a crash or a memory-gate
rejection that lost its runner.

**Recovery is a two-step sequence that only fires at startup:**

1. `db.recover_interrupted()` (`db/recovery.rs`) reads each `queued` or
   `running` row's status columns as their typed image
   (`JobStatusColumns`), applies `JobStatusColumns::stopped` with
   `Stop::Interrupted` (the one statement of the transition, which a live
   job's shutdown and cancellation also apply through `Job::stop`:
   the stop's status,
   completed now, no pending retry time, error and worker count kept), and
   writes the image whole, with each of the job's files the interruption
   moves, in one transaction per job. It does NOT touch existing
   `interrupted` rows and does NOT requeue.
2. `store.load_from_db()` (`store/queries/recovery.rs`) reads each row
   back into memory. For any job with `status ∈ {Interrupted, Running}`,
   it calls `Job::reconcile_recovered_runtime_state()`: which transitions
   the in-memory job (and writes back to the DB) to `Queued` if any file
   is resumable, or to a terminal state otherwise.

If the daemon stays alive across CLI sessions, neither step runs: the
bootstrap spawn after `load_from_db()` is the only mechanism that rescues
orphaned `Queued` jobs.

### What a cancel stops

A cancel stops job-slot and host-memory admission, command preparation and
in-flight file work.
`begin_runner` atomically mints a non-cloneable `RunnerClaim` with the run
generation and its `Arc<RunnerJobSnapshot>` (including cancellation).
`RunScope::for_run` consumes that claim; only the resulting scope can dispatch
this attempt. Children inherit a cloneable `RunContext`, which grants
cancellation and job registration but no dispatch or completion authority.

Consuming `RunScope::execute` supervises admission through command dispatch,
including capability probing and worker pre-scaling. Its heartbeat starts only
after the claim exists and is stopped when execution ends. Cancellation wins
before work is polled and drops pending admission, dispatch and worker startup.
`EndedDispatch` returns the claim with completion, deferral or failure; it
cannot be reused for another execution. Completion, failure and deferral each
consume the unique claim. A task that never owned a claim cannot release one.

If cancellation was accepted before claim creation, `StoppedRun` grants only
file completion. It has no conversion into a dispatch scope. Other terminal
jobs receive `NotRunnable`, with no heartbeat, admission or dispatch authority.

Store completion consumes that claim. In one registry actor mutation, it checks
the generation, closes unfinished files, derives the terminal status and
produces persistence/notification effects. A superseded claim produces no file
updates. There is no async gap between generation admission and cleanup in
which a restart can replace the files. Cancellation closes unfinished files
with category `cancelled`; unexplained unfinished work remains a system
failure. Any failed file makes an otherwise active job failed, including partial
success. Already accepted cancellation or shutdown keeps its status and time.
`FinishingRun` retains exclusive ownership until the admitted persistence
effects finish. Lease clearing is persisted before consuming the claim's
release, so it cannot erase a replacement runner's lease. The next runner
still has to admit the resulting status. File/attempt persistence and
progress remain separate store operations, not a database transaction with
the entire run. Host-memory polling is async: dropping the wait cannot leave a
blocking retry loop behind. One already-started ledger transaction may finish;
an unreceived plan drops its lease and cannot start a worker.

Every child file task inherits the same scope, because `tokio::spawn` does not
inherit task-locals. The scope carries two facts into each task:

- the job id, re-established as the pool's `CURRENT_JOB_ID`, so every worker
  dispatch registers against the job;
- the job's cancellation token, which the supervisor races against the task.
  On cancel the task's future is dropped wherever it is (a worker dispatch, a
  cache wait, between stages) and the file is recorded with the typed outcome
  `FileTaskOutcome::StoppedByCancellation`, category `cancelled`. Nothing after
  that point runs: no further inference, forced alignment, output write or
  debug dump belongs to a cancelled job. A checked-out worker dropped with a
  request in flight is discarded, not returned to the pool.

```mermaid
flowchart TD
    begin["begin_runner: atomic claim and snapshot"] --> claim["RunnerClaim: one owner"]
    claim --> scope["Consume into RunScope"]
    scope --> admission["Await job slot and host-memory lease"]
    admission --> probe["Probe capabilities"]
    probe --> warmup["Pre-scale workers"]
    warmup --> files["File tasks inherit RunContext"]
    files --> ended["EndedDispatch: claim and outcome"]
    scope -. "cancel during any wait" .-> stop["Drop admission, dispatch and pending startup"]
    stop --> ended
    ended --> finish["Consume claim in store completion"]
    ended -->|deferred| defer["Consume claim before scheduling replacement"]
    finish --> admit{"Current generation?"}
    admit -->|yes| close["Atomic file cleanup and derived job status"]
    admit -->|no| stale["Superseded: no file updates"]
    close --> effects["Persist and notify admitted effects"]
    effects --> release["Consume claim and release exclusive ownership"]
    stale --> release
    begin -->|cancelled before claim| stopped["StoppedRun: completion only"]
    stopped --> finish
    begin -->|other terminal job| none["NotRunnable: no execution owner"]
```

The model-free HTTP fixture in `tests/restart_handoff/live_probe.rs` drives a
failed run through restart, advancing capability loading or worker pre-scaling,
and cancellation. It admits the target boundary only while every file remains
queued. The pre-scaling case previously left the runner and queued files open
after cancellation; a file-only cancellation test could not detect that gap.
The peer performs no recognition, downloads or model loading.

`ServerBackend::cancel_job` detaches the job's registered workers from the
tracker BEFORE publishing the cancel and terminates them after, because the
publish drops the tasks and a dropped dispatch unregisters its worker as it
unwinds; a drain after the publish could miss a shared worker still computing
for the job.

Before this, neither fact crossed the spawn. A cancelled `align` job's
in-flight files ran to completion, each
finishing its Whisper pass, dumping UTR debug data, running forced alignment
and holding the worker the live job also needed, because no dispatch had
registered for the cancel's worker kill to find and nothing told the tasks to
stop. Contract test: `tests/gpu_concurrent_dispatch/cancellation.rs`, which
cancels a real transcribe job over HTTP while an echo worker holds its file.

### Cancelled vs Interrupted at shutdown

UTR's full-file and partial-window paths share an owned CPU-work boundary.
`PendingUtr` borrows the source exclusively and sends a private candidate to
the tracing-aware blocking pool. Only a completed candidate and its matching
`UtrResult` can be installed, with no intervening await. Dropping the pending
future aborts queued work and signals started work to stop at cooperative search
checkpoints. Neither cancellation nor a task panic installs candidate mutations.
This boundary retains one additional document clone while work is pending so
the source remains available unchanged if its candidate is abandoned. Provider
tokens move into the task rather than being cloned.
Cancellation is not recorded as unmatched lexical content or a search-budget
refusal; task panics are internal system failures, not invalid CHAT.

The pure alignment algorithms accept an explicit `WorkControl`, independent of
Tokio. Interleaved forward/backward/traceback passes check between reference rows;
Hirschberg checks rows and 1,024-cell blocks, and correspondence admission checks
candidate construction and ranking. Synchronous APIs use an uninhabited
interruption error, retaining their previous signatures and matching policy.
This is cooperative cancellation, not preemption: bounded row work, allocation,
document cloning and non-search traversals may finish before a stop is observed.
No search budget is raised and no timing-recovery claim follows from scheduling
alone. Tests use explicit channels/checkpoint counts, not elapsed-time thresholds.

`JobStatus::Cancelled` is reserved for user gestures (TUI cancel, HTTP
DELETE/cancel). It is permanent: a Cancelled job is never auto-resumed.

`JobStatus::Interrupted` is the system-initiated counterpart. The graceful
shutdown path writes `Interrupted` (not `Cancelled`) for in-flight jobs, with
an audit row in the `cancellations` table tagged `source=signal,
reason=server-cancel-all`. On the next server start, the recovery sequence
above transitions any Interrupted job whose file work is not yet complete back
to `Queued`.

Shutdown also supersedes the current run generation while its old runner keeps
teardown ownership until it returns. Its completion therefore releases the
claim without turning resumable files into terminal cancellation errors. The
shutdown token stops execution; the accepted `Interrupted` state determines
recovery. A user cancellation still closes unfinished files permanently.

This matters because a server bounce mid-job (deploy, OS restart, crash)
would otherwise be indistinguishable from a user cancel in the local DB, and
the user's dashboard would show the job as "cancelled" even though no user
pressed cancel.

The runtime supervisor waits for aborted job tasks to retire before returning
its shutdown summary. File tasks use abort-on-drop ownership: dropping their
parent's supervision cannot detach inference that keeps writing file failures
after shutdown starts. GPU transport admission reports `PoolShuttingDown`
during teardown, rather than misclassifying it as a protocol violation.
Regression tests cover both task retirement after a shutdown timeout and
file-task cancellation when supervision is dropped.
