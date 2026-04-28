# Chapter 8: Control Plane and State Management

## The Control Plane Is Carrying Too Much Coupled Logic

The server-side control plane has many strong ideas:

- job store
- runner
- queueing
- persistence
- websocket/eventing
- staged remote execution

But those ideas still meet each other inside oversized coordination modules.

Relevant file size signals:

- `runner/mod.rs`: 1107 lines
- `runner/util/file_status.rs`: 1395 lines
- `store/job/mod.rs`: 1040 lines
- `types/config.rs`: 999 lines

These files are not just "big." They are mixing lifecycle, policy, translation,
status shaping, and persistence concerns.

## Error and State Surfaces Still Sometimes Degrade Silently

One example:

- `staging/orchestrator.rs:162-175`

On remote submit failure, the code falls back to `submit_resp.text().await.unwrap_or_default()`,
which can erase useful evidence in exactly the cases where operators need it.

This is not catastrophic alone. It is emblematic of a broader issue: the
control plane still contains several "best effort" diagnostic behaviors where
release-grade software should preserve evidence.

## Job State Should Become More Explicitly Reducer- or Event-Driven

The job system today still reads like a set of coordinated mutable structures
that happen to remain mostly disciplined. That is better than chaos, but worse
than an explicit state transition model.

For a 15-year system, job lifecycle code should be auditable in terms of:

- allowed states
- allowed transitions
- durable evidence
- operator-facing truth
- recovery semantics

Not just "the current mutating methods seem to cooperate."

## Recommended Direction

This is a good place for targeted redesign, not total rewrite.

Specifically:

- define state transitions more explicitly
- make persistence and in-memory views derive from the same lifecycle model
- isolate operator-facing projections from mutable storage internals
- preserve failure evidence rather than normalizing it away

## Action Items

- Break the runner into planning, execution, persistence, and operator-notify layers.
- Reduce giant status-tracking modules into smaller reducers around file/job state.
- Audit every `unwrap_or_default`, `.ok()`, and similar silent degradation path
  in production control-plane code.
- Make remote/staged execution retain and expose full failure evidence.
- Add restart/recovery tests that validate evidence preservation, not just
  nominal state transitions.
