# Performance and Re-Architecture Backlog

**Status:** Current
**Last modified:** 2026-03-21 07:16 EDT

This is a running developer-facing backlog for opportunities spotted during
architecture work, code audits, and routine refactors.

It is intentionally different from the architecture audit:

- the [Architecture Audit](architecture-audit.md) tracks the current structure
  and the active refactor fronts
- this page records **promising future work** that is not yet implemented
- ideas should stay here only if they still matter after the current code
  settles; stale items should be deleted rather than preserved for history

The current goal is to keep this page focused on opportunities that could
materially improve throughput, memory use, latency, or architectural clarity.

The worker-boundary replacement itself now has a dedicated implementation spec
in [Worker Protocol V2](worker-protocol-v2.md). This page should track only
follow-on opportunities that remain once that migration is underway.
The Python cutover is complete. Live audio tasks and text-only NLP tasks now
use the typed V2 worker protocol; the remaining backlog is about tightening the
last compatibility seams and improving throughput on top of that boundary.

## 2026-03-20 Review Additions

These items were promoted during the pre-push review and history cleanup.

### A. Split worker supervision from worker selection

The new TCP/persistent-worker path is a real improvement, but pool lifecycle,
GPU sharing, target resolution, and stale-process cleanup still live too close
to each other.

Potential directions:

- split worker target planning from process lifecycle ownership
- make stale/crashed-worker cleanup one explicit subsystem with narrow inputs
- keep per-resource-class scheduling separate from daemon bootstrap/reap logic

### B. Remove the remaining stringly typed configuration seams

The newtype push landed in important server paths, but several config and
request seams still accept strings or loosely named fields before converting
them deeper in the stack.

Potential directions:

- convert command/runtime identifiers at boundary entry instead of in interior
  helpers
- keep language, worker-target, and artifact identifiers typed end to end
- audit display labels versus semantic stage identifiers so operators see
  strings but internals do not depend on them

### C. Tier the ML-heavy test matrix explicitly

Coverage is better, but the suite now mixes fast structural checks with slow
runtime-heavy golden verification.

Potential directions:

- define explicit fast, medium, and heavy verification tiers
- make ownership clear for snapshot/golden regeneration workflows
- isolate environment-sensitive suites so concurrency and timeout issues are
  diagnosed as harness problems rather than product regressions

### D. Collapse tooling sprawl into explicit layers

Developer automation is still spread across `Makefile`, shell scripts, Python
 helpers, and a few Rust bins. The first `xtask` slice now exists, but the
 broader tooling shape is still transitional.

Potential directions:

- use `xtask` for typed repo-local orchestration and affected-check policy
- keep shell/Python/Node only where the external ecosystem is the real owner
- promote long-lived semantic generators/analyzers into first-class Rust tool
  crates instead of ad hoc scripts
- delete duplicate task implementations so one tool path remains the source of
  truth

### E. Narrow macro usage to stable boilerplate only

The codebase has good macro uses and bad macro uses. The good uses compress
 stable mechanical boilerplate. The bad uses blur domain decisions that should
 stay explicit in normal Rust types.

Potential directions:

- keep one canonical macro source for shared newtypes instead of duplicating
  macro layers inside app crates
- keep semantic proc macros where they encode real model invariants
- avoid macro-generated wrappers for values that require validation or
  sentinel semantics
- audit blanket trait generation such as permissive `Default`, broad `From`,
  and implicit coercions on newtypes

### F. Keep shrinking the PyO3 native boundary

The `pyo3` crate is thinner than before, but it is still too close to being an
application boundary instead of a native adapter. One real cut already landed:
`extract_timed_tiers` no longer pulls in `talkbank-transform`; it now uses
`talkbank-parser` plus `talkbank-model` validation directly. The remaining work
is still substantial.

Potential directions:

- keep removing crates that `pyo3` only touches through one-off helpers
- split worker-execution helpers from general CHAT/document APIs so Python
  bindings do not need to compile every runtime surface together
- avoid growing new workflow logic inside `pyo3`; push that logic down into
  shared Rust crates first and bind the result second
- treat `pyo3` feature flags as architectural boundaries, not as a dumping
  ground for unrelated optional code
- keep Python extraction wrappers and other FFI-only adapters out of
  `talkbank-model` and other shared domain crates

Recent progress:

- the embedded `cli_entry` path no longer compiles the standalone binary's
  OTLP tracing stack or background update-check wiring; `batchalign-cli`
  now exposes those only behind its `binary-entry` feature, and the PyO3
  dependency path disables default features

## How to use this page

- Add items when a refactor reveals a larger optimization or redesign
  opportunity that is real but not part of the current patch.
- Prefer concrete opportunities over vague aspirations.
- If an item becomes active work, move the details into the relevant
  architecture page or PR notes and remove or simplify the item here.
- If an item becomes stale, delete it rather than letting this page turn into a
  historical planning archive.

## High-Value Current Opportunities

### 1. Push the remaining ASR provider-local preprocessing into Rust

The biggest response-shaping step is now done: Python ASR returns typed V2 raw
payloads and Rust owns the shared normalization layer. FunASR segment cleanup
plus Tencent and Aliyun projection now also live in Rust helpers. The
remaining Python surface is smaller, and all Python-hosted ASR now runs
through the live V2 boundary. What is left is mostly provider-local SDK
transport, FA audio transport redesign, and a few remaining runtime quirks
like NeMo.

Potential directions:

- reduce Python-side audio/request preparation when the same work can happen
  in Rust before the worker call
- centralize provider retry/backoff and normalization policy in one Rust
  boundary
- remove the remaining model-runtime monkeypatches entirely; the Whisper
  timestamp workaround is instance-local now, but NeMo speaker diarization
  still relies on a global class override

Why it matters:

- provider transport is not document orchestration
- it keeps Python closer to pure SDK/model hosting
- it reduces duplicated provider-shaping rules across worker paths

### 1a. Reintroduce batching on top of live worker protocol V2

Forced alignment now runs through the live typed `execute_v2` path, which is
the right architecture move because Rust owns prepared artifacts and result
normalization. The current live V2 FA transport still dispatches one execute
request per miss group, though, because the new protocol shape is strictly
single-request today.

Potential directions:

- add a batched or multiplexed V2 execute envelope once the single-request
  path has soaked
- let one worker call process several prepared-artifact requests in order
  without rebuilding the old generic batch-infer bag
- measure whether grouping by file or by backend profile gives better reuse

Why it matters:

- it can recover throughput that the old FA batch path got from one IPC roundtrip
- it keeps the typed boundary without forcing the worker loop back toward an
  untyped generic batch shape

### 2. Push more Rev.AI shaping into Rust

Rev.AI transport is now shared Rust code, server-side preflight is Rust owned,
and server-mode Rev.AI ASR plus Rev-backed UTR no longer route through the
Python worker. The remaining possible step is to move any leftover direct-
Python Rev.AI shaping behind the same Rust boundary so the Python package only
exposes a thin convenience wrapper.

Potential directions:

- centralize Rev.AI retry/backoff policy in one Rust boundary
- remove any remaining duplicate transcript-to-token shaping between direct
  Python workflows and server mode
- move the remaining UTR / timed-word Rev path into the same Rust client so
  Python no longer needs to host Rev convenience transport in production

Why it matters:

- Rev.AI is not a Python-only SDK problem
- it shrinks Python logic that is not model execution
- it reduces duplicated provider-shaping rules

### 2a. Stop masking per-file task failures behind terminal cleanup

The runner now has a supervised file-task boundary, so panics and early
non-terminal exits are recorded as explicit file failures instead of being
discovered only by the final cleanup sweep. The first durable file attempt
also begins before per-file setup, so missing-input and conversion failures are
now visible in attempt history. The runner also has a `FileRunTracker`
boundary, so per-file dispatch paths no longer hand-sequence the common
processing/retry/success mutations. The remaining step is to tighten that into
a smaller reducer/coordinator model so file setup, retry, progress, and
cancellation all flow through one owned state machine. Supervised file tasks
now report explicit `FileTaskOutcome` values to the runner instead of relying
on a post-task shared-state read to infer whether terminal state was recorded.
Preflight setup failures also use the same lifecycle helper now, which narrows
the remaining gap to a true file-stage reducer.

The runner-owned stage vocabulary is now typed through `FileStage`. The next
obvious cleanup used to be the lower-level progress channel payloads coming
from FA and transcribe internals; those now use the same typed `FileStage`
vocabulary too. The API now also exposes a parallel typed `progress_stage`
field for richer clients, with `progress_label` retained as a derived
operator-facing display string.

Potential directions:

- move per-file processing onto a typed reducer or actor boundary
- decide whether the new `file_setup` work unit should stay distinct or become
  part of a broader typed file-stage reducer
- remove or further narrow the final fallback sweep once supervision and
  earlier attempt boundaries have proved sufficient in production

Why it matters:

- race conditions and panics become diagnosable instead of looking identical
- dashboard state stays causally correct
- it removes a class of "runner succeeded at cleanup but failed at truth"
  bugs

### 3. Split worker pools by resource class, not only command/lang

The current worker pool keying is already better than ad hoc process spawning,
and the infer path now uses explicit task targets such as `infer:asr` instead
of overloading top-level command names. There is still room for a more
explicit runtime scheduler.

Potential directions:

- separate pools or semaphores for GPU-heavy, CPU-heavy, and provider-network
  work
- treat network-bound provider work differently from local model inference
- tune `ThreadPoolExecutor` vs process workers vs Rust async tasks by workload
  class instead of by command name alone

Why it matters:

- prevents network-bound work from looking like model-bound work
- reduces accidental contention between unrelated resource classes
- could improve warmup and scaling policy for fleet mode later

### 3a. Separate worker warmup from command advertisement even further

The latest control-plane pass made `transcribe`, `transcribe_s`, and
`benchmark` server-composed commands synthesized from ASR capability. The next
step could go further and make warmup/profile selection operate directly on
typed worker targets or resource classes instead of reusing command names.

Why it matters:

- warmup intent becomes clearer than "spawn whatever this command used to mean"
- the control plane can reason about `infer:asr` versus `infer:fa` explicitly
- it prepares the scheduler for fleet mode and resource-class-aware routing

### 4. Overlap more job stages with pipelining

Several command paths still process one file as a mostly linear chain even
though parts of the work are naturally pipelineable.

Potential directions:

- overlap media conversion, upload/preflight, and worker inference setup
- stream progress updates as stage completions instead of only file-state jumps
- precompute immutable per-job artifacts while earlier files are still running

Why it matters:

- lower wall-clock time for large batches
- better perceived responsiveness in the dashboard
- cleaner separation between I/O stages and inference stages

### 5. Extend incremental processing where it actually pays off

`batchalign3` already has some partial incremental ideas, especially around FA
and morphosyntax. Those should be treated as a real architecture seam rather
than a side optimization.

Potential directions:

- define explicit per-command change models: what invalidates what
- reuse stable utterance/file hashes more aggressively across retries and reruns
- make incremental eligibility visible in server diagnostics and UI
- explore partial retranscription or partial reinjection for large CHAT files

Current concrete progress:

- `align` already has a whole-file reusable `%wor` fast path
- `align --before` now preserves stable `%wor` timing for unchanged utterances
  before grouping, so preserved regions can skip both cache misses and worker
  FA

Why it matters:

- repeated corpus work is common
- incremental wins compound with caching and batching
- the logic belongs in Rust control-plane/document code, not Python

### 5a. Add stage-by-stage alignment trace artifacts, not only final debug dumps

UTR already has a useful offline reproduction hook: when
`$BATCHALIGN_DEBUG_DIR` is set, it writes the pre-injection CHAT plus the ASR
timing tokens consumed by `inject_utr_timing()`. That is enough to reproduce
the 407 token-starvation regression offline. It is not enough to debug the
full class of intricate pre/post-processing failures, though, because it omits
most of the intermediate state that explains *why* a file failed.

Potential directions:

- capture normalized transcript word lists, normalized ASR token lists, and
  the final DP match pairs for UTR
- capture interpolation windows, FA group boundaries, and postprocess /
  monotonicity stripping decisions
- make those artifacts replayable from one command or dashboard trace view
- keep the artifacts typed and file-scoped so they can be diffed across runs

Why it matters:

- the hardest alignment failures are pipeline interactions, not single-step bugs
- debugging needs causality, not only final output files
- richer trace artifacts would make complex alignment failure reports much
  easier to reproduce and minimize

### 5b. Explore adaptive "trouble-window" alignment above the global baseline

This idea now has a dedicated plan page in
[Trouble-Window Alignment Plan](trouble-window-alignment.md).

The current UTR correctness baseline is a single global Hirschberg alignment
across the whole file. That is the safest steady-state model for the
token-starvation class, because it gives one globally consistent answer. A
future redesign could still be more selective *if* it preserves that global
correctness boundary.

Potential directions:

- use a cheap first pass to find anchor regions with strong agreement
- identify divergence regions ("trouble windows") where coverage, ordering, or
  local confidence collapses
- run heavier or more permissive alignment only inside those windows, while
  keeping the rest of the file on the cheaper anchored path
- make the window finder itself testable and replayable so it does not become a
  hidden correctness cliff

Why it matters:

- it could reduce the cost of whole-file DP on very large files
- it may create space for richer overlap-aware handling without widening every
  normal file into the most expensive path
- it is only worth doing if the anchor/window detector is as trustworthy as the
  current all-file baseline

### 5c. Generalize feedback and retry loops beyond `align`

`align` is not the only command whose pipeline could benefit from a structural
feedback loop. Any command that can measure output health in domain terms could
use retries, alternate repair passes, or narrower fallback strategies instead
of a single straight-line pipeline.

Potential directions:

- retry morphosyntax when token-count or attachment invariants fail after
  retokenization
- retry utterance segmentation when assignment vectors or sentence boundaries
  become invalid
- retry translation or tier injection when payload shape is valid but the tier
  cannot be injected cleanly into the AST
- make command-specific health metrics explicit so retries are data-driven
  rather than ad hoc

Why it matters:

- a typed repair loop is often safer than silently accepting low-quality output
- the control plane can expose clearer diagnostics to operators
- this pushes the system toward resilient orchestrated pipelines rather than
  brittle one-shot transforms

## Control-Plane Opportunities

### Job registry as an actor, not only a guarded map

The `JobRegistry` refactor already improved ownership. A later step could move
the central control-plane mutation path to an actor/reducer model.

Potential directions:

- one owned registry task receiving typed transition commands
- append-only event stream for important lifecycle changes
- clearer write/read projection split for dashboard state

Tradeoff:

- higher implementation cost
- more ceremony around state transitions
- potentially much clearer fleet-readiness and replay/debuggability

### Cache and manifest precomputation

Some work still happens lazily inside hot paths that could move to submission
time or worker warmup time.

Potential directions:

- precompute per-job immutable dispatch summaries
- cache more derived file metadata once at submission
- move repeated config parsing or capability normalization out of per-worker
  spawn paths

## Python Boundary Opportunities

### Remove Python-side provider shaping where Rust can own it

The target state is that Python performs only the irreducible provider/model
call. Everything else should be considered movable by default.

Candidates:

- raw transcript/token shaping
- provider-specific response normalization
- request envelope preparation that does not require a Python SDK

### Keep standalone Python helpers small and optional

Modules like `inference/benchmark.py` are acceptable only as very thin Python
package conveniences. They should never quietly become part of the worker
runtime contract again.

## Frontend and Desktop Opportunities

### Reduce duplicated dashboard state movement

The dashboard is already cleaner, but there is still room to collapse repeated
state propagation between WebSocket events, React Query caches, and view-local
state.

Potential directions:

- move more list/detail recomputation to server-provided projections
- virtualize large tables earlier
- pre-aggregate progress summaries on the server when job counts grow

### Revive the desktop shell as a thin local host

Before public release, `dashboard-desktop` should return as a deliberately thin
shell around the web UI.

Minimum expectations:

- choose input files or folders
- target a local daemon or configured server
- show job state and progress
- avoid duplicating web-side orchestration logic

## Cross-Repo Opportunities

### Tighten `talkbank-tools` seams for `batchalign3`

Ergonomic cross-repo changes are worthwhile when they simplify the
architecture.

Potential directions:

- lower-allocation AST extraction helpers for batch job pipelines
- better zero-copy or streaming parse/serialize seams for large corpora
- more explicit Rust APIs for incremental document updates that `batchalign3`
  can call directly

## What to avoid

- adding Python orchestration just because it is convenient in the moment
- introducing generic async complexity where owned blocking workers or channels
  are clearer
- keeping legacy architectural compromises merely for internal compatibility
