# Server Known Issues

**Status:** Current open issues only
**Last modified:** 2026-04-14 19:20 EDT
**Last verified:** 2026-04-14

## Morphosyntax silent-corruption on pool saturation (resolved)

**Severity (historical):** Critical — affected CHAT files could ship with
stripped `%mor`/`%gra` tiers and `rc=0` status.

**Symptom (pre-fix):** On a large multi-language morphotag batch the pool
hit its global worker cap while early-language workers were still counted
(idle between batches). Later language groups logged
`worker error: ... cannot wait (would deadlock)` and returned empty
`UdResponse { sentences: [] }`. The orchestrator soft-failed those groups,
injection wrote no `%mor`/`%gra` onto the affected utterances, and the
empty-placeholder sweep stripped the pre-cleared tiers entirely. Files
serialized with `rc=0`.

**Fix landed 2026-04-14:** Two-layer safeguards in
[Worker-Pool Saturation and Morphosyntax Safeguards](worker-pool-saturation.md).
Idle-eviction + bounded wait in `WorkerPool::checkout` prevents the
saturation bailout in the common case, and typed language-group failure
propagation in `morphosyntax/outcomes.rs` converts every residual pool
failure into per-file errors the CLI surfaces with a non-zero exit code.
No code path now writes a file with a stripped tier.

Detection tooling: `scripts/analysis/morphotag_bailout_audit.py` in the
private workspace walks every data repo's recent diffs and flags any
`.cha` file whose `%mor`/`%gra` tier-line count went net-negative.

## Zombie job resurrection on server restart

**Severity:** High — blocks new job submissions via 409 conflict.

**Symptom:** Cancelled or stuck jobs reappear as "running" after every
server restart. They consume workers and block new submissions for the
same files (409 conflict detection).

**Root cause:** SQLite job store persists jobs with `status=running`.
On startup, recovery re-queues any job that was running when the server
stopped. Cancelled jobs that were stuck in synchronous code (e.g., the
injection loop) never transitioned to `cancelled` in SQLite because the
cancellation token is only checked at async yield points.

**Workaround:** Cancel zombie jobs after restart:
```bash
curl -s http://localhost:8001/jobs | python3 -c "
import json,sys
for j in json.load(sys.stdin):
    if j['status'] in ('running','queued'):
        print(j['job_id'])" | while read jid; do
  curl -s -X POST "http://localhost:8001/jobs/$jid/cancel"
done
```
Or delete `~/.batchalign3/jobs.db` before restart (loses all job history).

**Fix needed:**
1. Recovery should not re-queue jobs that were cancelled
2. Consider `max_recovery_age` — don't recover jobs older than N minutes
3. Cooperative cancellation in synchronous loops

**Observed:** 2026-03-29, Net + Bilbo with both Temporal and embedded
backends. Jobs resurrected across 3+ server restarts.

**Partial fix landed 2026-04-14 (Temporal backend only):** the Temporal
reconciler (`crates/batchalign-app/src/temporal_reconciler.rs`) now sweeps
stale local-store entries against Temporal's authoritative workflow status
on a 30 s tick and opportunistically on every `submit_job` (scoped to the
submitter). This closes the specific subclass of "409 Conflict on resubmit
because a peer-completed workflow left a ghost `Queued` job behind in the
submitter's local cache." Embedded/direct backends are unaffected — for
those, the local store is already authoritative and the recovery path
described above is the correct fix. See
[Observability: Job status authority](observability.md#job-status-authority-local-store-vs-temporal-2026-04-14)
for the full protocol.

## Transient accept-gap at job finalization

**Severity:** Medium — previously silent work-loss; now masked by CLI retry.

**Symptom:** A `POST /jobs` request issued while the daemon is finalizing a
previous job receives a connection-refused at the TCP layer. Scripts that
treat submission errors as terminal silently skip the chunk. On 2026-04-14
three submitted chunks (roughly 1,500 files) were lost this way before any
server-side work happened.

**Root cause:** Finalizing a job in
`crates/batchalign-app/src/store/queries/execution.rs::finalize_job`
(which calls `db_update_job`) holds the SQLite WAL checkpoint briefly, and
the tokio scheduler on net routinely interleaves a ~1.3 s gap during which
the axum listener does not accept new connections. The accept-gap is
structural — a consequence of single-process tokio + SQLite WAL — and is
not being redesigned in this round. A remote client must assume the
accept-gap exists.

**Fix landed 2026-04-14 (submit side, CLI only):** `submit_job` in
`crates/batchalign-cli/src/client.rs` now routes through the existing
`request_with_retry` helper. Up to `RETRY_ATTEMPTS = 3` attempts with
exponential backoff starting at `RETRY_BACKOFF = 2.0 s` (with
`0.5×–1.5×` jitter); the helper retries **only** on
`reqwest::Error::is_connect() || is_timeout()` and **never** on
deterministic 4xx/5xx rejections. Per-attempt timeout is now a parameter —
submission passes 120 s, health and result GETs keep 30 s. See
[Submit-path retries](observability.md#submit-path-retries-2026-04-14) for
the sequence diagram and the retry-semantics contract.

**Implication for non-CLI clients.** Any caller that is not the batchalign3
CLI (custom scripts, third-party integrations against the REST API) must
implement its own retry on connect/timeout to avoid silent work loss.
Retrying on 4xx/5xx is **wrong** — those are deterministic server
rejections that will not succeed on a second try.

**No server-side fix is planned here.** The accept-gap is documented as
structural; retry makes it invisible in practice. A successor who wants to
eliminate it would need to move job finalization off the accept thread or
off SQLite WAL, both of which are out of scope for this round.

## 413 Payload Too Large on large submission chunks

**Severity:** Medium — blocked one chunk on 2026-04-14 corpus run.

**Symptom:** The CLI reports
`server returned 413: length limit exceeded`. The server's
`RequestBodyLimitLayer` (configured at
`crates/batchalign-app/src/routes/mod.rs:36,55`) rejected the submission
before any handler ran.

**Root cause:** In `paths_mode=false` submissions the CLI ships each file's
full CHAT content in the request body. For 500-file chunks of large repos
(e.g. `childes-eng-uk`, `childes-other`) a single chunk routinely tops
100 MB. The previous default `max_body_bytes_mb = 100` (in
`crates/batchalign-app/src/types/config/server.rs::default_max_body_bytes_mb`)
was smaller than the observed payload ceiling. One chunk measured 107 MB
hit this directly.

**First fix landed 2026-04-14:** the default was raised from 100 MB to
`MemoryMb(512)`. `server.yaml`'s `max_body_bytes_mb` override path is
unchanged and the schema is unchanged; deployments that need a different
ceiling keep their existing override. The regression test
`crates/batchalign-app/src/types/config/tests.rs::default_config` pins the
new default.

**Structural fix for local submissions (also 2026-04-14):** text commands
(`morphotag`, `utseg`, `translate`, `coref`, `compare`) now opt into
`paths_mode=true` when submitting to a local daemon, same as the audio
commands already did. Opted-in via a typed `CommandIoProfile` on each
command's `CommandWorkflowDescriptor`
(`crates/batchalign-app/src/commands/spec.rs`): text commands use
`PathsModeText`, audio commands use `PathsModeAudio`, and `opensmile`
stays on `ContentOnly`. Gated at the CLI in
`crates/batchalign-cli/src/dispatch/single.rs:105-107`. In paths mode the
HTTP request body carries only path lists, so
`RequestBodyLimitLayer`/`max_body_bytes_mb` does not apply — 413 on a
local submission is now structurally unreachable. See
[Submission Modes](../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse)
for the selection rule and the per-command table.

**Operator guidance.** The 512 MB default now matters only for **remote**
submissions (explicit `--server http://host:port` with a non-loopback
host). Deployments expecting remote payloads above 512 MB (very large
corpora, or very many files per chunk) should set `max_body_bytes_mb`
explicitly in `~/.batchalign3/server.yaml`. There is no auto-resizing
based on observed traffic. Local submissions are unaffected regardless of
this setting.

## FA crash on audio timestamps past end of file (fixed 2026-04-08)

**Fixed in:** `crates/batchalign-app/src/worker/artifacts_v2.rs` (empty-segment check)

When an FA group's utterance timestamps extended past the actual end of the audio
file (common in PWA corpora where end-times are hand-estimated), `ffmpeg` produced
an empty PCM file but exited with code 0.  The 0-frame PCM was passed to Wave2Vec,
which crashed with:

```
RuntimeError: Calculated padded input size per channel: (0).
Kernel size: (10). Kernel size can't be greater than actual input size
```

This surfaced as a `RuntimeFailure` inside a `failed to parse worker protocol V2
FA response for group N` error, causing the entire file's alignment to fail.

**Fix:** `extract_prepared_audio_segment_f32le` now checks `frame_count == 0`
after ffmpeg and returns `PreparedArtifactErrorV2::EmptyAudioSegment`.  The
transport layer catches this and skips the group with all-`None` timings (words
left unaligned), logging a warning.  The rest of the file aligns normally.

See [Audio timestamps past end of file](../../reference/forced-alignment.md#audio-timestamps-past-end-of-file)
for full details.

## Tracing output lost in daemon mode (fixed 2026-03-29)

**Fixed in:** commit `faaaa31b`

`serve_cmd.rs` background path used `File::create()` which truncated
the server log on every restart. Changed to `OpenOptions::append()`.

## Ansible deploy kills running jobs

**Severity:** High — production jobs interrupted by routine deploys.

The `batchalign` Ansible role runs `batchalign3 serve stop` and
`pkill -9 batchalign-worker` on ALL targeted machines. It does not
check for active jobs before stopping.

**Workaround:** Check dashboard before deploying. Use `--limit` to
target specific machines. Never deploy to machines with active jobs.

**Fix needed:** The Ansible role should check for active jobs before
stopping (the `server` role does this, but the `batchalign` client
role does not).

This page contains current open operational issues only.

## Open Issues

### 1. First-call deadlock on align (MPS from background thread)

**Symptom:** The first `align` job submitted to the server hangs indefinitely. The process shows 0% CPU, blocked on `pthread_cond_wait`. Subsequent runs after a server restart work fine because HuggingFace model weights are cached on disk.

**Root cause:** Historically, first-use MPS model initialization in background worker threads could deadlock on macOS.

**Current state:** Partially mitigated. The server now warms up key pipelines
(`morphotag`, `align`, `transcribe`) on the main thread during startup. This
reduces first-call deadlock risk significantly, but does not guarantee
elimination if warmup is disabled or a warmup load fails.

**Current mitigations:**
- Keep `warmup_commands` configured in `~/.batchalign3/server.yaml` (the
  default is `["morphotag", "align", "transcribe"]`).
- If hangs still occur on a specific machine, retry with CPU-only (`--force-cpu` for CLI workloads) or isolate affected commands.
- If this becomes frequent in production, consider process isolation for the affected command path.

**Diagnosis tool:** If the server appears hung, sample the process:
```bash
sample <pid> -mayDie
```
Look for threads blocked on `lock_PyThread_acquire_lock` / `pthread_cond_wait`.

### 2. No run logs from server jobs

**Symptom:** `~/.batchalign3/logs/run-*.jsonl` files are empty or missing for server-processed jobs.

**Root cause:** Structured run logging (`run_log.py`) is tied to CLI dispatch paths. Server execution goes through `JobStore` worker threads/processes and updates `jobs.db` / API state, but does not emit CLI-style run logs.

**Status:** Open known limitation. Server errors are logged to server stderr (for example launchd log files), and per-job/per-file status is available via API/dashboard. Structured server-side run logs are not currently implemented.

### 3. Large concurrent FA waves can mix worker-protocol collapse with SQLite write contention

**Symptom:** A large `align` job fails many files with `worker_protocol`
errors in a burst. Server logs show repeated lines like:

- `GPU worker: orphaned execute_v2 response ... request_id=fa-v2-request-0`
- `FA processing failed: worker protocol error: GPU worker response channel closed`
- `DB insert_attempt_start failed ... database is locked`

**Observed evidence:** This exact pattern was captured on a fleet machine on March 20,
2026 under `~/.batchalign3/daemon.log` for job `f0d498b5-ad1`.

**Current understanding:** Two issues can overlap here:

- old FA request IDs were not globally unique enough under shared concurrent GPU
  workers, which could produce orphaned response routing
- attempt persistence is still vulnerable to transient SQLite lock/busy failures
  during bursty per-file startup

**Current state:** Partially mitigated in the rearchitecture branch. The FA
request-correlation fix is already in that branch, and attempt-start
persistence now retries bounded SQLite lock/busy failures instead of failing on
the first transient lock. This still needs soak time before it can be
considered closed.
