# Server Mode

**Status:** Current
**Last updated:** 2026-04-06 12:51 EDT

Batchalign includes a built-in HTTP server managed by `batchalign3 serve ...`.
Ordinary local processing commands can still run inline, but the CLI no longer
treats the direct host as the only default. When `auto_daemon: true` (the
default), the router first tries to reuse or start a loopback daemon so warm
workers survive across commands. `--no-server` and `--sequential` still force
direct local execution.

## Current routing rules

- With `--server URL`, the CLI submits supported jobs to that server in content
  mode.
- `transcribe`, `transcribe_s`, `benchmark`, and `avqi` prefer the local daemon
  when `auto_daemon` is enabled. In that case the CLI tries to reuse or start a
  loopback daemon first; if that local daemon path is unavailable, it falls back
  to the explicit `--server` target.
- Without an explicit remote target, `auto_daemon: true` makes the CLI reuse or
  start a loopback daemon before it falls back to the shared direct host.
- If daemon startup is unavailable, the CLI still reuses any already-running
  loopback server on the configured port before it falls back to direct local
  execution.
- Local-daemon and auto-detected loopback-server paths use shared-filesystem
  `paths_mode` for local-audio commands such as `align`, `transcribe`,
  `benchmark`, `opensmile`, and `avqi`. Explicit `--server` always stays on
  content mode, even when the URL is `localhost`.
- This matters most on Apple CPU-only hosts: repeated `align` / `transcribe` /
  `benchmark` runs are much faster when warm workers are preserved behind a
  loopback daemon than when each command pays the full cold direct-local cost.

## Start a server

Foreground:

```bash
batchalign3 serve start --foreground
```

Background:

```bash
batchalign3 serve start
```

Useful flags:

```bash
batchalign3 serve start --foreground --port 8000
batchalign3 serve start --foreground --config ~/server.yaml
batchalign3 serve start --foreground --warmup minimal
batchalign3 serve start --foreground --test-echo
batchalign3 serve start --foreground --backend temporal --test-echo --warmup off
```

## Backend selection

`batchalign3 serve start` now accepts `--backend embedded|temporal`.

- `embedded` remains the default and is still the recommended local/single-host
  backend.
- `temporal` is an experimental clean-slate control-plane backend. It keeps the
  shared Batchalign execution engine and worker pool, but hands queued-job
  orchestration, retry timing, and durable cancellation/restart state to
  Temporal workflows and activities.

This backend choice only affects explicit server mode. Ordinary local CLI use
can still reuse/start a loopback daemon or fall back to the direct host, and it
does not require any Temporal service.

To try the Temporal backend locally, start a Temporal dev server first:

```bash
temporal server start-dev
batchalign3 serve start --foreground --backend temporal --test-echo --warmup off
```

## Check and stop a server

```bash
batchalign3 serve status
batchalign3 serve status --server http://myserver:8000
batchalign3 serve stop
```

Inspect remote jobs:

```bash
batchalign3 jobs --server http://myserver:8000
batchalign3 jobs --server http://myserver:8000 <JOB_ID>
```

When that server is running with `--backend temporal`,
`batchalign3 jobs --server ... <JOB_ID>` also reports the Temporal workflow ID,
run ID, workflow status, task queue, and history length for that Batchalign
job.

Important `--test-echo` caveat:

- `--test-echo` is a control-plane smoke path, not a full model simulation.
- Text-only infer-task commands such as `morphotag`, `utseg`, `translate`,
  `coref`, and `compare` are expected to fail under `--test-echo` because the
  echo worker does not advertise real `infer_tasks`.
- Use it to validate startup, submission, restart, cancellation, deletion, and
  remote job inspection. Do not treat it as proof that infer-task commands are
  semantically correct.

## Server configuration

Default config path:

```text
~/.batchalign3/server.yaml
```

Minimal example:

```yaml
default_lang: eng
port: 8000
max_concurrent_jobs: 8
auto_daemon: true
warmup_commands: [morphotag, align, transcribe]
media_roots: []
media_mappings: {}
```

Temporal-specific config keys:

```yaml
backend: temporal
temporal_server_url: http://127.0.0.1:7233
temporal_namespace: default
temporal_task_queue: batchalign3-server
temporal_heartbeat_s: 5
temporal_activity_timeout_s: 3600
```

`warmup_commands` now marks commands that are *eligible* for warmup. The
current production startup path remains lazy by default, so this key is no
longer a promise that those workers will preload on every boot.

When warmup does spawn TCP daemons, they are now treated as **server-owned**
workers: reusable for that server instance, but cleaned up on routine shutdown.
If you want a daemon to survive server restarts, start it externally and let the
server discover it from `workers.json`. Direct local execution does not perform
that registry discovery step.

## Cold-start capability checks

On a cold server, especially with `--warmup off`, the startup path may only know
an **optimistic** command list before any real worker has been spawned. That is
intentional: startup no longer pays for a dedicated probe worker just to fill in
capability metadata.

What matters operationally is that execution now does a live check before it
trusts infer-task gating. The first real `morphotag`, `align`, `compare`, or
similar job boots the needed worker, probes its actual infer-task support, and
then runs. If the backend is truly unavailable, the job now fails with the
worker/bootstrap error instead of with a stale placeholder `infer_tasks: []`
message.

If startup finds healthy registry daemons that are already running, the server
now seeds capability state from those live workers immediately. That means a
server can come up with a real `/health.capabilities` surface without spawning a
fresh probe worker, as long as discoverable daemons already exist.

## Registry daemon ownership

`~/.batchalign3/workers.json` can now contain two daemon kinds:

- **external** daemons, started outside the current server lifecycle
- **server-owned** daemons, started by the current Rust server instance

Routine shutdown only kills the current server's own server-owned daemons.
External daemons are preserved and rediscovered on the next startup.

At startup, registry discovery reuses healthy external daemons, preserves live
foreign server-owned daemons by skipping them, and reaps stale foreign
server-owned daemons whose owning server is gone.

Important keys:

- `port` — server listen port
- `host` — bind address (defaults to `0.0.0.0`)
- `backend` — server control-plane backend: `embedded` (default) or experimental `temporal`
- `max_concurrent_jobs` — `0` means auto-tune
- `auto_daemon` — reuse or start a loopback daemon for ordinary CLI processing
- `warmup_commands` — list of commands eligible for warmup (see [Worker Tuning](worker-tuning.md))
- `media_roots` — local execution-host media lookup roots
- `media_mappings` — local execution-host root mappings from corpus paths to
  mounted media paths; useful when the CHAT/data clone root differs from the
  media root on the same machine
- `temporal_server_url` / `temporal_namespace` / `temporal_task_queue` — Temporal connection settings used only when `backend: temporal`
- `temporal_heartbeat_s` / `temporal_activity_timeout_s` — Temporal activity heartbeat and per-attempt timeout controls
- `memory_tier` — override auto-detected tier: `small`, `medium`, `large`, `fleet` (also controls task-vs-profile bootstrap mode)
- `memory_gate_mb` — host headroom reserve (default: tier-dependent, 2000-8000 MB)
- `gpu_startup_mb` / `stanza_startup_mb` / `io_startup_mb` — per-profile startup reservation overrides (0 = tier default)
- `worker_idle_timeout_s` — shut down idle workers after this many seconds (default: tier-dependent — 60s Small, 300s Medium, 600s Large/Fleet)
- `worker_health_interval_s` — health check frequency in seconds (default: 30)
- `job_ttl_days` — auto-delete completed jobs after this many days (default: 7)

OTLP tracing can be enabled by setting `BATCHALIGN_OTLP_ENDPOINT`
(or `OTEL_EXPORTER_OTLP_ENDPOINT`) in the server environment.

Reference example files live in `examples/server.yaml` and
`examples/launchd.plist`.

`server.yaml` uses a strict schema. Unknown keys are rejected at startup
instead of being silently ignored, so stale config like `warmup: false` must
be updated to the current `warmup_commands: []` form.

## Remote use

Commands that support explicit remote dispatch look like this:

```bash
batchalign3 --server http://myserver:8000 morphotag corpus/ -o output/
batchalign3 --server http://myserver:8000 align corpus/ -o output/
```

For audio commands, `--server` now means "run this on a host that can already
see these filesystem paths." The clean operational model is to run the CLI on
the execution host itself (or to reach it over SSH/VNC) rather than expecting
the server to infer media from a different client machine's directory layout.
When the corpus clone root and the mounted media root differ on that execution
host, use local `media_mappings` or `--media-dir` as explicit root replacement.

Health checks:

```bash
curl -s http://myserver:8000/health | python3 -m json.tool
batchalign3 serve status --server http://myserver:8000
```

The `/health` response includes a `capabilities` list. On a warm server, or on a
startup that already discovered live registry daemons, treat that list as the
detected command surface. On a cold lazy server with no discovered workers, it
may still be the optimistic startup surface until the first live worker probe
completes.

If a command is missing from `/health` **after** the server has warmed or run a
real job for that family, the server's Python environment is likely missing a
required package. See
[Troubleshooting: "Command not supported"](troubleshooting.md#command-not-supported-or-missing-commands).

## launchd example (macOS)

For always-on macOS hosts, use `examples/launchd.plist` as a template and
update the binary path, username, and log paths before installing it as a
LaunchDaemon.
