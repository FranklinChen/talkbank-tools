# Server Mode

**Status:** Current
**Last updated:** 2026-09-16 09:47 EDT

Batchalign includes a built-in HTTP server managed by `batchalign3 serve ...`.
Ordinary local processing commands can still run inline, but when
`auto_daemon: true` (the default) the CLI first tries to reuse or start a
loopback daemon so warm workers survive across commands. `--no-server` and
`--sequential` still force direct local execution.

## Current routing rules

- With `--server URL`, the CLI submits supported jobs to that server in content mode.
- `transcribe`, `transcribe_s`, `benchmark`, and `avqi` prefer the local daemon when `auto_daemon` is enabled.
- Without an explicit remote target, `auto_daemon: true` makes the CLI reuse or start a loopback daemon before it falls back to direct local execution.
- Local-daemon and auto-detected loopback-server paths use shared-filesystem `paths_mode` for local-audio commands such as `align`, `transcribe`, `benchmark`, `opensmile`, and `avqi`.
- Explicit `--server` always stays on content mode, even when the URL is `localhost`.

## Build identity check

Before it submits anything, the CLI reads the server's `/health` and compares
its `build_hash` with the CLI's own build identity. This applies to every
server the CLI submits to: an explicit `--server`, the local daemon, and a
loopback server it detects. If the server reports another build, or no build
at all, the command is refused before any job is submitted, so no work runs,
and it exits with code `5` (server/job lifecycle error). The message names
both builds and the remedy: restart that server with this build (on its host,
`batchalign3 serve stop`, then `batchalign3 serve start`) and run the command
again.

A server left running across an upgrade would otherwise produce results with
the older build's engines and fixes, which the CLI would then write as if this
build had produced them. Earlier builds only printed a warning and ran the job
anyway.

The same rule governs REUSE, not only submission. When the CLI finds a
manually started server on the port that server published, it reads that
server's `/health` before adopting it: on this build the server is reused, and
on another build, or one reporting no build at all, the command is refused
there with the same message and the same exit code `5`. Earlier builds printed
a warning at that point and reused the server anyway, leaving the refusal to
happen later at submission. A server that answers nothing recognizable is not
reused either; the CLI falls through to its own daemon path, which probes the
configured port and reports what holds it.

## Backend model

The server now has a single **local in-process control plane**.

- There is no Temporal backend and no backend-selection config.
- Job detail surfaces still report `control_plane.backend`, but the only released value is `local`.
- On restart, in-flight work from the old process does **not** continue running in place. Recovery reloads queued/interrupted work from SQLite and re-dispatches resumable jobs when the server comes back up.
- A recovered file that had finished is named by that command's primary output artifact, with that artifact's content type, rather than by the input file it was produced from. The persisted file-status rows name inputs, not artifacts, so a recovered result would otherwise be offered under the wrong name.

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
batchalign3 serve start --foreground --test-echo
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
media_roots: []
media_mappings: {}
```

Important keys:

- `port`: server listen port
- `host`: bind address (defaults to `0.0.0.0`)
- `max_concurrent_jobs`: `0` means auto-tune
- `auto_daemon`: reuse or start a loopback daemon for ordinary CLI processing
- `media_roots`: local execution-host media lookup roots
- `media_mappings`: local execution-host root mappings from corpus paths to mounted media paths
- `memory_tier`: override auto-detected tier: `small`, `medium`, `large`, `fleet`
- `memory_gate_mb`: host headroom reserve in MB (default: 2048)
- `gpu_startup_mb` / `stanza_startup_mb` / `io_startup_mb`: per-profile startup reservation overrides
- `worker_health_interval_s`: health check frequency in seconds (default: 30)
- `job_ttl_days`: auto-delete completed jobs after this many days (default: 7)

OTLP tracing can be enabled by setting `BATCHALIGN_OTLP_ENDPOINT`
(or `OTEL_EXPORTER_OTLP_ENDPOINT`) in the server environment.

`server.yaml` uses a strict schema. Unknown keys are rejected at startup
instead of being silently ignored, so stale config must be updated to the
current key set.

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
