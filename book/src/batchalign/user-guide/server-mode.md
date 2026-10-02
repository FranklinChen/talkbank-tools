# Server Mode

**Status:** Current
**Last updated:** 2026-10-01 14:52 EDT

Batchalign includes a built-in HTTP server managed by `batchalign3 serve ...`.
Ordinary local processing commands can still run inline, but when
`auto_daemon: true` (the default) the CLI first tries to reuse or start a
loopback daemon so warm workers survive across commands. `--no-server` and
`--sequential` still force direct local execution.

## Current routing rules

What a command's inputs ARE decides where it can run:

- **Inputs are transcripts:** `morphotag`, `utseg`, `translate`, `coref`,
  `compare`, `align`, `speaker-identify`. A `--server` on another host runs
  them. The CLI sends the transcript text and receives the results as text;
  for `align` and `speaker-identify` the server finds each recording on its
  own filesystem (see "Remote use" below). No recording crosses the
  connection.
- **Inputs are recordings:** `transcribe`, `transcribe_s`, `benchmark`,
  `opensmile`, `avqi`, `diarize`. No transport carries a recording to
  another host, so a `--server` on another host is refused before anything
  is sent. Run these where the recordings are.
- A `--server` naming this machine (`localhost` or a loopback address) uses
  the shared filesystem for every command: the CLI submits paths, the
  server reads the inputs and writes the outputs in place.
- Without `--server`, `auto_daemon: true` (the default) reuses or starts a
  loopback daemon on the same shared-filesystem terms before falling back to
  direct in-process execution. `--no-server` and `--sequential` force direct
  execution.

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

`serve stop` asks the server to shut down and waits up to 60 seconds for it to
finish: the server records its running jobs as interrupted (they resume on the
next start) and retires its workers. Only a server that has not finished by
then is killed. Its workers never outlive it either way: every worker watches
the server that launched it and exits when that process exits, even in the
middle of an inference.

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
- `media_roots`: directories the execution host searches for a transcript's recording when no mapping matched
- `media_mappings`: corpus repository name to media directory, on the execution host; selected automatically when a submitted path contains the key (see "Remote use")
- `memory_tier`: override auto-detected tier: `small`, `medium`, `large`, `fleet`
- `memory_gate_mb`: host headroom reserve in MB (default: 2048)
- `gpu_startup_mb` / `stanza_startup_mb` / `io_startup_mb`: per-profile startup reservation overrides
- `worker_health_interval_s`: health check frequency in seconds (default: 30)
- `worker_ready_timeout_s`: seconds a spawned worker has to become ready (default: 300)
- `audio_task_timeout_s`: raise the transport ceiling for audio tasks, in
  seconds. Leave it out for the built-in ceiling: ASR's scales with the audio
  (an override can only raise it), forced alignment and speaker diarization
  use 1800. `--timeout` sets the same value from the command line.
- `analysis_task_timeout_s`: transport ceiling for OpenSMILE and AVQI, in
  seconds (built-in: 120)
- `ensure_task_timeout_s`: how long on-demand model loading may take, in
  seconds (built-in: 120); raise it on slow networks
- `worker_registry_path`: the pre-started worker registry file (built-in:
  `workers.json` in the state directory)
- `job_ttl_days`: auto-delete completed jobs after this many days (default: 7)

Every timeout and interval above is a positive whole number of seconds. "Use
the built-in value" is spelled by leaving the key out, never by writing `0`.
Older files that wrote `0` (or `""` for `worker_registry_path`), which used to
mean exactly that, still load: the timeout overrides and the registry path
read it as absent, and `worker_health_interval_s` and `worker_ready_timeout_s`
read it as their default. In code the overrides are `Option<PositiveSeconds>`
and the two intervals are `PositiveSeconds`, so no consumer tests for a zero.
`--timeout 0` on the command line is refused with that instruction.

A count that must be positive and is written as `0` or below is refused, not
corrected: `job_ttl_days`, `memory_gate_poll_s`, `max_concurrent_worker_startups`
and `local_lease_ttl_s` stop the server, the CLI and `batchalign3 doctor` alike
with an error naming the field (`job_ttl_days must be >= 1 (got 0); fix it in
server.yaml, or leave it out for the default`). Until 2026-10-01 such a value
was quietly raised to 1, with a warning only some loaders printed, so
`doctor` could pass a file the server then ran differently. Every loader now
reads the file the same way, through one function.

OTLP tracing can be enabled by setting `BATCHALIGN_OTLP_ENDPOINT`
(or `OTEL_EXPORTER_OTLP_ENDPOINT`) in the server environment.

`server.yaml` uses a strict schema. Unknown keys are rejected at startup
instead of being silently ignored, so stale config must be updated to the
current key set.

## Remote use

A server on another host runs the transcript commands. The transcript text
crosses the connection, the results come back the same way, and recordings
never cross it:

```bash
batchalign3 --server http://myserver:8000 morphotag corpus/ -o output/
batchalign3 --server http://myserver:8000 align corpus/ -o output/
```

### How the server finds a recording

For `align` and `speaker-identify` the execution host looks for a file with
the transcript's stem, in this order, and stops at the first hit:

1. `--media-dir`, if you gave one; it must name a directory on the server.
2. Beside the transcript, when the server shares your filesystem (local runs
   and loopback servers).
3. The directory you submitted from, if that same path exists on the server.
4. A `media_mappings` entry, selected by the path you submitted: when any
   component of that path equals a mapping key, such as a corpus repository
   name, that mapping's root is searched under the same subdirectory.
5. The server's `media_roots`.
6. Beside the transcript again, as a last resort.

The same order, as the code walks it, is drawn on
[Path Provenance](../architecture/path-provenance.md).

Step 4 is the one to know. A transcript inside a checkout of
`childes-eng-na-data/Brown/Adam/` aligns on a server whose `media_mappings`
has a `childes-eng-na-data` entry, with no flag and nothing printed; a
transcript in an arbitrary folder aligns only on a server that has
`media_roots`. Both are configuration of the execution host, never of the
client, and a client's private directory layout is never dereferenced. A
recording that exists only on your machine cannot be aligned on a remote
server: run `align` where the recording is, or against a loopback server.

`transcribe` and the other recording-input commands cannot use a remote
server at all; the CLI says so before contacting it. Run them on the host
that holds the recordings.

What each choice costs over the network, including the case where the
server's own media root is a network mount, is on
[Network and Transfer Costs](network-costs.md).

Builds from 0.3.0 (2026-08-30) to 2026-09-22 refused `align` for every
non-loopback server with "cannot send local audio to non-loopback server".
That was a bug, not the design.
