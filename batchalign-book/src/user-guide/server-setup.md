# Server and Fleet Setup

**Status:** Current
**Last updated:** 2026-03-29 17:58 EDT

## Overview

Batchalign3 can run as a persistent server that accepts jobs from remote
clients. This is useful for teams where:

- Multiple people share one powerful machine
- You want to keep ML models warm (no startup delay)
- Long-running batch jobs need crash recovery
- Audio files are on a central server, not on each laptop

## Architecture

```
┌───────────────────────────────────────┐
│  Server machine (GPU, lots of RAM)    │
│                                       │
│  ┌────────────────┐  ┌─────────────┐ │
│  │ Temporal server │  │ batchalign3 │ │
│  │ (job queue)     │──│ server      │ │
│  │ port 7233       │  │ port 8001   │ │
│  └────────────────┘  └─────────────┘ │
│                            │          │
│                       Python workers  │
│                       (Stanza, Whisper)│
└───────────────────────────────────────┘
         ▲           ▲           ▲
    ┌────┘           │           └────┐
 Laptop A         Desktop B      Laptop C
 --server URL     --server URL   --server URL
```

Clients use `--server http://server:8001` to send work. The server
dispatches to Python workers, manages job lifecycle, and returns results.

## Single-Machine Server

The simplest setup: one server, multiple clients.

### 1. Install batchalign3 on the server

```bash
uv tool install batchalign3
```

### 2. Install Temporal (job queue)

Temporal provides crash recovery, activity timeouts, and job history.

```bash
# macOS
brew install temporal

# Linux
curl -sSf https://temporal.download/cli | sh
```

### 3. Start Temporal

```bash
temporal server start-dev \
  --db-filename ~/.temporal/temporal.db \
  --ip 0.0.0.0 \
  --log-level warn
```

For production, run Temporal as a system service (launchd on macOS,
systemd on Linux) so it survives reboots.

### 4. Configure batchalign3

Create `~/.batchalign3/server.yaml`:

```yaml
port: 8001
host: "0.0.0.0"
max_concurrent_jobs: 4
temporal_server_url: "http://127.0.0.1:7233"
temporal_namespace: "default"
temporal_task_queue: "batchalign3-queue"
temporal_activity_timeout_s: 3600
temporal_heartbeat_s: 30

# Map data repository names to media file locations.
# Adjust paths to your media storage.
media_mappings:
  my-corpus: /path/to/audio/files
  another-corpus: /path/to/more/audio
```

### 5. Start the server

```bash
batchalign3 serve start --port 8001 --host 0.0.0.0 -v
```

### 6. Connect from clients

On any machine that can reach the server:

```bash
batchalign3 --server http://server:8001 morphotag corpus/ -o output/
batchalign3 --server http://server:8001 align corpus/ -o output/
```

## Multi-Machine Fleet

For larger teams, multiple machines can serve as Temporal workers. All
workers poll the same task queue — Temporal automatically load-balances.

### Prerequisites

- One machine runs Temporal (the coordinator)
- All worker machines can reach Temporal's port (7233)
- All worker machines can access media files (NFS, shared storage, etc.)

### Worker setup

On each additional worker machine:

1. Install batchalign3
2. Create `~/.batchalign3/server.yaml` pointing at the Temporal server:

```yaml
port: 8001
host: "0.0.0.0"
max_concurrent_jobs: 2
temporal_server_url: "http://coordinator:7233"
temporal_namespace: "default"
temporal_task_queue: "batchalign3-queue"

media_mappings:
  my-corpus: /mnt/nfs/audio/files
```

3. Start the server:

```bash
batchalign3 serve start --port 8001 --host 0.0.0.0 -v
```

The worker registers with Temporal and begins polling for tasks.

### Shared media via NFS

For audio commands (align, transcribe), worker machines need access to
the same media files. The recommended approach:

1. Export media directories from the central server via NFS
2. Mount NFS on all worker machines at a canonical path
3. Configure `media_mappings` in each worker's `server.yaml` to point
   at the NFS mount paths

### Client routing

Clients can submit to any server's REST endpoint. Temporal routes the
actual work to whichever worker is idle:

```bash
# Submit to the coordinator — Temporal may route to any worker
batchalign3 --server http://coordinator:8001 morphotag corpus/ -o output/
```

## Temporal UI

Temporal includes a web UI for monitoring workflows:

```
http://coordinator:8233
```

The UI shows:
- Active and completed workflows
- Activity history and timing
- Retry counts and failure reasons
- Worker task queue status

## Server Management

```bash
# Check server status
batchalign3 serve status

# Stop the server
batchalign3 serve stop

# View server health
curl http://localhost:8001/health | python3 -m json.tool
```

## Direct Mode vs Server Mode

| Aspect | Direct mode | Server mode |
|--------|------------|-------------|
| Setup | None | Temporal + server.yaml |
| Model loading | ~4-7s on first run | Always warm (instant) |
| Crash recovery | None (restart manually) | Temporal resumes jobs |
| Multi-user | No | Yes (concurrent jobs) |
| Remote audio | Must be local | Via NFS/media_mappings |
| Monitoring | Terminal output | Web dashboard + Temporal UI |

**Most users should start with direct mode.** Server mode is for teams
managing shared infrastructure.
