# How `align` Throughput Works

**Status:** Current  
**Last verified:** 2026-03-05

This page describes the current Rust server / Python worker runtime for
`batchalign3 align`. Older BA2 Python-CLI executor details are relevant only in
the migration book, not as the active release path.

## Current execution model

`align` is no longer a Python CLI `ThreadPoolExecutor` pipeline. In the current
system:

1. The Rust CLI submits the job to a single server or local daemon.
2. The Rust server groups utterances into FA windows per file.
3. The server checks the utterance cache per group.
4. Cache misses are sent to Python workers through the worker pool.
5. Rust applies returned timings, generates `%wor`, and enforces monotonicity.

Key current components:

- server runner: `crates/batchalign-app/src/runner/mod.rs`
- FA orchestrator: `crates/batchalign-app/src/fa/`
- FA grouping/alignment: `crates/batchalign-chat-ops/src/fa/`
- worker pool: `crates/batchalign-app/src/worker/pool/`

## Throughput levers that matter now

### Worker reuse

Python workers are persistent subprocesses keyed by `(command, lang)`. The
server reuses idle workers instead of paying cold-start cost on every file.

### Job concurrency cap

The server limits concurrent jobs with `max_concurrent_jobs`. This is the main
top-level throughput guard, not the old Python CLI executor split.

### Pre-scaling

For multi-file jobs, the server pre-scales workers before dispatch to reduce
sequential spawn latency.

### Cache reuse

Forced-alignment results are cached per audio window + transcript text +
timing mode + engine. Re-runs mainly pay for changed groups.

### Largest-first discovery

Current CLI discovery sorts matching files by size descending before
submission, which reduces straggler-heavy long runs.

## What is still parallel and what is not

- parallel:
  - multiple files/jobs can be active concurrently
  - multiple workers can handle different infer/execute requests
- not magically parallel:
  - a single FA window still depends on the underlying model/runtime cost
  - cache misses still require real model work
