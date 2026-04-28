# Performance

**Status:** Current
**Last updated:** 2026-03-26 18:12 EDT

This page covers what to expect from Batchalign's processing times and how to
improve throughput.

## Cold vs warm starts

The first run of any command downloads ML models and initializes them — expect
5-20x longer than subsequent runs. After the first run:

- **Model cache:** Stanza, Whisper, and other ML models are cached on disk
  (~2 GB total). They load from cache on subsequent runs.
- **Server warmth:** When an explicit server is running, workers can stay warm
  in memory across multiple jobs. Direct local execution does not keep a daemon
  alive between CLI invocations.
- **Analysis cache:** Batchalign caches NLP results (morphosyntax, alignment)
  in a local SQLite database keyed by content hash. Re-running the same file
  returns cached results instantly.

| Scenario | Relative Speed |
|----------|---------------|
| First run (model download + init) | 1x (baseline) |
| Cold start (models cached on disk) | 3-5x faster |
| Warm server (models in memory) | 5-20x faster |
| Cached result (identical input) | Near-instant |

## Worker count

By default, Batchalign uses one worker per command. For batch processing of
many files, increase the worker count:

```bash
batchalign3 morphotag ~/corpus/ -o ~/output/ --workers 4
```

Each worker loads its own copy of the ML models. Memory usage scales linearly
with worker count — see the memory section below.

## CPU vs GPU

Batchalign automatically uses GPU acceleration when available (CUDA on Linux,
MPS on macOS). To force CPU-only processing:

```bash
batchalign3 morphotag ~/corpus/ -o ~/output/ --force-cpu
```

CPU-only is slower but uses less memory and avoids GPU driver issues. On
machines without a supported GPU, CPU mode is selected automatically.

## Memory patterns

Memory usage depends on the command and number of workers:

| Command | ~Memory per Worker |
|---------|-------------------|
| `morphotag` | 1-2 GB (Stanza models) |
| `align` | 2-4 GB (Whisper/Wave2Vec) |
| `transcribe` | 2-4 GB (Whisper + diarization) |
| `translate` | 1-2 GB (translation model) |
| `utseg` | 1-2 GB (constituency parser) |
| `benchmark` | <500 MB (no ML models) |

With `--workers N`, total memory is roughly `N * per-worker cost`. The Rust
runtime adds minimal overhead (~50 MB).

**Lazy audio loading:** Audio files are loaded on demand and released after
processing — memory does not grow with corpus size, only with concurrent
workers.

## Server mode for warm models

For repeated interactive use, keep models loaded in the background:

```bash
batchalign3 serve start
```

Subsequent commands automatically connect to the running daemon. Stop it when
done:

```bash
batchalign3 serve stop
```

See [Server Mode](server-mode.md) for configuration details and
[Worker Tuning](worker-tuning.md) for memory budgets and warmup configuration.

## The `bench` command

Measure processing throughput on your hardware:

```bash
batchalign3 bench morphotag ~/sample-corpus/ --workers 1
batchalign3 bench morphotag ~/sample-corpus/ --workers 4
```

This runs the command with timing instrumentation and reports files/second and
wall-clock time per file.

## Estimated times per command

Rough estimates for a single file (~100 utterances) on a modern laptop with
warm daemon:

| Command | Warm Daemon | Cold Start |
|---------|------------|------------|
| `morphotag` | 2-5 seconds | 30-60 seconds |
| `align` | 5-15 seconds | 45-90 seconds |
| `transcribe` | 10-60 seconds (depends on audio length) | 60-120 seconds |
| `translate` | 2-5 seconds | 30-60 seconds |
| `utseg` | 3-8 seconds | 30-60 seconds |
| `benchmark` | <1 second | <1 second |

Times vary significantly with hardware, file size, and language. GPU
acceleration typically provides a 2-5x speedup for model inference.
