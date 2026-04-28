# Python Version Support

**Status:** Current
**Last verified:** 2026-04-07 06:29 EDT

## Current policy

Python 3.12 is the current contributor and deployment baseline.

Python 3.14t (free-threaded / no-GIL) is **paused again**. Do not treat it as a
supported install path, do not deploy it to fleet machines, and do not weaken
the main package contract just to make 3.14t possible.

The next realistic revisit point is **Python 3.15 or newer**, assuming the
relevant ML ecosystem has better wheel coverage and better evidence of runtime
stability.

## Why 3.14t is paused again

### 1. The default install must stay complete

`batchalign3 transcribe --diarize` is a real supported CLI path. After the BA2
parity audit, that path once again means "run the dedicated post-ASR speaker
stage" even on top of Rev-labeled output.

Because of that, `pyannote.audio` and `onnxruntime` belong in the **standard**
`batchalign3` install. We are not keeping speaker diarization in a special
optional tier just to unblock a narrow 3.14t experiment.

### 2. 3.14t already has a bad operational history for us

See the kernel panic postmortem: `docs/postmortems/2026-03-11-kernel-panic.md`.

The free-threaded Python path is still risky for our actual ML stack:

- PyTorch / NumPy / related C extensions under real concurrent load remain a
  stability concern
- the production fleet needs boring, predictable behavior more than it needs an
  experimental concurrency win
- we already have a reliable 3.12 deployment path

### 3. The wheel ecosystem is still incomplete

When speaker diarization stays in the base package, `onnxruntime` comes back
into the required dependency set. That means the old Apple/macOS `cp314t`
wheel-coverage problem is once again a hard blocker for a full standard 3.14t
install.

## Historical finding worth preserving

We should still remember **why** 3.14t was attractive.

The real benefit was never small Python startup wins. It was **shared Stanza
model memory** for `morphotag` and `utseg`.

### February 2026 measurements

These measurements came from the earlier pipeline architecture, but the core
observation remains important for future revisits:

| Host | Scenario | Peak RSS | Files/hour |
| --- | --- | ---: | ---: |
| worker-machine | GIL=1, 4 workers | 13.5 GB | 10,069 |
| worker-machine | GIL=0, 4 threads | 3.0 GB | 10,158 |

That is roughly a **77% memory reduction** with essentially unchanged
throughput. If the ecosystem matures, this is the reason to reconsider
free-threaded Python later.

## What remains in the codebase

Some free-threaded groundwork remains in tree and is harmless to keep:

- runtime detection of free-threaded interpreters
- distinct memory-budget tables for process vs. threaded serving
- thread-safe tokenizer realignment state
- harness-side cleanup of `PYTHON_GIL` inheritance

Those are acceptable to keep as future-facing infrastructure. They should **not**
be taken as permission to target 3.14t in packaging, CI, or fleet policy.

## Packaging policy

- `pyannote.audio` and `onnxruntime` are part of the standard package again
- a missing speaker runtime is a broken install, not an alternate supported mode
- the public install contract remains: one normal install provides the supported
  command surface, including dedicated speaker diarization

## Revisit criteria

Revisit no-GIL Python only when **all** of the following are true:

1. Python 3.15 or newer has materially better ecosystem support for our stack.
2. Required diarization/runtime dependencies have compatible wheels on the
   platforms we actually use.
3. We can demonstrate stable end-to-end ML workloads on isolated hosts without
   crashes or pathological memory behavior.
4. CI and release automation can exercise that runtime intentionally instead of
   relying on ad hoc contributor machines.

Until then:

- use Python 3.12 for development
- use Python 3.12 for deployment
- treat 3.14t as shelved research, not an active engineering target
