# Apple MPS Workarounds

**Status:** Current
**Last updated:** 2026-04-06 10:01 EDT

## MPS Exclusion Decision (2026-04-05)

**As of 2026-04-05, MPS is excluded from ALL model loaders.** All GPU-profile
workers (ASR, FA, Speaker) run on CPU when CUDA is unavailable. MPS is never
used, on any fleet machine.

### Why: confirmed unsafe behavior, with host-specific failure modes

On 2026-04-05, FA jobs on MPS produced confirmed AGXG14X shutdown stalls on
two fleet machines. Surviving local artifacts recovered after reboot show a
batchalign FA worker inside
`batchalign_core::worker_fa_exec::run_wave2vec_like` →
`at::mps::MPSStream::executeMPSGraph`, a 108.88 GB footprint, only 191.93 MB
free on the internal drive, `ffmpeg` `No space left on device` failures while
building media-cache WAVs and FA temp PCM, and follow-on shared GPU worker
crashes.

That is enough to treat MPS as unsafe by default. Detailed operator
postmortems are tracked out-of-tree. One machine gave the clean AGX
shutdown-stall signature; the other proved MPS involvement plus
catastrophic disk and memory pressure during the same incident window.

There is no user-space mitigation for the AGX deadlock path. You cannot:
- Time out the operation (kernel mutex, not user-space)
- Kill the stuck process (unkillable zombie in kernel sleep)
- Break the compute into smaller chunks (deadlock triggered on normal 5–30s chunks)
- Use a subprocess watchdog (PyTorch MPS silently hangs or SIGSEGVs in forks:
  [pytorch/pytorch#178037](https://github.com/pytorch/pytorch/issues/178037))

The `net` incident also showed that MPS exclusion is not sufficient by itself:
long FA jobs can fill both
`~/Library/Application Support/batchalign3/media_cache/` and
`/var/folders/.../fa_v2/audio/` if internal free space is already low.
Apple-performance work must therefore include cache cleanup and temp-space
admission control, not just device selection.

### Ecosystem evidence

No major ML project defaults to MPS:

| Project | MPS status |
|---------|-----------|
| **OpenAI Whisper** | PR [#382](https://github.com/openai/whisper/pull/382) to enable MPS was never merged — users reported crashes, MPS slower than CPU |
| **faster-whisper** | MPS not supported at all (`ValueError: unsupported device mps`) |
| **pyannote-audio** | MPS produces wrong timestamps ([#1337](https://github.com/pyannote/pyannote-audio/issues/1337), closed wontfix); kernel crashes on M4 ([#1886](https://github.com/pyannote/pyannote-audio/issues/1886)) |
| **whisper.cpp** | Uses Metal API directly, bypassing PyTorch MPS entirely |
| **PyTorch Lightning** | MPS marked "experimental" |

PyTorch MPS status as of April 2026:
- **1,106 total MPS issues** (252 open, 12 high-priority)
- **178 silent correctness bugs** (`module: mps` + `module: correctness (silent)`)
- [#178497](https://github.com/pytorch/pytorch/issues/178497): `sum`, `mean`, `count_nonzero` give wrong results — confirmed by PyTorch team as originating in Apple's MPS framework
- [#179352](https://github.com/pytorch/pytorch/issues/179352): `scaled_dot_product_attention` incorrect for large batches (cosine similarity 0.49 vs CPU)
- [#154329](https://github.com/pytorch/pytorch/issues/154329): memory leak (~1 MB/sec) confirmed on M4 Max/Studio
- [#144634](https://github.com/pytorch/pytorch/issues/144634): `torch.mps.synchronize()` hangs on error — Apple engineer acknowledged, still open

Apple has not publicly acknowledged the AGXG14X kernel deadlock.

### Deadlock and hang categories

Two PyTorch issues carry the dual labels `module: mps` + `module: deadlock`:

1. [pytorch/pytorch#144634](https://github.com/pytorch/pytorch/issues/144634) —
   `torch.mps.synchronize()` hangs on shader fault (open since 2025-01-12).
   Filed by PyTorch maintainer `@malfet`: "few attempt[s] to reproduce the same
   resulted in **system hang**." Apple engineer acknowledged, still open.

2. [pytorch/pytorch#162872](https://github.com/pytorch/pytorch/issues/162872) —
   `Event.synchronize()` deadlocks before `elapsed_time()` (open since
   2025-09-13). Reproduced on M4 Pro. Simple timing code hangs permanently.

Additionally:

- [pytorch/pytorch#178037](https://github.com/pytorch/pytorch/pull/178037) —
  MPS silently hangs or SIGSEGVs in forked subprocesses (PR filed 2026-03-21).
  Unlike CUDA, MPS had no `_lazy_init()` check in forked children. This rules
  out the "watchdog subprocess" mitigation strategy.

- The MLX project documented macOS GPU watchdog kills at
  [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267): the
  error `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` fires when a
  Metal command buffer blocks WindowServer compositing for too long. An
  undocumented env var `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` exists but is
  unsupported; MLX maintainers marked this "wontfix."

### Silent correctness failures

The most insidious MPS problem is not crashes but **silently wrong results**:

- **178 issues** labeled `module: mps` + `module: correctness (silent)`
- Reductions (`sum`, `mean`, `nansum`, `trace`, `count_nonzero`) give wrong
  results when called in rapid succession — confirmed by PyTorch team as
  originating in Apple's MPS framework itself, not PyTorch
  ([#178497](https://github.com/pytorch/pytorch/issues/178497))
- `scaled_dot_product_attention` returns incorrect output for large
  batch × sequence products — cosine similarity drops to 0.49 vs CPU
  ([#179352](https://github.com/pytorch/pytorch/issues/179352), filed
  2026-04-04)
- "Catastrophically wrong gradients" (1,000×–100,000× too large) when total
  elements exceed 32K
  ([#177116](https://github.com/pytorch/pytorch/issues/177116), fixed in
  PyTorch 2.11)
- `torch.multinomial` crashes with SIGSEGV on MPS for larger tensors
  ([#178579](https://github.com/pytorch/pytorch/issues/178579))
- `uint16/uint32/uint64` binary ops produce garbage values
- `ComplexFloat` dtype not supported at all

For an ASR/FA pipeline, silent attention bugs mean **wrong transcriptions and
wrong alignments with no error**. This is arguably worse than a crash.

### Performance on Apple Silicon

Academic benchmarking ([arxiv:2511.05502](https://arxiv.org/abs/2511.05502))
found that for LLM inference on Apple Silicon:
- **MLX** achieves highest throughput (~230 tok/s)
- **llama.cpp** excels for single-stream inference
- **PyTorch MPS** ranked last among 5 frameworks tested (~7–9 tok/s)
- PyTorch MPS "remains limited by memory constraints on large models"

For Whisper specifically, OpenAI Whisper PR #382 found MPS was **slower than
CPU** on Apple Silicon (5.25s vs 3.26s). The `PYTORCH_ENABLE_MPS_FALLBACK`
path was "20× slower than CPU alone" because unsupported ops bounce between
GPU and CPU with expensive data transfers.

### Apple's response

- Apple engineer `@jhavukainen` is tagged on PyTorch MPS deadlock issues
  (#144634, #162872) but responses have been limited to "I'll need to consult
  a colleague."
- Apple's own `tensorflow-metal` plugin had GPU hangup issues; v0.5.1 fixed
  "multiple memory leak issues leading to GPU hangups." Users reported
  `IOGPUDevice::new_resource: PID likely leaking IOGPUResource (count=200000)`.
- macOS 15 (Sequoia) added native non-contiguous tensor support in Metal,
  fixing a class of silent correctness bugs. macOS 26 (Tahoe) introduced
  Metal 4 with `MTLTensor`, but no stability improvements for ML compute
  workloads were announced.
- The GPU watchdog timer that kills long compute is by design; there is no
  official mechanism to disable it.

### What changed

| Module | Before (pre-2026-04-05) | After |
|--------|------------------------|-------|
| `fa.py` (Whisper FA) | CUDA (float16) > MPS (float32) > CPU (float32) | CUDA (float16) > CPU (float32) |
| `fa.py` (Wave2Vec FA) | CUDA > MPS (float32 via `.float()`) > CPU | CUDA > CPU |
| `asr.py` (Whisper ASR) | CUDA (bfloat16) > MPS (float32) > CPU (bfloat16) | CUDA (float16) > CPU (float32) |
| `speaker.py` | CUDA > CPU (already excluded) | CUDA > CPU (unchanged) |
| `_main.py` (serving) | GPU profile = always concurrent (ThreadPoolExecutor) | GPU profile = concurrent on CUDA only; sequential on CPU |
| `__init__.py` | `PYTORCH_ENABLE_MPS_FALLBACK=1` set | Removed (not needed) |

### Worker concurrency impact

GPU-profile workers previously used `ThreadPoolExecutor(gpu_thread_pool_size)`
for concurrent inference, relying on PyTorch releasing the GIL during GPU
kernels. On CPU, this causes thread oversubscription: each thread's PyTorch ops
use all cores via OpenMP, so 4 threads × 24 cores = 96 threads fighting for 24
cores on net's M3 Ultra.

Fix: GPU-profile workers now serve sequentially on CPU (one request at a time,
all cores per request). `gpu_thread_pool_size` in `server.yaml` takes effect
only when CUDA is available.

### Implications for platform strategy

The MPS exclusion means Apple Silicon machines run all ML inference on CPU.
This has direct consequences for fleet planning:

**Current state:** `net` (M3 Ultra, 24 P-cores, 256 GB) runs Whisper, Wav2Vec2,
and Stanza on CPU. We do not yet have a fresh post-exclusion benchmark set for
`align`, `transcribe`, or mixed workloads, and the Apr 5 incident showed that
cache/temp-space pressure can dominate host behavior on long Apple jobs. Treat
Apple CPU-only as the safe baseline, not yet a fully characterized performance
baseline.

**Linux with CUDA changes everything.** A Linux box with even a mid-range NVIDIA
GPU (e.g., RTX 4090, 24 GB VRAM) would:

1. **Restore GPU acceleration** — CUDA is the only stable, production-grade GPU
   backend for PyTorch. No workarounds, no silent correctness bugs, no kernel
   deadlocks. The entire PyTorch ecosystem is built and tested on CUDA first.
2. **Enable concurrent GPU serving** — the `ThreadPoolExecutor` in GPU-profile
   workers would activate, allowing multiple inference requests to overlap on
   the GPU. This is the architecture we designed for but cannot use on Apple.
3. **Float16/bfloat16 inference** — CUDA supports both natively, giving 2×
   throughput and 50% memory savings vs float32 on CPU.
4. **Pyannote speaker diarization on GPU** — currently forced to CPU. On CUDA,
   pyannote is fully supported and significantly faster.
5. **Eliminate MPS-related complexity** — no dtype workarounds, no device
   exclusion code, no `PYTORCH_ENABLE_MPS_FALLBACK`, no MPS-specific tests.
   The entire MPS section of this document becomes irrelevant.
6. **Standard deployment** — Docker containers, NVIDIA Container Toolkit,
   well-documented MLOps patterns. The successor will find abundant community
   resources for maintaining a CUDA deployment.

The code is already CUDA-ready: all model loaders select CUDA first when
available, `gpu_thread_pool_size` activates, and the worker profiles are
designed for GPU concurrency. No code changes would be needed to benefit from
CUDA — just deploy on a Linux box with an NVIDIA GPU.

This reinforces the succession plan's preference for "managed services over
self-hosted" and "standard tooling over custom scripts." A Linux + CUDA
deployment is the industry standard; Apple Silicon + CPU-only is the
workaround.

---

The sections below are preserved as historical reference for the MPS
workarounds that were in place before the full exclusion.

## Historical Context

Apple's Metal Performance Shaders (MPS) backend in PyTorch provides GPU
acceleration on Apple Silicon but has fundamental limitations. This page
documents all MPS issues we encountered during the period when MPS was used
(2025–2026-04-05), the workarounds we applied, and the upstream issues that
remain unresolved.

Our primary deployment target is `net` — a Mac Studio with an M3 Ultra and
256 GB RAM.

## Hardware Limitations

Metal (Apple's GPU framework) does **not** support:

| Type | Status | PyTorch behavior |
|------|--------|-----------------|
| **bfloat16** | Not in Metal spec | Crashes, wrong results, or `TypeError` depending on operation |
| **float64** | Not in Metal spec | `TypeError: Cannot convert Double to MPS` |
| **int64** | Not in Metal spec | Crashes on some ops (e.g. `abs_out_mps`) |
| **complex128** | Not in Metal spec | Conversion failure |

These are hardware/framework limitations, not PyTorch bugs. No fix is expected.

## Global Workaround (Removed)

Previously set in `batchalign/__init__.py`:

```python
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
```

This was removed on 2026-04-06 because MPS is no longer used. When MPS was
active, it caused unsupported operations to fall back to CPU instead of
crashing. The fallback was slow and produced unexpected device transfers.

## Per-Module Workarounds

### ASR — Whisper (`inference/asr.py`)

```python
if device.type == "mps":
    asr_dtype = torch.float32   # not bfloat16
```

Whisper ASR uses `bfloat16` on CUDA for speed. On MPS, this crashes with Metal
assertion failures. We force `float32`. A second fallback path also forces
`float32` on MPS for older transformers versions that don't accept `bfloat16`
at all.

The HuggingFace Transformers Whisper pipeline requires
`attn_implementation="eager"` on MPS — the SDPA attention path broke MPS in
transformers v4.40.0.

### Forced Alignment — Whisper FA (`inference/fa.py`)

```python
if device.type == "mps":
    torch_dtype = torch.float32   # not float16
```

Same pattern as ASR. Whisper FA uses `float16` on CUDA, `float32` on MPS/CPU.

### Forced Alignment — Wave2Vec FA (`inference/fa.py`)

```python
model = bundle.get_model()
if device.type == "mps":
    model = model.float()  # Force float32
model = model.to(device)
```

The torchaudio `MMS_FA` bundle's default parameters can include bfloat16 ops
on MPS. Under concurrent load with large audio files (200+ MB video → WAV →
inference), this causes worker crashes that surface as `Broken pipe (os error
32)`. The `.float()` call converts all parameters to float32 before moving to
device.

**Incident:** 2026-03-16, a user's aphasia-data ACWT job — 6/11 files failed.
See `docs/postmortems/2026-03-16-wave2vec-mps-crash.md`.

### Speaker Diarization (`inference/speaker.py`)

Speaker diarization was the first module to exclude MPS (before the 2026-04-05
incident). The old code used `CUDA > MPS > CPU`; the fix changed it to:

```python
# Current code (MPS excluded):
return "cuda" if torch.cuda.is_available() else "cpu"
```

MPS was excluded from diarization because:

- **Pyannote on MPS** produces wrong timestamps
  ([pyannote/pyannote-audio#1337](https://github.com/pyannote/pyannote-audio/issues/1337),
  closed as wontfix). Kernel crashes also reported on M4
  ([#1886](https://github.com/pyannote/pyannote-audio/issues/1886)).
- **NeMo** is CUDA-only by design — no MPS support at all.

The device selector (`_device_for_speaker_runtime`) returns `"cuda"` or
`"cpu"`, never `"mps"`.

### Device Policy (`device.py`)

The `BATCHALIGN_FORCE_CPU` environment variable (or `DevicePolicy(force_cpu=True)`)
forces all model loaders onto CPU. This is the escape hatch when MPS causes
problems that dtype coercion alone can't fix.

## Memory Issues on MPS

MPS has well-documented memory management problems:

- **Memory leaks** during inference: usage climbs steadily, eventually OOM
  ([pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329),
  [#145374](https://github.com/pytorch/pytorch/issues/145374))
- **OOM with memory available**: MPS cache doesn't release when it should
  ([pytorch/pytorch#105839](https://github.com/pytorch/pytorch/issues/105839))
- **`sysinfo::available_memory()`** on macOS undercounts — reports only
  free + purgeable, missing reclaimable file cache. On net (256 GB, heavy I/O),
  this can underreport by tens of GB. No fix exists because macOS doesn't
  expose a `MemAvailable` equivalent like Linux.

**Mitigations:**
- `torch.mps.empty_cache()` — call periodically during long-running inference
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — disables MPS memory limit (risks
  system instability, not recommended for production)
- Our Rust server's memory gate uses `sysinfo::available_memory()` with a
  configurable threshold (default 2048 MB, `0` to disable). Idle worker bypass
  prevents deadlock when loaded workers hold RAM.

## Upstream Issues to Track

Check these periodically. If an issue is resolved, we may be able to remove
the corresponding workaround.

### bfloat16

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#141864](https://github.com/pytorch/pytorch/issues/141864) | Closed (won't fix) | N/A — Metal lacks native bfloat16. Would require Apple hardware/firmware change. |
| [pytorch/pytorch#136624](https://github.com/pytorch/pytorch/issues/136624) | Closed | Specific to `torch.arange`; the broader bfloat16 gap remains. |
| [pytorch/pytorch#104191](https://github.com/pytorch/pytorch/issues/104191) | Closed | Specific to `torch.embedding`. |

**Verdict:** bfloat16 on MPS will not be fixed. Our float32 workarounds are permanent.

### Memory

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#105839](https://github.com/pytorch/pytorch/issues/105839) | Open | MPS OOM with memory available. If fixed, we could remove `empty_cache()` calls. |
| [pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329) | Open | MPS memory leak during inference. Critical for long-running server. |
| [pytorch/pytorch#145374](https://github.com/pytorch/pytorch/issues/145374) | Open | MPS memory leak in LSTM iterations. |
| [pytorch/pytorch#114096](https://github.com/pytorch/pytorch/issues/114096) | Open | Leak when converting device+type simultaneously via `.to()`. |

### Whisper

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [huggingface/transformers#31408](https://github.com/huggingface/transformers/issues/31408) | Closed | SDPA broke MPS in v4.40.0. Our `attn_implementation="eager"` workaround is for this. Check if later versions fixed SDPA on MPS. |
| [pytorch/pytorch#141774](https://github.com/pytorch/pytorch/issues/141774) | Open | Autocast fails for `scaled_dot_product_attention` on MPS. Related to the SDPA issue above. |
| [pytorch/pytorch#162092](https://github.com/pytorch/pytorch/issues/162092) | Open | Voxtral (Whisper variant) produces gibberish on MPS. |

### Speaker Diarization

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pyannote/pyannote-audio#1337](https://github.com/pyannote/pyannote-audio/issues/1337) | Closed (wontfix) | Wrong timestamps on MPS. If reversed, we could enable MPS for diarization. |
| [pyannote/pyannote-audio#1886](https://github.com/pyannote/pyannote-audio/issues/1886) | Open | Kernel crash on M4 with MPS. |

### MPS Correctness

| Issue | Status | What to do if fixed |
|-------|--------|-------------------|
| [pytorch/pytorch#134534](https://github.com/pytorch/pytorch/issues/134534) | Open | Model returns wrong tokens on MPS vs CPU. Broad correctness concern. |

## Checklist for New Model Loaders

When adding a new inference module that loads a PyTorch model:

1. **Never use MPS.** Our standard device selection is CUDA > CPU. MPS is
   permanently excluded due to kernel-level deadlocks (2026-04-05).
2. **Use `force_cpu_preferred()` as the first check** — respect the operator's
   CPU override.
3. **Test device selection** — add a parametrized test with
   `(force_cpu, cuda_available, mps_available)` that verifies MPS availability
   is ignored and CPU is selected when CUDA is unavailable.
4. **Use `float16` on CUDA, `float32` on CPU** — unless the model specifically
   requires a different dtype.

## Test Coverage

Device selection is covered by parametrized tests that verify MPS is ignored
and CPU is selected when CUDA is unavailable:

| Test | File | What it verifies |
|------|------|-----------------|
| `test_load_whisper_fa_selects_device_and_dtype` | `tests/pipelines/fa/test_fa_inference.py` | Whisper FA: CPU when MPS-only, float16 on CUDA, float32 on CPU |
| `test_load_wave2vec_fa_selects_expected_device` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: CPU when MPS-only |
| `test_load_wave2vec_fa_forces_float32_on_mps` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: no MPS-specific `.float()` needed (MPS excluded) |
| `test_load_whisper_asr_ignores_mps_and_applies_cantonese_overrides` | `tests/pipelines/asr/test_asr_inference.py` | ASR: CPU when MPS available, Cantonese config still applied |
| `TestGpuHasCudaDevice` (4 tests) | `tests/test_worker_serving_mode.py` | CUDA detection helper: force_cpu interaction |
| `TestServingModeSelection` (6 tests) | `tests/test_worker_serving_mode.py` | GPU profile: sequential on CPU, concurrent on CUDA only |
| Speaker device selection test | `tests/pipelines/speaker/test_speaker_inference.py` | Speaker: CUDA > CPU, MPS never selected |
