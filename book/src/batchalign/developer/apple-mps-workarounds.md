# Apple MPS Workarounds

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

## MPS Exclusion Policy

**MPS is excluded by default.** ASR and forced-alignment loaders select CPU
when CUDA is unavailable unless the operator explicitly enables `--allow-mps`.
Local speaker engines exclude MPS even with that opt-in. `--force-cpu`
overrides accelerator selection. The opt-in is described below.

### Why: confirmed unsafe behavior

FA workloads on MPS can produce AGXG14X shutdown stalls on Apple
Silicon hardware. The failure path goes through
`batchalign_core::worker_fa_exec::run_wave2vec_like` →
`at::mps::MPSStream::executeMPSGraph`, with very large memory
footprint, low free disk, `ffmpeg` `No space left on device` failures
while building media-cache WAVs and FA temp PCM, and follow-on shared
GPU worker crashes.

That is enough to treat MPS as unsafe by default.

There is no user-space mitigation for the AGX deadlock path. You cannot:
- Time out the operation (kernel mutex, not user-space)
- Kill the stuck process (unkillable zombie in kernel sleep)
- Break the compute into smaller chunks (deadlock can trigger on normal 5-30s chunks)
- Use a subprocess watchdog (PyTorch MPS silently hangs or SIGSEGVs in forks:
  [pytorch/pytorch#178037](https://github.com/pytorch/pytorch/issues/178037))

MPS exclusion alone is not sufficient: long FA jobs can fill both
`~/Library/Application Support/batchalign3/media_cache/` and
`/var/folders/.../fa_v2/audio/` if internal free space is already low.
Apple-performance work must therefore include cache cleanup and
temp-space admission control, not just device selection.

### Ecosystem evidence

No major ML project defaults to MPS:

| Project | MPS status |
|---------|-----------|
| **OpenAI Whisper** | PR [#382](https://github.com/openai/whisper/pull/382) to enable MPS was never merged, users reported crashes, MPS slower than CPU |
| **faster-whisper** | MPS not supported at all (`ValueError: unsupported device mps`) |
| **pyannote-audio** | MPS produces wrong timestamps ([#1337](https://github.com/pyannote/pyannote-audio/issues/1337), closed wontfix); kernel crashes on M4 ([#1886](https://github.com/pyannote/pyannote-audio/issues/1886)) |
| **whisper.cpp** | Uses Metal API directly, bypassing PyTorch MPS entirely |
| **PyTorch Lightning** | MPS marked "experimental" |

Representative open PyTorch MPS issues:
- [#178497](https://github.com/pytorch/pytorch/issues/178497): `sum`, `mean`, `count_nonzero` give wrong results, confirmed by PyTorch team as originating in Apple's MPS framework. Closed 2026-05-14 as completed; fix landed on `main` after the v2.12.0 branch cut, so it ships in v2.13, **not** v2.11 or v2.12.
- [#179352](https://github.com/pytorch/pytorch/issues/179352): `scaled_dot_product_attention` incorrect for large batches (cosine similarity 0.49 vs CPU)
- [#154329](https://github.com/pytorch/pytorch/issues/154329): memory leak (~1 MB/sec) confirmed on M4 Max/Studio
- [#144634](https://github.com/pytorch/pytorch/issues/144634): `torch.mps.synchronize()` hangs on error, Apple engineer acknowledged, still open

Apple has not publicly acknowledged the AGXG14X kernel deadlock.

### Deadlock and hang categories

Two PyTorch issues carry the dual labels `module: mps` + `module: deadlock`:

1. [pytorch/pytorch#144634](https://github.com/pytorch/pytorch/issues/144634),
   `torch.mps.synchronize()` hangs on shader fault.
   Filed by PyTorch maintainer `@malfet`: "few `attempt[s]` to reproduce the same
   resulted in **system hang**." Apple engineer acknowledged, still open.

2. [pytorch/pytorch#162872](https://github.com/pytorch/pytorch/issues/162872),
   `Event.synchronize()` deadlocks before `elapsed_time()`. Reproduced on
   M4 Pro. Simple timing code hangs permanently.

Additionally:

- [pytorch/pytorch#178037](https://github.com/pytorch/pytorch/pull/178037),
  MPS silently hangs or SIGSEGVs in forked subprocesses. Unlike CUDA, MPS
  had no `_lazy_init()` check in forked children. This rules out the
  "watchdog subprocess" mitigation strategy.

- The MLX project documented macOS GPU watchdog kills at
  [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267): the
  error `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` fires when a
  Metal command buffer blocks WindowServer compositing for too long. An
  undocumented env var `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` exists but is
  unsupported; MLX maintainers marked this "wontfix."

### Silent correctness failures

The most insidious MPS problem is not crashes but **silently wrong results**:

- A class of issues are labeled `module: mps` + `module: correctness (silent)`.
- Reductions (`sum`, `mean`, `nansum`, `trace`, `count_nonzero`) give wrong
  results when called in rapid succession, confirmed by PyTorch team as
  originating in Apple's MPS framework itself, not PyTorch
  ([#178497](https://github.com/pytorch/pytorch/issues/178497))
- `scaled_dot_product_attention` returns incorrect output for large
  batch × sequence products, cosine similarity drops to 0.49 vs CPU
  ([#179352](https://github.com/pytorch/pytorch/issues/179352))
- "Catastrophically wrong gradients" (1,000×-100,000× too large) when total
  elements exceed 32K
  ([#177116](https://github.com/pytorch/pytorch/issues/177116))
- `torch.multinomial` crashes with SIGSEGV on MPS for larger tensors
  ([#178579](https://github.com/pytorch/pytorch/issues/178579), closed 2026-04-16 as completed; fix landed on `main` after the v2.12.0 branch cut, so it ships in v2.13, **not** v2.11 or v2.12)
- `uint16/uint32/uint64` binary ops produce garbage values
- `ComplexFloat` dtype not supported at all

For an ASR/FA pipeline, silent attention bugs mean **wrong transcriptions and
wrong alignments with no error**. This is arguably worse than a crash.

### Performance on Apple Silicon

Academic benchmarking ([arxiv:2511.05502](https://arxiv.org/abs/2511.05502))
found that for LLM inference on Apple Silicon:
- **MLX** achieves highest throughput (~230 tok/s)
- **llama.cpp** excels for single-stream inference
- **PyTorch MPS** ranked last among 5 frameworks tested (~7-9 tok/s)
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
- macOS Sequoia added native non-contiguous tensor support in Metal,
  fixing a class of silent correctness bugs. Later macOS releases
  introduced Metal 4 with `MTLTensor`, but no stability improvements
  for ML compute workloads were announced.
- The GPU watchdog timer that kills long compute is by design; there is no
  official mechanism to disable it.

### Current device selection per module

| Module | Device order | Dtype |
|--------|--------------|-------|
| `fa.py` (Whisper FA) | CUDA → CPU | CUDA: float16; CPU: float32 |
| `fa.py` (Wave2Vec FA) | CUDA → CPU | float32 |
| `asr.py` (Whisper ASR) | CUDA → CPU | CUDA: float16; CPU: float32 |
| `speaker.py` | CUDA → CPU |, |
| `_main.py` (serving) |: | GPU profile = concurrent on CUDA only; sequential on CPU |

### Worker concurrency impact

GPU-profile workers previously used `ThreadPoolExecutor(gpu_thread_pool_size)`
for concurrent inference, relying on PyTorch releasing the GIL during GPU
kernels. On CPU, this causes thread oversubscription: each thread's PyTorch ops
use all cores via OpenMP, so 4 threads × 24 cores = 96 threads
fighting for 24 cores on an M3 Ultra-class CPU.

Fix: GPU-profile workers now serve sequentially on CPU (one request at a time,
all cores per request). `gpu_thread_pool_size` in `server.yaml` takes effect
only when CUDA is available.

### Implications for platform strategy

Without the MPS opt-in, Apple Silicon machines run these PyTorch ASR and FA
models on CPU. Cache/temp-space pressure can dominate host behavior on long
jobs. This policy does not describe non-PyTorch backends such as whisper.cpp.

The code is CUDA-ready: all model loaders select CUDA first when
available, `gpu_thread_pool_size` activates, and the worker profiles
are designed for GPU concurrency. A Linux + NVIDIA-GPU deployment
restores GPU acceleration, enables concurrent GPU serving, supports
float16/bfloat16 inference, runs Pyannote speaker diarization on GPU,
and eliminates the MPS-related complexity below.

## Hardware Limitations

Metal (Apple's GPU framework) does **not** support:

| Type | Status | PyTorch behavior |
|------|--------|-----------------|
| **bfloat16** | Not in Metal spec | Crashes, wrong results, or `TypeError` depending on operation |
| **float64** | Not in Metal spec | `TypeError: Cannot convert Double to MPS` |
| **int64** | Not in Metal spec | Crashes on some ops (e.g. `abs_out_mps`) |
| **complex128** | Not in Metal spec | Conversion failure |

These are hardware/framework limitations, not PyTorch bugs. No fix is expected.

## Per-Module Workarounds

### ASR: Whisper (`inference/asr.py`)

```python
if device.type == "mps":
    asr_dtype = torch.float32   # not bfloat16
```

Whisper ASR uses `bfloat16` on CUDA for speed. On MPS, this crashes with Metal
assertion failures. We force `float32`. A second fallback path also forces
`float32` on MPS for older transformers versions that don't accept `bfloat16`
at all.

The HuggingFace Transformers Whisper pipeline requires
`attn_implementation="eager"` on MPS, the SDPA attention path broke MPS in
transformers v4.40.0.

### Forced Alignment: Whisper FA (`inference/fa.py`)

```python
if device.type == "mps":
    torch_dtype = torch.float32   # not float16
```

Same pattern as ASR. Whisper FA uses `float16` on CUDA, `float32` on MPS/CPU.

### Forced Alignment: Wave2Vec FA (`inference/fa.py`)

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

### Speaker Diarization (`inference/speaker.py`)

```python
return "cuda" if torch.cuda.is_available() else "cpu"
```

MPS is excluded from diarization because:

- **Pyannote on MPS** produces wrong timestamps
  ([pyannote/pyannote-audio#1337](https://github.com/pyannote/pyannote-audio/issues/1337),
  closed as wontfix). Kernel crashes also reported on M4
  ([#1886](https://github.com/pyannote/pyannote-audio/issues/1886)).
- **NeMo** is CUDA-only by design, no MPS support at all.

The device selector (`_device_for_speaker_runtime`) returns `"cuda"` or
`"cpu"`, never `"mps"`.

### Device Policy (`batchalign/device.py`)

The `BATCHALIGN_FORCE_CPU` environment variable (or `DevicePolicy(force_cpu=True)`)
forces all model loaders onto CPU. This is the escape hatch when MPS causes
problems that dtype coercion alone can't fix.

### The MPS opt-in (`--allow-mps`, added 2026-07-17)

MPS is excluded by default (the April 2026 AGX kernel deadlock hard-stalled
machines, with no known driver fix), but the failure proved RARE in a year of
fleet use, so the brave can opt in with a real CLI flag:

```bash
batchalign3 --allow-mps align ...
```

The flag threads end to end exactly like `--force-cpu`: CLI to dispatch to
the daemon (whose daemon.json records the resolved value, so a running
daemon restarts when the flag changes) to each worker's own `--allow-mps`
argument to the typed `DevicePolicy`. `allow_mps: true` in `server.yaml`
sets it for a standing server. (The `BATCHALIGN_ALLOW_MPS=1` environment
variable exists only as the library-level seam for embedding callers; the
flag is the interface.)

Selection order becomes CUDA > MPS > CPU; `--force-cpu` conflicts with and
overrides it. What the opt-in does and does not change:

- Model dtypes stay **float32** on MPS (every loader's non-CUDA branch):
  fp16-on-MPS caused the Feb/Mar 2026 corruption incidents and remains off.
- The **speaker (diarization) stage never uses MPS**, opt-in or not:
  Pyannote emits wrong timestamps on MPS (upstream wontfix), which is a
  correctness bug, not a stability trade.
- A one-per-process warning is logged when MPS actually engages, naming the
  rare-stall risk, so a wedged machine is not a mystery.
- The flag is per-process and inherited by spawned workers, same as
  `BATCHALIGN_FORCE_CPU`. Do not set it on unattended fleet servers; it is
  for interactive/supervised runs where a reboot is cheap.

## Memory Issues on MPS

MPS has well-documented memory management problems:

- **Memory leaks** during inference: usage climbs steadily, eventually OOM
  ([pytorch/pytorch#154329](https://github.com/pytorch/pytorch/issues/154329),
  [#145374](https://github.com/pytorch/pytorch/issues/145374))
- **OOM with memory available**: MPS cache doesn't release when it should
  ([pytorch/pytorch#105839](https://github.com/pytorch/pytorch/issues/105839))
- **`sysinfo::available_memory()`** on macOS undercounts, reports
  only free + purgeable, missing reclaimable file cache. On a
  Fleet-tier host (≥ 256 GB RAM, heavy I/O), this can underreport by
  tens of GB. No fix exists because macOS doesn't expose a
  `MemAvailable` equivalent like Linux.

**Mitigations:**
- `torch.mps.empty_cache()`: call periodically during long-running inference
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`: disables MPS memory limit (risks
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
| [pytorch/pytorch#141864](https://github.com/pytorch/pytorch/issues/141864) | Closed (won't fix) | N/A, Metal lacks native bfloat16. Would require Apple hardware/firmware change. |
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
| [pytorch/pytorch#141774](https://github.com/pytorch/pytorch/issues/141774) | Closed 2024-11-29 (completed) | Autocast fails for `scaled_dot_product_attention` on MPS. Fixed upstream; our `attn_implementation="eager"` workaround may no longer be required when MPS is re-evaluated. |
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

1. **Use the shared device policy.** ASR/FA loaders call
   `resolve_inference_device`; the default is CUDA > CPU, and MPS requires
   explicit opt-in. Speaker loaders keep their separate MPS exclusion.
2. **Respect the CPU override**: `DevicePolicy.force_cpu` takes precedence
   over accelerator availability and MPS opt-in.
3. **Test device selection**: add a parametrized test with
   `(force_cpu, allow_mps, cuda_available, mps_available)` covering the default,
   opt-in and CPU override. Speaker tests must retain exclusion under opt-in.
4. **Keep each model's CUDA dtype and use `float32` on CPU/MPS**, unless the
   model specifically requires a different dtype.

## Test Coverage

The default-policy cases below verify CPU selection when only MPS is
available; they do not mean the explicit opt-in is unavailable:

| Test | File | What it verifies |
|------|------|-----------------|
| `test_load_whisper_fa_selects_device_and_dtype` | `tests/pipelines/fa/test_fa_inference.py` | Whisper FA: CPU when MPS-only, float16 on CUDA, float32 on CPU |
| `test_load_wave2vec_fa_selects_expected_device` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: CPU when MPS-only |
| `test_load_wave2vec_fa_forces_float32_on_mps` | `tests/pipelines/fa/test_fa_inference.py` | Wave2Vec FA: no MPS-specific `.float()` needed (MPS excluded) |
| `test_load_whisper_asr_ignores_mps_and_applies_cantonese_overrides` | `tests/pipelines/asr/test_asr_inference.py` | ASR: CPU when MPS available, Cantonese config still applied |
| `TestGpuHasCudaDevice` (4 tests) | `tests/test_worker_serving_mode.py` | CUDA detection helper: force_cpu interaction |
| `TestServingModeSelection` (6 tests) | `tests/test_worker_serving_mode.py` | GPU profile: sequential on CPU, concurrent on CUDA only |
| Speaker device selection test | `tests/pipelines/speaker/test_speaker_inference.py` | Speaker: CUDA > CPU, MPS never selected |
