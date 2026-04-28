# Python API

**Last modified:** 2026-03-21 14:47 EDT

Batchalign3 is a **CLI-first** tool. The primary interface is the `batchalign3`
command-line program. Python is used internally as a stateless ML model server —
it is not a public API surface.

## Install

```bash
uv venv .venv
uv pip install batchalign3
```

## CLI is the entry point

All processing is done through the CLI:

```bash
batchalign3 transcribe input/ output/ --lang eng
batchalign3 morphotag input/ output/ --lang eng
batchalign3 align input/ output/ --lang eng
```

For programmatic use from Python, call the CLI via `subprocess`:

```python
import subprocess

subprocess.run([
    "batchalign3", "morphotag",
    "input/", "output/",
    "--lang", "eng",
], check=True)
```

## Provider wire types

`batchalign.providers` re-exports the typed worker payload surface:

- `BatchInferRequest`
- `BatchInferResponse`
- `InferResponse`
- `InferTask`
- `WorkerJSONValue`

Use those types when you build provider adapters or tests around the worker
contract.

## What not to build against

- Do not build new integrations against `batchalign.worker._*`.
- Do not assume undocumented inference modules are stable public API.
- The old `BatchalignPipeline`, `WhisperEngine`, `CHATFile`, `Document`,
  `ParsedChat`, `run_pipeline()`, and `batchalign.compat` surfaces have been
  removed. Use the CLI instead.

## Removed APIs (2026-03-21)

The following Python APIs were removed as part of the pyo3 slimdown. The Rust
server now owns all CHAT manipulation natively, making these redundant:

- **`batchalign.compat`** — BA2 compatibility shim (`CHATFile`, `Document`,
  `BatchalignPipeline`). Was deprecated and emitting warnings. Use the CLI.
- **`batchalign.pipeline_api`** — `run_pipeline()`, `LocalProviderInvoker`,
  `PipelineOperation`. The Rust server handles pipeline orchestration.
- **`batchalign.inference.benchmark`** — `compute_wer()` Python wrapper.
  WER scoring is available via `batchalign3 compare`.
- **`batchalign_core.ParsedChat`** — opaque CHAT handle with mutation methods.
  The Rust server uses `ChatFile` directly; no Python-side handle is needed.

## See also

- [CLI Reference](cli-reference.md)
- [Developer Migration Guide](../migration/developer-migration.md)
