# Models Training Runtime ADR

## Status

Accepted.

## Context

`batchalign3 models ...` currently delegates to the Python training runtime:

```text
python -m batchalign.models.training.run ...
```

The CLI/server control plane has migrated to Rust, but model training still
depends on Python-first ML stacks and training code paths.

## Decision

Keep `models` as a Python bridge for now, with explicit boundaries:

1. Rust owns argument parsing, UX, and process orchestration.
2. Python owns training/inference library integration for model training.
3. Interpreter resolution must remain uv-friendly:
   `BATCHALIGN_PYTHON` -> `VIRTUAL_ENV` -> `python3`.

## Rationale

1. Training-specific dependencies are Python-native and already production
   validated.
2. Rewriting training loops in Rust now would be high-risk, low-ROI versus
   finishing CLI/server/runtime migration.
3. The bridge keeps migration momentum while avoiding duplicate training stacks.

## Consequences

1. Shipping still requires a compatible Python runtime for `models`.
2. CLI/server/runtime operations remain Rust-first.
3. Migration accounting treats `models` as an intentional Python-core island
   rather than accidental legacy code.

## Exit Criteria For Future Rust Port

Revisit only when all are true:

1. A Rust training stack is selected and benchmarked with parity targets.
2. Feature parity test corpus exists for training outputs.
3. Operational benefits (startup, packaging, observability, maintenance)
   clearly exceed migration cost.
