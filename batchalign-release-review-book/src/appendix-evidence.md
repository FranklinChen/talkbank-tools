# Appendix: Evidence and Reproductions

## Commands Run

The following checks were run during this review:

```text
cargo check --workspace
cargo test --workspace --lib -q
uv run mypy
uv run pytest batchalign/tests -q
```

Observed results:

- `cargo check --workspace`: passed with warnings
- `cargo test --workspace --lib -q`: passed
- `uv run mypy`: passed
- `uv run pytest batchalign/tests -q`: `691 passed, 7 skipped`

## Important Point About Those Green Results

These green results do not contradict the audit findings.

They demonstrate that:

- the source tree currently compiles
- many unit tests pass
- typing discipline exists

They do not demonstrate:

- a correct public wheel artifact
- a trustworthy release workflow
- real cross-platform support
- a sufficient integration and failure-path test portfolio

## Broken Wheel Reproduction

I reproduced the packaging failure from a clean exported source snapshot:

1. export tracked source with `git archive`
2. provide a sibling `talkbank-tools` checkout because the repo requires it
3. build the wheel
4. inspect the wheel contents
5. install into a clean `uv tool` sandbox
6. run `batchalign3 --help`

Observed facts:

- the clean wheel contained `batchalign/_bin/.gitignore`
- the clean wheel did not contain `batchalign/_bin/batchalign3`
- the installed command failed with:

```text
batchalign3 CLI binary not found. Reinstall batchalign3 or, in a source checkout, run `cargo build -p batchalign-cli`.
```

That is the strongest empirical finding in this review.

## Key Source References

Packaging and entrypoint:

- `pyproject.toml:83-106`
- `batchalign/_cli.py:52-90`
- `.github/workflows/release.yml:44-63`

CI trigger weakness:

- `.github/workflows/test.yml:3-7`

Cross-repo release coupling:

- `crates/batchalign-chat-ops/Cargo.toml:22-24`
- `README.md:217-223`

Release-state contradictions:

- `batchalign/version:1-3`
- `installers/README.md:51-57`
- `book/src/developer/api-stability.md:25-39`

Cross-platform/process-lifecycle assumptions:

- `worker/handle/mod.rs:191-204`
- `worker/handle/mod.rs:1026-1071`
- `cli/daemon.rs:477-485`
- `cli/serve_cmd.rs:211-220`

Status/observability approximation:

- `worker/pool/status.rs:1-77`

Representative silent-evidence loss:

- `staging/orchestrator.rs:170-174`
