# Development reference

**Last modified:** 2026-11-06 17:45 EST

Read the sections relevant to your task. [AGENTS.md](../../AGENTS.md)
is the canonical policy entry point and resolves workflow conflicts here.
Dated incidents, measurements, versions and paths are historical evidence;
verify current source and live artifacts before relying on them. Inline
repository paths are relative to the repository root unless stated otherwise.

## Build, Test, Lint

`make help` lists the targets; CI workflows live in
`.github/workflows/`. Shell scripts pass `shellcheck` at default
severity (`scripts/lint/shellcheck-all.sh`; wired into the ci-report
gate). Rust tests: prefer scoped `cargo test -p <crate>` (never
`cargo nextest`, which is banned and uninstalled in this workspace).
Developer procedure pages: `book/src/batchalign/developer/`.

## Documentation belongs to the implementation batch

Update the existing owning user and developer book pages as behavior,
architecture or type boundaries change. Explain who owns each prepared input,
which transition consumes it, what completion permits and how refusal or
cancellation preserves accepted work. Mark approved but unfinished designs
explicitly; an implemented type boundary does not establish a deployed feature.

Maintain Mermaid diagrams for meaningful ownership, admission and recovery
flows. Keep their labels and edges aligned with actual types and callers, build
the book, then inspect the rendered diagrams for legible text, connections and
syntax/runtime errors. Retain render evidence with the batch's qualification;
successful Markdown processing alone is insufficient.

For CLI-affecting changes, updating the book's
[CLIARE record](../../book/src/batchalign/developer/cliare-compliance.md) is a
required review step. Preserve the documented legacy pathways and distinguish
the current source from the exact candidate last measured or executed.

## Releases and Versioning

No release semver is baked in: `batchalign3 version` reports a
`git describe`-based BUILD identity assembled in
`crates/batchalign/build.rs`; staleness is judged by build identity,
never semver. Releases are GitHub Releases (uv-bootstrap installer +
abi3 wheels), **never PyPI**. Release workflow:
`.github/workflows/batchalign-release.yml` (tag push or
workflow_dispatch with dry_run).

## Debugging Recipes

Canonical: `book/src/batchalign/developer/tracing-and-debugging.md`
(py-spy over workers) and `cpu-profiling.md` (tokio-console,
`debug-runtime` feature). Do not restate the recipes here.

