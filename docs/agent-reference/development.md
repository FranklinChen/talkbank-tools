# Development reference

**Last modified:** 2026-09-09 20:37 EDT

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

