# Introduction

This review is a release-readiness audit of `batchalign3` as it exists in the
repository on 2026-04-02.

It is not a quick code-style pass. The goal is to answer a harder question:

Can this software be trusted as a publicly released research tool that must
stay stable and maintainable for the next 15 years?

## Scope

This review covers:

- Rust control-plane crates under `crates/`
- Python worker/runtime code under `batchalign/`
- PyO3 packaging/runtime glue under `pyo3/`
- frontend and desktop delivery surfaces
- test, CI, packaging, installer, and release workflows
- the practical dependency boundary with `talkbank-tools`

This review does not attempt a full correctness audit of every linguistic
algorithm. Instead, it focuses on release risk, operational reliability,
architectural maintainability, and the specific failure modes that make public
release dangerous even when many unit tests are green.

## Method

The review combined:

- repository structure and code inspection
- build and gate execution:
  - `cargo check --workspace`
  - `cargo test --workspace --lib -q`
  - `uv run mypy`
  - `uv run pytest batchalign/tests -q`
- packaging reproduction in a clean exported source snapshot
- workflow, installer, and metadata audit
- maintainability signals such as oversized modules, dead scripts, and boundary drift

## Top-Line Verdict

`batchalign3` is not ready for a trustworthy public release yet.

The main reason is not that the codebase is hopeless. The main reason is that
the release contract is still incoherent:

- the shipped wheel path is currently broken in a clean-source build
- CI is not enforcing normal development changes
- release/version/license metadata are inconsistent
- the repository is not independently releasable because it still depends on
  unreleased `talkbank-tools` path dependencies
- test volume is high, but the test portfolio is badly allocated relative to
  the failure modes that matter most
- cross-platform support is asserted far more strongly than it is demonstrated

The right response is not a ground-up rewrite of all algorithms. The right
response is a staged stabilization program that rewrites or replaces the
release boundary, repository boundary, and test strategy first, then pays down
selected control-plane complexity.
