# Chapter 6: Test Strategy and Quality Gates

## The Problem Is Not Raw Test Count

Approximate current test inventory is large:

- Python test functions: about 623
- Rust test attributes: about 1726
- `batchalign/tests`: about 15.8k lines
- `crates/batchalign-app/tests`: about 10.2k lines

The problem is not "there are too few tests" in an absolute sense.

The problem is that too much confidence is being drawn from the wrong tests.

## What the Current Gates Actually Tell You

Current local checks I ran:

- `cargo check --workspace`: passed
- `cargo test --workspace --lib -q`: passed
- `uv run mypy`: passed
- `uv run pytest batchalign/tests -q`: `691 passed, 7 skipped`

Those are useful signals. They say:

- the current source tree compiles
- many unit and library tests pass
- Python typing is currently disciplined

They do not say:

- the public wheel works
- the release workflow works
- the three advertised platforms work
- the true model-loading and worker lifecycle paths are robust
- the integration boundary with `talkbank-tools` is safe

## Current Misallocation

The suite is relatively strong on:

- type conformance
- DTO/schema roundtrips
- isolated algorithm behavior
- many small infrastructure helpers

It is much weaker on:

- artifact install smoke tests
- real CLI entrypoint behavior
- model-loading integration paths
- concurrent worker routing
- recovery after worker crashes/timeouts
- cross-platform startup and process-management behavior
- release workflow and installer verification inside CI

This is exactly the kind of test portfolio that can produce a false sense of
quality while still letting researchers find serious defects first.

## What a Release-Grade Portfolio Should Look Like

### Lane 1: Mandatory Fast Gates

- compile, typecheck, lint
- core unit tests
- protocol and schema checks
- packaging smoke:
  - build artifact
  - install artifact
  - run artifact

### Lane 2: Mandatory Integration Gates

- CLI -> server -> worker happy-path tests
- worker crash/restart/retry paths
- concurrent GPU/shared-worker routing
- minimal real-model inference smoke for each released command family

### Lane 3: Platform Gates

- Windows install + `batchalign3 --help`
- macOS install + `batchalign3 --help`
- Linux install + `batchalign3 --help`

### Lane 4: Scheduled Heavy Gates

- full golden ML suites
- broad corpus parity runs
- long-running concurrency stress
- memory-pressure tests

## Key Principle

Release confidence should come from tests that exercise the actual artifact and
actual operational boundary, not just internal units.

## Action Items

- Add wheel/install/CLI smoke tests to required CI.
- Define one minimal real-inference smoke per released command family.
- Add failure-path integration tests:
  - worker timeout
  - worker crash
  - retry/requeue
  - partial results
- Add concurrency tests that prove correct routing and cleanup under load.
- Re-tier the suite explicitly so heavy tests stay real without making the
  mandatory lanes unusably slow.
