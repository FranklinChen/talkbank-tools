# Batchalign agent guidance

**Last modified:** 2026-09-09 20:36 EDT

Canonical guidance for all coding agents. `CLAUDE.md` imports this file. Read
applicable nested guidance in the affected crates and frontend, plus the
references relevant to the task.

## Scope and safety

- This repository owns the Batchalign ML/audio pipeline, Rust orchestration,
  PyO3 bridge, Python workers and dashboard. Chatter owns generic CHAT parsing,
  validation, model and transforms. Consume its published pinned dependencies;
  do not copy CHAT primitives or commit local dependency patches.
- All committed content is public. Do not add private identities, hosts, paths,
  corpus operations or incident records. Keep examples abstract and reproducible.
- Preserve the existing checkout, unrelated work and published history. Push,
  release and deploy require explicit authorization, which persists within its
  scope. Squash unpublished commits since the last push. No force-push or hook
  bypass. Routine implementation, review and verification need no repeated
  approval ceremony.
- Runtime freshness is the embedded build identity, not semver or an old install.
  Releases use the GitHub release workflow and uv-bootstrap/abi3 artifacts,
  never PyPI. Verify the artifact and live runtime for an authorized deployment.
- Do not disrupt unrelated worker jobs or silently change engine selection.
  Preserve cache provenance and compatibility checks; reuse requires matching
  inputs/options and demonstrated compatibility with the recorded producing
  implementation. A cache hit alone is not a correctness certificate. Read runtime/worker guidance before changes.
- Preserve Rust orchestration and typed worker/PyO3 contracts. Use `uv` for Python
  tooling. Regenerate affected wire/API artifacts when producers change.
  Operations over about a second must report progress through `progress_v2`.

## Design and verification

Use typestate, ownership and validated constructors to express actual invariants,
including pairing a plan with its inputs. Return typed errors in pipeline,
runner, store, FFI and background code; do not hide failures with defaults.
Chatter adjudicates CHAT validity; fix parser versus data defects at their real
source. Morphology output is UD, not legacy CLAN-mor syntax.

Retain validating-constructor, policy, wire-format and external-boundary tests.
A proof type does not justify deleting its own admission tests. Exercise the
actual failing boundary: daemon HTTP, worker engine loading, PyO3 requests, CLI
subprocesses or model transforms. Review the final diff before committing;
apply proportional checks to documentation/configuration-only edits.

Use `make help`, the current Makefile and CI workflows for supported gates.
Prefer scoped `cargo test -p <crate>`; `cargo nextest` is prohibited here.
Do not run concurrent cargo commands in one workspace. Run required final checks
on final content and reuse receipts only while inputs/tooling remain unchanged.
Shell scripts must pass default-severity shellcheck. Preserve release gates.

## Task references

| Task | Read |
| --- | --- |
| Dependencies, crate ownership and morphology | [Architecture](docs/agent-reference/architecture.md) |
| Typestate, error handling and real-boundary tests | [Design](docs/agent-reference/design.md) |
| Builds, releases and debugging | [Development](docs/agent-reference/development.md), `book/src/batchalign/developer/` |
| Pipeline/runtime/worker/cache work | `crates/batchalign/CLAUDE.md` |
| Python bridge or dashboard | `crates/batchalign-pyo3/CLAUDE.md`, `frontend/CLAUDE.md` |

This entry point resolves workflow conflicts in the references. Historical
examples are not current dependency versions or live runtime evidence. Generic
skills cannot add redundant approval pauses or authorize publishing.
