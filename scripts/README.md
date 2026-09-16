# scripts/

**Status:** Current
**Last updated:** 2026-09-16 09:28 EDT

This directory is a shared toolbox for the batchalign3 workspace: generated
artifact refreshes, drift checks, targeted smoke tests, diagnostics, and fixture
prep. For normal contributor work, start with repo-native entrypoints first:

- `make help` for supported top-level tasks
- `cargo run -q -p xtask -- help` for lower-level Rust maintenance helpers

Most scripts here are wrappers or focused helpers behind those entrypoints. Use
the index below when you need the direct script.

The CHAT core (grammar, spec, tree-sitter parser, data model, validation, CLAN)
lives in the chatter repo now; the scripts that maintained it (symbol/node-type
generation, parser guardrails, CHAT-manual anchors, CHECK-parity corpora,
corpus-age fixers) were removed when this workspace became batchalign3-only.

## Start here

| I want to... | Prefer this entrypoint | Notes |
|---|---|---|
| Run the normal contributor gates | `make check`, `make test`, `make verify`, `make ci-local`, `make ci-full` | Prefer these over ad hoc script chains. |
| Run the batchalign Rust / Python gates | `make batchalign-ci-rust`, `make batchalign-ci-python` | Compile + tests for the imported Batchalign crates and wheel. |
| Install the repo's local git guardrail | `make install-hooks` | Installs `scripts/pre-push.sh`. |
| Verify imported Batchalign generated surfaces | `make batchalign-dashboard-api-check`, `make batchalign-runtime-check`, `bash scripts/check_ipc_type_drift.sh` | Use the first two via `make`; IPC schema drift is still a direct script. |
| Run a focused dashboard smoke or E2E helper | `bash scripts/run_react_dashboard_smoke.sh` | Builds the frontend dependencies it needs. |
| Prepare a minimal repro fixture from real CHAT + media | `python3 scripts/trim_chat_audio.py ...` or `python3 scripts/prepare_corpus_media_fixture.py ...` | Prefer these over hand-editing fixtures. |

## Taxonomy

### Generators and regeneration

| Script | What it does | Canonical entrypoint |
|---|---|---|
| `generate_dashboard_api_types.sh` | Regenerates dashboard API artifacts from the Batchalign server surface. | Called by `make batchalign-dashboard-api-check` and dashboard smoke scripts |
| `generate_ipc_types.sh` | Regenerates Python IPC models from Rust JSON Schema. | Direct script after a Rust/Python boundary change |
| `generate_stanza_language_table.py` | Refreshes the hardcoded Stanza support table from upstream resource data. | Direct script after Stanza/resource updates |

### Drift checks and guardrails

| Script | What it checks | Canonical entrypoint |
|---|---|---|
| `check_dashboard_api_drift.sh` | Verifies generated dashboard API artifacts are up to date. | `make batchalign-dashboard-api-check` |
| `check_ipc_type_drift.sh` | Verifies IPC JSON Schema artifacts match committed sources. | Direct script after IPC type changes |
| `check_model_pin_drift.py` | Checks every pinned model in the manifest against its upstream (Hugging Face commits, ModelScope tag existence). Needs the network; queries public APIs only. Exits 0 match, 1 moved, 2 could not check. | Direct script when reviewing whether a pinned model should be upgraded |
| `check_runtime_drift.py` | Verifies shared runtime constants still parse and expose expected keys. | `make batchalign-runtime-check` |
| `run-drift-probes.sh` + `generate_drift_report.py` | Runs the Stanza drift probes and renders a report for inspection (never fails the caller). | Direct scripts for targeted investigation |
| `lint/shellcheck-all.sh` | Runs shellcheck (strictest severity) over every tracked shell script. | `bash scripts/lint/shellcheck-all.sh` (also a CI job) |

### Smoke tests, test helpers, and diagnostics

| Script | What it helps with | Preferred use |
|---|---|---|
| `pre-push.sh` | Runs CI's own target (`make batchalign-ci-rust`) plus the drift checks, then chains to an untracked `.git/hooks/pre-push.local` if one exists. | Install via `make install-hooks`; re-run it after anything else writes `.git/hooks/pre-push` |
| `pre-commit-check.sh` | Manual fmt + clippy + build + unit-test pass before committing. | Direct script (overlaps `make ci-full`) |
| `run_react_dashboard_smoke.sh` | Frontend/dashboard smoke and E2E flow. | Direct script |
| `build_react_dashboard.sh` | Builds the dashboard bundle into the configured target directory. | Direct script |
| `test-bg.sh` + `test-bg-status.sh` | Fire-and-forget background test runner plus status view. | Direct scripts for long local runs |
| `test-affected.py` | Computes a test subset from changed files and `affects:` annotations. | Direct script for targeted local testing |
| `choose-test-concurrency.sh` | Picks a safer test parallelism based on memory. | Support helper for local test workflows |
| `test_lazy_profile_e2e.sh` | Focused E2E check for LazyProfile worker behavior. | Direct script |
| `compare_stock_batchalign.py` + `stock_batchalign_harness.py` | Compares this workspace's output against a stock Batchalign baseline. | Direct diagnostics for output regressions |

### Fixture prep

| Script | What it does | Preferred use |
|---|---|---|
| `trim_chat_audio.py` | Trims a CHAT file and matching audio to a focused utterance range. | Minimal repro fixture creation |
| `prepare_corpus_media_fixture.py` | Copies a CHAT file plus matching media, then delegates trimming. | Fixture prep when media lives on a remote host |

### Book and docs maintenance

`scripts/doc-audit/` holds focused helpers for the mdBook under `book/`:
`triage_mdbook_test_failures.py` classifies failing doctest fences,
`apply_fence_rewrites.py` applies the resulting rewrites, and
`fix_header_convention.py` bulk-fixes the doc date/status header convention.

## Rules of thumb

- Prefer the `make` or `xtask` entrypoint when one exists; it captures the
  supported workflow and usually composes multiple lower-level steps.
- Reach for direct scripts when you need a focused generator, a narrow
  diagnostic, or a fixture-prep helper that does not have a stable top-level
  target.
- If you add a new script that contributors might reasonably discover by
  browsing, add it to this index in the matching category.
- CHAT-format tooling belongs in the chatter repo, not here.
