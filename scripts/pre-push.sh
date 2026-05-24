#!/usr/bin/env bash
# Pre-push hook: fast local checks that mirror CI gates.
# Install: make install-hooks
#
# Coverage goal: catch anything the GitHub "main CI" workflow would flag
# on a push to main, without running long test suites. If a CI job can
# fail purely because of committed content (not runtime behavior), this
# hook must cover it.
set -euo pipefail

echo "==> pre-push: fmt check"
cargo fmt --all -- --check
cd spec/tools && cargo fmt --all -- --check && cd ../..

echo "==> pre-push: affected compile check"
cargo run -q -p xtask -- affected-rust check

echo "==> pre-push: parser guardrail"
scripts/check-errorsink-option-signatures.sh

# Mirrors the "Generated Artifacts Up To Date" CI job. This is the
# gate that caught commit 8b483edef (E316 spec added, docs/errors/index.md
# not regenerated) only after push.
echo "==> pre-push: generated artifacts up to date"
make generated-check

# Mirrors the "Fuzz Smoke Test" CI job's workspace discovery step.
# Cheap (no compile), catches manifests that drift out of
# workspace.members / workspace.exclude.
echo "==> pre-push: fuzz workspace isolation"
make fuzz-check

# Mirrors the "TalkBank Toolchain mdBook" CI workflow. mdbook's
# linkcheck2 backend exhaustively verifies every relative link
# against SUMMARY.md — catching SUMMARY-unreachable targets like
# the 2026-05-22 batchalign/introduction.md regression that broke
# CI after a 68-commit squash push. Requires mdbook + mdbook-
# linkcheck + mdbook-mermaid on PATH (make book-check enforces).
echo "==> pre-push: mdBook build + linkcheck"
make book-check

if [[ "${TALKBANK_PRE_PUSH_CLIPPY:-0}" == "1" ]]; then
  echo "==> pre-push: affected clippy"
  cargo run -q -p xtask -- affected-rust clippy
fi

echo "✓ All pre-push checks passed"
