#!/usr/bin/env bash
# Run the former pre-push checks once, on a stable tree, before opening a push.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=scripts/gate-receipt.sh
source scripts/gate-receipt.sh
gate_begin
cargo metadata --locked --format-version 1 >/dev/null
make lint-shell
make lint-actionlint
make gate-receipts-test
make batchalign-ci-rust
make batchalign-lint-python-source
make batchalign-typecheck-python-source
make batchalign-dev-workflow-test
# The worker suite needs the development environment with a current
# extension (its tests fail rather than skip without one), exactly as the
# batchalign-worker-integration CI job builds it.
make batchalign-develop
make batchalign-test-worker-integration
make batchalign-ipc-schema-check
# Reuse the exact binary whose currency the preceding target established.
BATCHALIGN_BIN="$PWD/target/debug/batchalign3" make batchalign-dashboard-schema-check
# The same binary, with the development environment batchalign-develop made:
# the server smoke and the dashboard suites (mock and real server, about a
# minute), as CI's Rust Integration and Dashboard jobs run them.
BATCHALIGN_BIN="$PWD/target/debug/batchalign3" make batchalign-server-smoke
BATCHALIGN_BIN="$PWD/target/debug/batchalign3" make batchalign-dashboard-e2e-real
make book-check
gate_finish
printf '%s\n' 'Gate passed; matching committed trees may now be pushed.'
