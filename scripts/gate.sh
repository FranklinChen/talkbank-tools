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
make batchalign-ipc-schema-check
# Reuse the exact binary whose currency the preceding target established.
BATCHALIGN_BIN="$PWD/target/debug/batchalign3" make batchalign-dashboard-schema-check
make book-check
gate_finish
printf '%s\n' 'Gate passed; matching committed trees may now be pushed.'
