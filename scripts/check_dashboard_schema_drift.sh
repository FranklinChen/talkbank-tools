#!/usr/bin/env bash
# The NO-NETWORK half of the dashboard API drift gate: `openapi.json` is
# generated FROM the Rust types, so a doc-comment or a `required` change
# regenerates it and the committed copy goes stale.
#
# Why this exists as its own script. The whole check
# (`check_dashboard_api_drift.sh`) also runs `npx openapi-typesafe`, which
# needs the npm-installed frontend, and it was EXEMPT from the pre-push gate
# for that reason. True of the TypeScript half, irrelevant to this half: the
# schema reuses a development binary whose freshness Cargo has proved.
#
# That exemption is what let a stale `openapi.json` reach main on 2026-08-27.
# Making `HealthResponse::status` required and rewriting two doc comments
# changed the generated schema; every local gate passed and CI failed on
# `Verify dashboard API artifacts`. The same shape as
# `batchalign-ipc-schema-check`, which was exempt for "needs the built binary"
# and is now in the hook for exactly this reason.
#
# Regenerate both halves with `scripts/generate_dashboard_api_types.sh`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${BATCHALIGN_BIN:-}" && -x "${BATCHALIGN_BIN}" ]]; then
    "$BATCHALIGN_BIN" openapi --output openapi.json
else
    # Cargo proves the development binary is current. This shares the ordinary
    # gate's default-feature artifact rather than linking a second binary for a
    # schema whose bytes do not depend on optimization or dashboard embedding.
    cargo build -q -p batchalign
    ./target/debug/batchalign3 openapi --output openapi.json
fi
cp "$ROOT/openapi.json" "$ROOT/frontend/openapi.json"

# Only the tracked copy is diffed: frontend/openapi.json is gitignored, so
# naming it here would look like a check that cannot fail.
if ! git diff --exit-code openapi.json; then
    echo "" >&2
    echo "openapi.json is stale: it is GENERATED from the Rust types and your" >&2
    echo "change altered them. Run:" >&2
    echo "    bash scripts/generate_dashboard_api_types.sh" >&2
    echo "and commit openapi.json, frontend/openapi.json and the regenerated" >&2
    echo "frontend/src/generated/api.ts." >&2
    exit 1
fi
