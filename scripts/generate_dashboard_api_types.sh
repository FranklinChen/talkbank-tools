#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
if [[ -n "${BATCHALIGN_BIN:-}" ]]; then
    "$BATCHALIGN_BIN" openapi --output openapi.json
else
    cargo run -q -p batchalign --no-default-features --features binary-entry,server -- \
        openapi --output openapi.json
fi

cp "$ROOT/openapi.json" "$ROOT/frontend/openapi.json"

cd "$ROOT/frontend"
npx --no-install openapi-typesafe openapi.json -o src/generated/api.ts
npm run generate:decoders

echo "Regenerated dashboard API schema and TypeScript types."
