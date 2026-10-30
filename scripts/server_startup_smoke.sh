#!/usr/bin/env bash
# Start the server with the test-echo runtime and wait for a healthy /health.
#
# The same check in CI (the Rust Integration job) and in the local gate. It
# was an inline workflow block, which the gate-sync checker cannot see, so a
# green gate said nothing about it.
#
# Inputs (environment):
#   BATCHALIGN_BIN     the binary to start (default: target/debug/batchalign3)
#   BATCHALIGN_PYTHON  the interpreter whose environment has batchalign
#                      installed (default: .venv/bin/python)
#
# The server runs under a throwaway BATCHALIGN_HOME on a port the kernel
# picks, so it never touches a developer's real state and never collides with
# a server already listening.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BATCHALIGN_BIN:-$ROOT/target/debug/batchalign3}"
PYTHON="${BATCHALIGN_PYTHON:-$ROOT/.venv/bin/python}"
READY_SECONDS=30

if [[ ! -x "$BIN" ]]; then
  echo "error: no batchalign3 binary at $BIN" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: no Python interpreter at $PYTHON (run make batchalign-develop)" >&2
  exit 1
fi

home="$(mktemp -d)"
port="$("$PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
server_pid=""
# Single-quoted so it reads server_pid when the script exits, not now.
trap '[[ -z "$server_pid" ]] || { kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null; }; rm -rf "$home"' EXIT

BATCHALIGN_HOME="$home" BATCHALIGN_PYTHON="$PYTHON" \
  "$BIN" serve start --foreground --test-echo --port "$port" &
server_pid=$!

for second in $(seq 1 "$READY_SECONDS"); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "error: server exited before becoming healthy" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$port/health" | grep -q '"status":"ok"'; then
    echo "Server healthy after ${second}s"
    exit 0
  fi
  sleep 1
done
echo "error: server not healthy within ${READY_SECONDS}s" >&2
exit 1
