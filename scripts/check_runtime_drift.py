#!/usr/bin/env python3
"""Verify that runtime_constants.toml parses correctly and contains expected keys.

Both Python (batchalign/runtime.py) and Rust (batchalign-types/src/runtime.rs)
read from runtime_constants.toml at import/compile time, so drift between
runtimes is structurally impossible.  This script validates the TOML itself.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TOML_PATH = ROOT / "batchalign" / "runtime_constants.toml"

REQUIRED_SECTIONS = [
    "cmd2task",
    "worker_caps",
    "memory",
    "gpu_heavy_commands",
    "process_commands",
    "command_base_mb",
    "known_engine_keys",
]


def _main() -> int:
    if not TOML_PATH.exists():
        print(f"MISSING: {TOML_PATH}", file=sys.stderr)
        return 1

    data = tomllib.loads(TOML_PATH.read_text())

    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        print(f"Missing TOML sections: {missing}", file=sys.stderr)
        return 1

    # Verify Python runtime loads successfully from it
    import batchalign.runtime as rt

    assert len(rt.Cmd2Task) > 0, "Cmd2Task is empty"
    assert rt.MAX_GPU_WORKERS > 0, "MAX_GPU_WORKERS is zero"
    assert rt.DEFAULT_BASE_MB > 0, "DEFAULT_BASE_MB is zero"
    assert len(rt.GPU_HEAVY_COMMANDS) > 0, "GPU_HEAVY_COMMANDS is empty"
    assert len(rt.PROCESS_COMMANDS) > 0, "PROCESS_COMMANDS is empty"

    print("Runtime constants check passed (TOML valid, Python loads OK).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
