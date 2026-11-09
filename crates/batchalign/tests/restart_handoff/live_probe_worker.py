"""Model-free protocol peer: a refused load, then a live unfinished load."""

import json
import os
import sys
import time
from enum import Enum
from pathlib import Path


class LoadMode(Enum):
    REFUSE = "refuse"
    LIVE = "live"
    WARMUP = "warmup"


state = Path(os.environ["BATCHALIGN_STATE_DIR"])
with (state / "spawns").open("a") as audit:
    audit.write(f"{os.getpid()}\n")


def emit(message: dict[str, object]) -> None:
    print(json.dumps(message), flush=True)


def hold_with_progress() -> None:
    tick = 0
    while True:
        tick += 1
        emit(
            {
                "op": "progress_v2",
                "event": {
                    "scope": "load",
                    "stage": "downloading_model",
                    "measure": {"kind": "download_growth", "grown_kib": tick},
                },
            }
        )
        (state / "load-ticks").write_text(str(tick))
        time.sleep(0.05)


if LoadMode((state / "mode").read_text()) is LoadMode.WARMUP:
    # The first process was retained after its refused load. Only the extra
    # worker spawned by pre-scaling reaches this startup boundary.
    hold_with_progress()

emit(
    {
        "ready": True,
        "pid": os.getpid(),
        "transport": "stdio",
        "runtime": {
            "schema_version": 1,
            "python_version": "3.14.0",
            "python_executable_sha256": "a" * 64,
            "batchalign_package_tree_sha256": "b" * 64,
            "batchalign_core_extension_sha256": "c" * 64,
            "distribution_inventory_sha256": "d" * 64,
        },
    }
)

for line in sys.stdin:
    request = json.loads(line)
    match request["op"]:
        case "capabilities":
            emit(
                {
                    "op": "capabilities",
                    "request_id": request["request"]["request_id"],
                    "response": {
                        "commands": ["transcribe"],
                        "free_threaded": False,
                        "infer_tasks": ["asr"],
                        "engine_versions": {"asr": None},
                        "stanza_catalog": {
                            "kind": "not_installed",
                            "import_error": "model-free peer",
                        },
                    },
                }
            )
        case "ensure_task":
            match LoadMode((state / "mode").read_text()):
                case LoadMode.REFUSE:
                    emit(
                        {
                            "op": "error",
                            "request_id": request["request"]["request_id"],
                            "error": "synthetic load refusal",
                            "kind": "runtime",
                        }
                    )
                case LoadMode.LIVE:
                    hold_with_progress()
                case LoadMode.WARMUP:
                    emit(
                        {
                            "op": "ensure_task",
                            "request_id": request["request"]["request_id"],
                            "response": {
                                "status": "loaded",
                                "task": "asr",
                                "elapsed_s": 0.0,
                            },
                        }
                    )
        case "shutdown":
            break
        case other:
            raise AssertionError(f"unexpected operation: {other}")
