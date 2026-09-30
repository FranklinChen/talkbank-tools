# affects: batchalign/worker/_main.py
# affects: batchalign/worker/_types.py
"""Tests for typed worker bootstrap runtime resolution."""

from __future__ import annotations

import os

from batchalign.worker._main import build_worker_bootstrap_runtime, parse_worker_args
from batchalign.worker._types import InferTask

# Flags the Rust pool always passes when it launches a stdio worker: how the
# process serves requests, and the server it watches so it exits with it.
_REQUIRED_LAUNCH_ARGS = [
    "--serving",
    "sequential",
    "--supervisor-pid",
    str(os.getpid()),
]


def test_worker_bootstrap_runtime_resolves_boundary_inputs() -> None:
    """CLI args and boundary env should become one typed bootstrap object."""

    args = parse_worker_args(
        [
            "--task",
            "asr",
            "--lang",
            "eng",
            "--num-speakers",
            "2",
            "--engine-overrides",
            '{"asr":"whisper"}',
            "--force-cpu",
            *_REQUIRED_LAUNCH_ARGS,
        ]
    )

    bootstrap = build_worker_bootstrap_runtime(
        args,
        environ={
            "HOME": "/tmp/bootstrap-home",
            "BATCHALIGN_REV_API_KEY": " from-rust ",
        },
    )

    assert bootstrap.task is InferTask.ASR
    assert bootstrap.lang == "eng"
    assert bootstrap.num_speakers == 2
    assert bootstrap.engine_overrides == {"asr": "whisper"}
    assert bootstrap.device_policy.force_cpu is True
    assert bootstrap.revai_api_key == "from-rust"


def test_allow_mps_flag_reaches_device_policy() -> None:
    """--allow-mps must land in the typed DevicePolicy (the MPS opt-in
    travels as a real CLI flag end to end, never an env hack)."""
    args = parse_worker_args(
        ["--task", "asr", "--lang", "eng", "--allow-mps", *_REQUIRED_LAUNCH_ARGS]
    )
    runtime = build_worker_bootstrap_runtime(args, environ={})
    assert runtime.device_policy.allow_mps is True
    assert runtime.device_policy.force_cpu is False


def test_allow_mps_defaults_off() -> None:
    args = parse_worker_args(["--task", "asr", "--lang", "eng", *_REQUIRED_LAUNCH_ARGS])
    runtime = build_worker_bootstrap_runtime(args, environ={})
    assert runtime.device_policy.allow_mps is False
