# affects: batchalign/worker/_main.py
# affects: batchalign/worker/_types.py
"""Tests for typed worker bootstrap runtime resolution."""

from __future__ import annotations

import os
from enum import Enum

import pytest

from batchalign.worker._launch import admit_worker_launch
from batchalign.worker._types import InferTask, ModelStartup, TaskTarget

# Flags the Rust pool always passes when it launches a stdio worker: how the
# process serves requests, and the server it watches so it exits with it.
_REQUIRED_LAUNCH_ARGS = [
    "--serving",
    "sequential",
    "--supervisor-pid",
    str(os.getpid()),
]


class DeviceChoice(Enum):
    DEFAULT = ((), False, False)
    CPU = (("--force-cpu",), True, False)
    MPS = (("--allow-mps",), False, True)


@pytest.mark.parametrize("device", list(DeviceChoice))
def test_worker_bootstrap_runtime_resolves_boundary_inputs(
    device: DeviceChoice,
) -> None:
    """Guards: Task, language, speaker count, engine override, the injected key
        and each device choice survive Python launch admission.
    Kind: wire
    Not a type: Actual argv and environment spellings are parsed at the process
        boundary. In particular MPS defaults off and the injected key is trimmed.
    State: The parser receives declared bytes and a synthetic environment.
    Concurrency: The test owns its input values and reads no ambient credentials.
    """
    flags, force_cpu, allow_mps = device.value
    startup = admit_worker_launch(
        [
            "--task",
            "asr",
            "--lang",
            "eng",
            "--num-speakers",
            "2",
            "--engine-overrides",
            '{"asr":"whisper"}',
            *flags,
            *_REQUIRED_LAUNCH_ARGS,
        ],
        environ={
            "HOME": "/tmp/bootstrap-home",
            "BATCHALIGN_REV_API_KEY": " from-rust ",
        },
    ).startup
    assert isinstance(startup, ModelStartup)
    assert startup.target == TaskTarget(InferTask.ASR)
    bootstrap = startup.bootstrap
    assert bootstrap.lang == "eng"
    assert bootstrap.num_speakers == 2
    assert bootstrap.engine_overrides == {"asr": "whisper"}
    assert bootstrap.device_policy.force_cpu is force_cpu
    assert bootstrap.device_policy.allow_mps is allow_mps
    assert bootstrap.revai_api_key == "from-rust"
