"""Python launch admission and model-free serving over actual stdio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

import pytest

from batchalign.worker._types import (
    InferTask,
    ModelTarget,
    ProfileTarget,
    TaskTarget,
    UntargetedEcho,
    WorkerProfile,
    WorkerServing,
)


class Device(Enum):
    DEFAULT = ()
    CPU = ("--force-cpu",)


@dataclass(frozen=True, slots=True)
class EchoLaunch:
    """The exact profile, device and serving choice the child receives."""

    target: ModelTarget | UntargetedEcho
    serving: WorkerServing
    device: Device

    def arguments(self) -> list[str]:
        if isinstance(self.target, ProfileTarget):
            target = ["--profile", self.target.profile.value]
        elif isinstance(self.target, TaskTarget):
            target = ["--task", self.target.task.value]
        else:
            target = []
        return [
            "--test-echo",
            *target,
            "--lang",
            "eng",
            "--serving",
            self.serving.value,
            "--supervisor-pid",
            str(os.getpid()),
            "--gpu-thread-pool-size",
            "3",
            "--verbose",
            "1",
            *self.device.value,
        ]


# Run the real module with an import fence in a fresh process. Loading the
# fence into pytest would miss imports that its plugins already cached.
_ECHO_PROGRAM = """
import importlib.abc
import runpy
import sys

class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'stanza'}:
            raise RuntimeError('echo startup imported ' + fullname)

sys.meta_path.insert(0, NoModelImports())
runpy.run_module('batchalign.worker', run_name='__main__')
"""


@pytest.mark.parametrize(
    ("arguments", "diagnostic", "supply_supervisor"),
    [
        (["--test-echo"], "the following arguments are required: --serving", True),
        (
            [
                "--test-echo",
                "--profile",
                "gpu",
                "--task",
                "morphosyntax",
                "--serving",
                "sequential",
            ],
            "--task: not allowed with argument --profile",
            True,
        ),
        (
            ["--serving", "sequential"],
            "--task or --profile is required",
            True,
        ),
        (
            ["--test-echo", "--serving", "sequential"],
            "--supervisor-pid is required for --transport stdio",
            False,
        ),
        (
            ["--test-echo", "--serving", "sequential", "--supervisor-pid", "0"],
            "--supervisor-pid must be a positive PID",
            False,
        ),
        (
            [
                "--test-echo",
                "--serving",
                "sequential",
                "--supervisor-pid",
                "2147483648",
            ],
            "--supervisor-pid must be a positive PID",
            False,
        ),
        (
            [
                "--test-echo",
                "--serving",
                "sequential",
                "--transport",
                "tcp",
                "--port",
                "65536",
            ],
            "--port must be between 0 and 65535",
            False,
        ),
    ],
)
def test_invalid_worker_argv_is_refused_before_ready(
    arguments: list[str],
    diagnostic: str,
    supply_supervisor: bool,
) -> None:
    """Guards: Invalid argv receives a launch diagnostic without a ready line.
    Kind: wire
    Not a type: An independent process can deliver malformed argument bytes
        before Python has an admitted target, supervisor or bind request.
    State: A fresh subprocess receives the raw argv over its real launch edge.
        Cases supplying a valid supervisor use this test process.
    Concurrency: Each case owns its child and pipes, with a bounded deadline.
    """
    supervisor = ["--supervisor-pid", str(os.getpid())] if supply_supervisor else []
    result = subprocess.run(
        [sys.executable, "-c", _ECHO_PROGRAM, *supervisor, *arguments],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert not result.stdout
    assert diagnostic in result.stderr


@pytest.mark.parametrize(
    "launch",
    [
        *(
            EchoLaunch(ProfileTarget(profile), WorkerServing.SEQUENTIAL, Device.CPU)
            for profile in WorkerProfile
        ),
        *(
            EchoLaunch(ProfileTarget(WorkerProfile.GPU), serving, Device.DEFAULT)
            for serving in WorkerServing
        ),
        EchoLaunch(UntargetedEcho(), WorkerServing.SEQUENTIAL, Device.DEFAULT),
        EchoLaunch(TaskTarget(InferTask.FA), WorkerServing.CONCURRENT, Device.CPU),
    ],
)
def test_model_free_echo_worker_serves_the_launch_it_received(
    launch: EchoLaunch,
) -> None:
    """Guards: Python enters the requested serving loop, emits ready and health,
        and drains the health reply at EOF without importing model stacks.
    Kind: contract
    Not a type: The independently implemented Python process can import an
        unexpected model or enter the wrong loop despite admitted argv.
    State: A fresh interpreter exercises the actual import and stdio boundaries.
        It receives only echo work and the import fence refuses model libraries.
    Concurrency: Each case owns its subprocess and pipes. The parent's actual
        PID is the supervisor, and the subprocess deadline bounds a refusal.
    """
    result = subprocess.run(
        [sys.executable, "-c", _ECHO_PROGRAM, *launch.arguments()],
        input='{"op":"health"}\n',
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "echo startup imported" not in result.stderr
    assert f"serving={launch.serving.value} threads=3" in result.stderr
    ready, health = (json.loads(line) for line in result.stdout.splitlines())
    assert ready["ready"] is True
    assert health["op"] == "health"
    assert health["response"]["status"] == "ok"
    assert health["response"]["command"] == launch.target.label
