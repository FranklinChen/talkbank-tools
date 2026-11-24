"""Admit the Python worker launch before process or model setup."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from batchalign.device import DevicePolicy
from batchalign.worker._model_loading import (
    parse_engine_overrides,
    resolve_injected_revai_api_key,
)
from batchalign.worker._supervisor import SupervisorPid
from batchalign.worker._types import (
    EchoStartup,
    InferTask,
    ModelStartup,
    ModelTarget,
    ProfileTarget,
    TaskTarget,
    UntargetedEcho,
    WorkerBootstrapRuntime,
    WorkerProfile,
    WorkerServing,
    WorkerStartup,
)


@dataclass(frozen=True, slots=True)
class StdioLaunch:
    supervisor: SupervisorPid


@dataclass(frozen=True, slots=True)
class FixedPort:
    value: int

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 65535:
            raise ValueError("--port must be between 0 and 65535")


@dataclass(frozen=True, slots=True)
class AutomaticPort:
    """Request an available TCP port at bind time."""


@dataclass(frozen=True, slots=True)
class TcpLaunch:
    host: str
    port: FixedPort | AutomaticPort


@dataclass(frozen=True, slots=True)
class AdmittedLaunch:
    startup: WorkerStartup
    transport: StdioLaunch | TcpLaunch
    serving: WorkerServing
    threads: int
    verbosity: int


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the internal worker CLI parser used by the Rust launcher."""
    parser = argparse.ArgumentParser(description="Batchalign worker process")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--task", default="", help="Infer task bootstrap target")
    parser.add_argument("--lang", default="eng", help="Language code")
    parser.add_argument("--num-speakers", type=int, default=1)
    parser.add_argument(
        "--engine-overrides", default="", help="JSON dict of engine overrides"
    )
    parser.add_argument(
        "--test-echo",
        action="store_true",
        help="Test mode: echo input unchanged (no ML models)",
    )
    parser.add_argument(
        "--test-delay-ms",
        type=int,
        default=0,
        help="Test mode: sleep this many ms before each response (for timeout testing)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (0=warn, 1=info, 2=debug, 3=trace)",
    )
    target.add_argument(
        "--profile",
        default="",
        help="Worker profile (gpu, stanza, io), groups related tasks into one process",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-mps",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu-thread-pool-size",
        type=int,
        default=4,
        help="Maximum concurrent requests served inside one worker process "
        "when --serving is concurrent.",
    )
    parser.add_argument(
        "--supervisor-pid",
        type=int,
        default=None,
        help="PID of the server that launched this stdio worker; the worker "
        "exits when that process exits. Required for --transport stdio.",
    )
    parser.add_argument(
        "--serving",
        choices=[mode.value for mode in WorkerServing],
        required=True,
        help="How this process serves requests, decided by the Rust pool that "
        "launched it and routes to it: 'concurrent' (threads sharing the "
        "loaded models) or 'sequential' (one request at a time).",
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "tcp"],
        default="stdio",
        help="IPC transport: stdio (child process) or tcp (persistent daemon)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port to listen on (0 = auto-assign from 9100-9199). Only used with --transport tcp.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP bind address (default: 127.0.0.1). Only used with --transport tcp.",
    )
    return parser


def admit_worker_launch(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> AdmittedLaunch:
    """Resolve argv and injected settings into the one launch startup can use."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        target: ModelTarget | None = None
        if args.task:
            target = TaskTarget(InferTask(args.task))
        elif args.profile:
            target = ProfileTarget(WorkerProfile(args.profile))

        transport: StdioLaunch | TcpLaunch
        if args.transport == "stdio":
            if args.supervisor_pid is None:
                parser.error("--supervisor-pid is required for --transport stdio")
            transport = StdioLaunch(SupervisorPid(args.supervisor_pid))
        else:
            port = AutomaticPort() if args.port == 0 else FixedPort(args.port)
            transport = TcpLaunch(args.host, port)

        env = environ if environ is not None else os.environ
        bootstrap = WorkerBootstrapRuntime(
            lang=args.lang,
            num_speakers=args.num_speakers,
            engine_overrides=parse_engine_overrides(args.engine_overrides),
            device_policy=DevicePolicy(
                force_cpu=args.force_cpu, allow_mps=args.allow_mps
            ),
            revai_api_key=resolve_injected_revai_api_key(env),
        )
        startup: WorkerStartup
        if args.test_echo:
            startup = EchoStartup(
                target or UntargetedEcho(), bootstrap, max(0, args.test_delay_ms)
            )
        elif target is not None:
            startup = ModelStartup(target, bootstrap)
        else:
            parser.error("--task or --profile is required (or use --test-echo)")

        return AdmittedLaunch(
            startup,
            transport,
            WorkerServing(args.serving),
            max(1, args.gpu_thread_pool_size),
            args.verbose,
        )
    except ValueError as error:
        parser.error(str(error))
