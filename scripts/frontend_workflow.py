"""Own locked dependency admission before dashboard generation and execution."""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Never, assert_never, final

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@final
@dataclass(frozen=True, init=False)
class WorkspaceLease:
    """One workflow owns the checkout's dependency and generated-asset writes."""

    root: Path
    lock: BinaryIO

    def __init__(self, proof: Never) -> None:
        raise TypeError("use WorkspaceLease.acquire")

    @classmethod
    @contextmanager
    def acquire(cls, root: Path) -> Iterator[WorkspaceLease]:
        root = root.resolve()
        target = root / "target"
        target.mkdir(parents=True, exist_ok=True)
        print("Acquiring frontend workspace lease...", flush=True)
        with (target / "frontend-workflow.lock").open("a+b") as lock:
            if sys.platform == "win32":
                lock.seek(0)
                while True:
                    try:
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EDEADLK):
                            raise
                        time.sleep(0.1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            lease = object.__new__(cls)
            object.__setattr__(lease, "root", root)
            object.__setattr__(lease, "lock", lock)
            yield lease


class Project(Enum):
    FRONTEND = "frontend"
    E2E = "frontend/e2e"


class Action(Enum):
    GENERATE = "generate"
    VERIFY = "verify"
    VERIFY_BUILD = "verify-build"
    BUILD = "build"
    EXISTING_BUILD = "build-existing"
    CHECK = "check"
    SMOKE = "smoke"


class UnitChecks(Enum):
    RUN = "run"
    SKIP = "skip"


@final
@dataclass(frozen=True, init=False)
class InstalledNpm:
    """A successful locked install bound to its directory and manifest bytes."""

    lease: WorkspaceLease
    directory: Path
    executable: str
    manifests: tuple[str, str]

    def __init__(self, proof: Never) -> None:
        raise TypeError("use InstalledNpm.prepare")

    @classmethod
    def prepare(cls, lease: WorkspaceLease, project: Project) -> InstalledNpm:
        directory = lease.root / project.value
        executable = shutil.which("npm")
        if executable is None:
            raise RuntimeError("npm not found. Install Node.js and npm first.")
        paths = (directory / "package.json", directory / "package-lock.json")
        before = (
            hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        )
        subprocess.run([executable, "ci"], cwd=directory, check=True)
        after = (
            hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        )
        if before != after:
            raise RuntimeError("npm manifests changed during dependency preparation")
        installed = object.__new__(cls)
        object.__setattr__(installed, "lease", lease)
        object.__setattr__(installed, "directory", directory)
        object.__setattr__(installed, "executable", executable)
        object.__setattr__(installed, "manifests", before)
        return installed

    def run(self, *arguments: str, environment: dict[str, str] | None = None) -> None:
        subprocess.run(
            [self.executable, *arguments],
            cwd=self.directory,
            env=environment,
            check=True,
        )


@dataclass(frozen=True)
class ApiArtifacts:
    dependencies: InstalledNpm
    artifacts: tuple[Path, Path]


@dataclass(frozen=True)
class Generated(ApiArtifacts):
    schema: Path


@dataclass(frozen=True)
class Verified(Generated):
    comparison: subprocess.CompletedProcess[bytes]


@dataclass(frozen=True)
class Built:
    generated: ApiArtifacts
    distribution: Path


@final
@dataclass(frozen=True, init=False)
class Frontend[Stage]:
    """Operations belong only to the phase that owns their admitted inputs."""

    root: Path
    state: Stage

    def __init__(self, proof: Never) -> None:
        raise TypeError("use Frontend.prepare")

    @classmethod
    def prepare(cls, lease: WorkspaceLease) -> Frontend[InstalledNpm]:
        return _frontend(lease.root, InstalledNpm.prepare(lease, Project.FRONTEND))

    def generate(self: Frontend[InstalledNpm]) -> Frontend[Generated]:
        schema = self.root / "openapi.json"
        binary = os.environ.get("BATCHALIGN_BIN")
        command = (
            [binary, "openapi", "--output", str(schema)]
            if binary
            else [
                "cargo",
                "run",
                "-q",
                "-p",
                "batchalign",
                "--no-default-features",
                "--features",
                "binary-entry,server",
                "--",
                "openapi",
                "--output",
                str(schema),
            ]
        )
        subprocess.run(command, cwd=self.root, check=True)
        shutil.copyfile(schema, self.state.directory / "openapi.json")
        self.state.run(
            "exec",
            "--no",
            "--",
            "openapi-typesafe",
            "openapi.json",
            "-o",
            "src/generated/api.ts",
        )
        self.state.run("run", "generate:decoders")
        admitted = self.admit_generated().state
        return _frontend(
            self.root, Generated(admitted.dependencies, admitted.artifacts, schema)
        )

    def admit_generated(self: Frontend[InstalledNpm]) -> Frontend[ApiArtifacts]:
        artifacts = (
            self.state.directory / "src/generated/api.ts",
            self.state.directory / "src/generated/decoders.ts",
        )
        for path in artifacts:
            if not path.is_file():
                raise RuntimeError(f"generated dashboard artifact is missing: {path}")
        return _frontend(self.root, ApiArtifacts(self.state, artifacts))

    def verify(self: Frontend[Generated]) -> Frontend[Verified]:
        comparison = subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                "--",
                str(self.state.schema),
                *map(str, self.state.artifacts),
            ],
            cwd=self.root,
            check=True,
        )
        return _frontend(
            self.root,
            Verified(
                self.state.dependencies,
                self.state.artifacts,
                self.state.schema,
                comparison,
            ),
        )

    def build(self: Frontend[ApiArtifacts], checks: UnitChecks) -> Frontend[Built]:
        match checks:
            case UnitChecks.RUN:
                self.state.dependencies.run("test")
            case UnitChecks.SKIP:
                pass
            case unexpected:
                assert_never(unexpected)
        self.state.dependencies.run("run", "build")
        distribution = self.state.dependencies.directory / "dist"
        if not (distribution / "index.html").is_file():
            raise RuntimeError("dashboard build did not produce dist/index.html")
        return _frontend(self.root, Built(self.state, distribution))

    def smoke(self: Frontend[Built]) -> None:
        e2e = InstalledNpm.prepare(self.state.generated.dependencies.lease, Project.E2E)
        if os.environ.get("BATCHALIGN_SKIP_BROWSER_INSTALL") != "1":
            if os.environ.get("BATCHALIGN_PLAYWRIGHT_WITH_DEPS") == "1":
                e2e.run(
                    "exec",
                    "--no",
                    "--",
                    "playwright",
                    "install",
                    "--with-deps",
                    "chromium",
                )
            else:
                e2e.run("run", "install:browsers")
        environment = dict(os.environ)
        if environment.get("BATCHALIGN_REAL_SERVER_E2E") == "1":
            binary = Path(
                environment.get(
                    "BATCHALIGN_BIN", str(self.root / "target/release/batchalign3")
                )
            )
            if not os.access(binary, os.X_OK):
                subprocess.run(
                    [
                        "cargo",
                        "build",
                        "--manifest-path",
                        str(self.root / "Cargo.toml"),
                        "-p",
                        "batchalign",
                        "--release",
                    ],
                    check=True,
                )
            environment["BATCHALIGN_BIN"] = str(binary)
            environment.setdefault(
                "BATCHALIGN_DASHBOARD_DIR", str(self.state.distribution)
            )
        e2e.run("test", environment=environment)


def _frontend[NextStage](root: Path, state: NextStage) -> Frontend[NextStage]:
    frontend: Frontend[NextStage] = object.__new__(Frontend)
    object.__setattr__(frontend, "root", root)
    object.__setattr__(frontend, "state", state)
    return frontend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[action.value for action in Action])
    action = Action(parser.parse_args().action)
    with WorkspaceLease.acquire(Path(__file__).resolve().parent.parent) as lease:
        frontend = Frontend.prepare(lease)
        match action:
            case Action.GENERATE:
                frontend.generate()
            case Action.VERIFY:
                frontend.generate().verify()
            case Action.VERIFY_BUILD:
                frontend.generate().verify().build(UnitChecks.SKIP)
            case Action.BUILD:
                frontend.generate().build(UnitChecks.SKIP)
            case Action.EXISTING_BUILD:
                frontend.admit_generated().build(UnitChecks.SKIP)
            case Action.CHECK:
                frontend.admit_generated().build(UnitChecks.RUN)
            case Action.SMOKE:
                frontend.generate().build(UnitChecks.SKIP).smoke()
            case unexpected:
                assert_never(unexpected)


if __name__ == "__main__":
    main()
