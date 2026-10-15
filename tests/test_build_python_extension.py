"""Exercise producer subprocesses and the artifact bytes they actually admit."""

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import build_python_extension as producer
from scripts.extension_build_evidence import BuildEvidenceError


@pytest.mark.parametrize("mode", list(producer.BuildMode))
def test_producer_runs_backend_and_admits_actual_extension_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: producer.BuildMode
) -> None:
    extension = tmp_path / "batchalign_core.abi3.so"
    extension.write_bytes(b"actual extension bytes")
    calls: list[list[str]] = []

    def backend(
        argv: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path and check
        calls.append(argv)
        if mode is not producer.BuildMode.EDITABLE:
            dist = tmp_path / "dist"
            dist.mkdir()
            with zipfile.ZipFile(dist / "batchalign3-test.whl", "w") as wheel:
                wheel.write(extension, "batchalign_core/batchalign_core.abi3.so")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", backend)
    monkeypatch.setattr(producer, "imported_extension", lambda root: extension)
    assert mode.build(tmp_path) == (hashlib.sha256(extension.read_bytes()).hexdigest(),)
    if mode is producer.BuildMode.EDITABLE:
        assert calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--no-build-isolation",
                "--reinstall",
                "--editable",
                str(tmp_path),
            ]
        ]
    else:
        profile = (
            ["--release"]
            if mode is producer.BuildMode.WHEEL_RELEASE
            else ["--profile", "dev"]
        )
        assert calls == [
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                *profile,
                "--out",
                "dist/",
            ]
        ]


@pytest.mark.parametrize("mode", list(producer.BuildMode))
def test_failed_backend_cannot_admit_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: producer.BuildMode
) -> None:
    def failed_backend(
        argv: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(73, argv)

    monkeypatch.setattr(subprocess, "run", failed_backend)
    with pytest.raises(subprocess.CalledProcessError):
        mode.build(tmp_path)


@pytest.mark.parametrize("members", [[], ["first.so", "second.so"]])
def test_wheel_requires_one_unambiguous_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, members: list[str]
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "batchalign3-test.whl", "w") as wheel:
        for member in members:
            wheel.writestr(f"batchalign_core/batchalign_core.{member}", b"extension")

    def backend(
        argv: list[str], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", backend)
    with pytest.raises(BuildEvidenceError, match="exactly one extension"):
        producer.BuildMode.WHEEL_DEV.build(tmp_path)
