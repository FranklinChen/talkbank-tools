"""Real wheel recipes over isolated, stateless build-tool doubles."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def wheel_recipe(
    tmp_path: Path, profile: str, *, cargo_code: int = 0
) -> subprocess.CompletedProcess[str]:
    tools = tmp_path / "tools"
    tools.mkdir()
    packaged = tmp_path / "batchalign" / "_bin" / "batchalign3"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("previous-artifact", encoding="utf-8")
    packaged.chmod(0o755)
    scripts = {
        "cargo": f"""#!/bin/sh
printf '%s\\n' "$*" >> cargo.log
if [ {cargo_code} -ne 0 ]; then exit {cargo_code}; fi
mkdir -p target/debug target/release
printf native-dev > target/debug/batchalign3
printf native-release > target/release/batchalign3
""",
        "uv": """#!/bin/sh
printf '%s\\n' "$*" >> uv.log
""",
    }
    for name, content in scripts.items():
        executable = tools / name
        executable.write_text(content, encoding="utf-8")
        executable.chmod(0o755)
    make = shutil.which("make")
    assert make is not None, "wheel recipe verification requires make"
    environment = {
        **os.environ,
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "BATCHALIGN_PRESTAGED_BIN": "1" if profile == "dev" else "0",
        "MAKEFLAGS": "",
        "MFLAGS": "",
    }
    return subprocess.run(
        [
            make,
            "-f",
            str(Path(__file__).resolve().parents[1] / "Makefile"),
            "batchalign-build-wheel",
            f"BATCHALIGN_BUILD_PROFILE={profile}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("profile", ["release", "dev"])
def test_native_binary_and_extension_use_the_same_profile(
    tmp_path: Path, profile: str
) -> None:
    result = wheel_recipe(tmp_path, profile)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "batchalign/_bin/batchalign3").read_text() == f"native-{profile}"
    assert (
        f"build --profile {profile} -p batchalign"
        in (tmp_path / "cargo.log").read_text()
    )
    wheel_args = (tmp_path / "uv.log").read_text()
    assert (
        "maturin build --profile dev" in wheel_args
        if profile == "dev"
        else "maturin build --release" in wheel_args
    )


def test_failed_native_build_cannot_package_a_previous_binary(tmp_path: Path) -> None:
    result = wheel_recipe(tmp_path, "release", cargo_code=73)
    assert result.returncode != 0
    assert (tmp_path / "batchalign/_bin/batchalign3").read_text() == "previous-artifact"
    assert not (tmp_path / "uv.log").exists(), (
        "failed compilation cannot reach wheel packaging"
    )


def test_unknown_profile_is_refused_before_build_mutations(tmp_path: Path) -> None:
    result = wheel_recipe(tmp_path, "unknown")
    assert result.returncode != 0
    assert "must be release or dev" in result.stderr
    assert not (tmp_path / "cargo.log").exists()
    assert not (tmp_path / "uv.log").exists()
