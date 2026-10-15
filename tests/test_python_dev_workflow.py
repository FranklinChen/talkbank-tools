"""Developer entry points must keep dependency and build ownership explicit."""

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dev_tools_do_not_depend_on_the_unpublished_project_itself() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] not in project["dependency-groups"]["dev"]


@pytest.mark.parametrize("sync_exit", [0, 73])
def test_develop_recipe_provisions_before_building(
    tmp_path: Path, sync_exit: int
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> uv.log\n'
        f'if [ "$1" = sync ]; then exit {sync_exit}; fi\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    make = shutil.which("make")
    assert make is not None
    result = subprocess.run(
        [make, "-f", str(ROOT / "Makefile"), "batchalign-develop"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "MAKEFLAGS": "",
            "MFLAGS": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        (tmp_path / "uv.log").read_text().splitlines()
        if (tmp_path / "uv.log").exists()
        else []
    )
    assert calls and calls[0] == "sync --frozen --group dev --no-install-project", (
        result.stderr
    )
    if sync_exit:
        assert result.returncode != 0
        assert len(calls) == 1, "dependency failure cannot reach a build"
    else:
        assert result.returncode == 0, result.stderr
        assert (
            calls[1]
            == "run --no-sync python -m scripts.build_python_extension editable"
        )
