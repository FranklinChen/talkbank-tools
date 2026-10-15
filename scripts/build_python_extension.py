"""Build through the supported backend and record content-bound local evidence."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from enum import Enum
from pathlib import Path

from scripts.check_extension_freshness import ExtensionInputs, imported_extension
from scripts.extension_build_evidence import (
    BuildEvidenceError,
    CompletedBuild,
    artifact_digest,
)


class BuildMode(Enum):
    EDITABLE = "editable"
    WHEEL_DEV = "wheel-dev"
    WHEEL_RELEASE = "wheel-release"

    def build(self, root: Path) -> tuple[str, ...]:
        match self:
            case BuildMode.EDITABLE:
                # Explicit local-directory installation invokes the backend;
                # cached Cargo compilation remains valid and is not discarded.
                subprocess.run(
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
                        str(root),
                    ],
                    cwd=root,
                    check=True,
                )
                return (artifact_digest(imported_extension(root)),)
            case BuildMode.WHEEL_DEV | BuildMode.WHEEL_RELEASE:
                profile = (
                    ["--release"]
                    if self is BuildMode.WHEEL_RELEASE
                    else ["--profile", "dev"]
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "maturin",
                        "build",
                        *profile,
                        "--out",
                        "dist/",
                    ],
                    cwd=root,
                    check=True,
                )
                wheels = tuple((root / "dist").glob("batchalign3-*.whl"))
                if len(wheels) != 1:
                    raise BuildEvidenceError("build must produce exactly one wheel")
                with zipfile.ZipFile(wheels[0]) as wheel:
                    extensions = [
                        name
                        for name in wheel.namelist()
                        if Path(name).name.startswith("batchalign_core.")
                        and Path(name).suffix in (".so", ".pyd")
                    ]
                    if len(extensions) != 1:
                        raise BuildEvidenceError(
                            "wheel must contain exactly one extension"
                        )
                    return (hashlib.sha256(wheel.read(extensions[0])).hexdigest(),)


def build(root: Path, mode: BuildMode) -> CompletedBuild:
    completed = CompletedBuild.run(
        lambda: ExtensionInputs.inspect(root).fingerprint(),
        lambda: mode.build(root),
    )
    completed.record(root)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=[mode.value for mode in BuildMode])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    completed = build(args.root.resolve(), BuildMode(args.mode))
    print(f"Recorded {args.mode} extension build for inputs {completed.inputs.digest}")


if __name__ == "__main__":
    main()
