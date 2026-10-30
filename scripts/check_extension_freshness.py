"""Check the imported extension against its local Cargo dependency closure.

Cargo owns the dependency graph. The default check never builds or installs;
the developer and wheel recipes own those transitions. Explicit --prepare-wheel
first admits an installed wheel, then removes only shadowing in-tree extensions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import sysconfig
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts.extension_build_evidence import InputFingerprint, has_build_evidence


class FreshnessError(Exception):
    """The artifact or its input graph could not be established."""


@dataclass(frozen=True)
class InstalledWheel:
    """A wheel-owned extension discovered before removing a shadowing copy."""

    extension: Path

    @classmethod
    def inspect(cls, root: Path) -> InstalledWheel:
        distribution = importlib.metadata.distribution("batchalign3")
        files = distribution.files
        if files is None:
            raise FreshnessError("installed distribution has no file manifest")
        candidates = [
            Path(str(distribution.locate_file(file))).resolve()
            for file in files
            if file.name.startswith("batchalign_core.")
            and file.suffix in (".so", ".pyd")
        ]
        if len(candidates) != 1 or not candidates[0].is_file():
            raise FreshnessError(
                "installed wheel must own exactly one existing extension"
            )
        if candidates[0].parent == (root / "batchalign_core").resolve():
            raise FreshnessError("editable extension is not an installed wheel")
        return cls(candidates[0])

    def require_current(self, inputs: ExtensionInputs) -> CurrentInstalledWheel:
        return CurrentInstalledWheel(self, inputs)


@dataclass(frozen=True, init=False)
class CurrentInstalledWheel:
    """Cleanup is paired with the checkout whose inputs admitted the wheel."""

    extension: Path
    root: Path

    def __init__(self, wheel: InstalledWheel, inputs: ExtensionInputs) -> None:
        verdict = assess(wheel.extension, inputs)
        if isinstance(verdict, StaleExtension):
            raise FreshnessError(
                "installed wheel has no matching build evidence; no in-tree extensions removed"
            )
        object.__setattr__(self, "extension", verdict.artifact)
        object.__setattr__(self, "root", inputs.root)

    def clear_shadowing_extensions(self) -> tuple[Path, ...]:
        directory = self.root / "batchalign_core"
        removed: list[Path] = []
        for path in directory.iterdir():
            if path.name.startswith("batchalign_core.") and path.suffix in (
                ".so",
                ".pyd",
            ):
                # A symlink to the wheel is not an independent old copy.
                if path.resolve() == self.extension:
                    continue
                path.unlink()
                removed.append(path)
        return tuple(removed)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FreshnessError("Cargo metadata must contain objects with string keys")
    return cast(dict[str, object], value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise FreshnessError("Cargo metadata must contain an array")
    return cast(list[object], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise FreshnessError("Cargo metadata must contain a string")
    return value


@dataclass(frozen=True)
class CargoPackage:
    name: str
    directory: Path
    dependencies: tuple[Path, ...]

    @classmethod
    def admit(cls, value: object) -> CargoPackage:
        wire = _object(value)
        dependencies: list[Path] = []
        for raw in _array(wire["dependencies"]):
            dependency = _object(raw)
            # Dev-only edges do not contribute to the imported extension.
            if dependency.get("kind") == "dev":
                continue
            path = dependency.get("path")
            if path is not None:
                dependencies.append(Path(_string(path)).resolve())
        return cls(
            _string(wire["name"]),
            Path(_string(wire["manifest_path"])).resolve().parent,
            tuple(dependencies),
        )


def local_crate_directories(metadata: object) -> tuple[Path, ...]:
    packages = tuple(
        CargoPackage.admit(value) for value in _array(_object(metadata)["packages"])
    )
    roots = [package for package in packages if package.name == "batchalign-pyo3"]
    if len(roots) != 1:
        raise FreshnessError(
            "Cargo metadata must identify exactly one batchalign-pyo3 package"
        )
    by_directory = {package.directory: package for package in packages}
    pending = [roots[0].directory]
    reached: set[Path] = set()
    while pending:
        directory = pending.pop()
        if directory in reached:
            continue
        if directory not in by_directory:
            raise FreshnessError(
                f"local dependency absent from Cargo metadata: {directory}"
            )
        reached.add(directory)
        pending.extend(by_directory[directory].dependencies)
    return tuple(sorted(reached))


def _run(root: Path, *argv: str) -> str:
    try:
        return subprocess.run(
            argv, cwd=root, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreshnessError(f"could not inspect {' '.join(argv)}: {error}") from error


@dataclass(frozen=True)
class ExtensionInputs:
    """A nonempty input selection issued from Cargo's local package graph."""

    root: Path
    files: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise FreshnessError("the extension has no inspectable inputs")

    @classmethod
    def inspect(cls, root: Path) -> ExtensionInputs:
        metadata: object = json.loads(
            _run(
                root,
                "cargo",
                "metadata",
                "--offline",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
            )
        )
        directories = local_crate_directories(metadata)
        relative = [str(directory.relative_to(root)) for directory in directories]
        tracked = _run(
            root,
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *relative,
        )
        files = {root / name for name in tracked.split("\0") if name}
        # Lock/config changes can alter this artifact even without Rust edits.
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "rust-toolchain",
            ".cargo/config.toml",
            ".cargo/config",
        ):
            path = root / name
            if path.exists():
                files.add(path)
        if not files:
            raise FreshnessError("the extension has no inspectable inputs")
        return cls(root, tuple(sorted(files)))

    def fingerprint(self) -> InputFingerprint:
        digest = hashlib.sha256()

        def include(value: bytes) -> None:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        include(b"extension-build-evidence-v1")
        for path in sorted(self.files):
            include(path.relative_to(self.root).as_posix().encode())
            # Retain deleted tracked inputs in the inventory, not a stat error.
            include(b"file" if path.is_file() else b"missing")
            if path.is_file():
                include(path.read_bytes())
        project = self.root / "pyproject.toml"
        metadata = tomllib.loads(project.read_text()) if project.exists() else {}
        build_settings = {
            "backend": metadata.get("build-system"),
            "maturin": metadata.get("tool", {}).get("maturin"),
            "maturin_version": importlib.metadata.version("maturin"),
            "python_abi": sysconfig.get_config_var("SOABI"),
            "machine": platform.machine(),
            "platform": sys.platform,
            # The toolchain's identity (release, commit, host). It, not the
            # RUSTUP_TOOLCHAIN that may have selected it, is the build input:
            # rustup's proxy sets that variable for every child of a cargo
            # process, so hashing it made one toolchain fingerprint two ways
            # and a Rust test harness saw every fresh build as stale.
            "rustc": _run(self.root, "rustc", "-vV"),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "RUSTFLAGS",
                    "CARGO_ENCODED_RUSTFLAGS",
                    "CARGO_BUILD_TARGET",
                    "CARGO_TARGET_DIR",
                    "PYO3_PYTHON",
                    "PYO3_CONFIG_FILE",
                    "MACOSX_DEPLOYMENT_TARGET",
                    "CC",
                    "CXX",
                    "CFLAGS",
                    "CXXFLAGS",
                    "LDFLAGS",
                )
            },
        }
        include(json.dumps(build_settings, sort_keys=True).encode())
        return InputFingerprint(digest.hexdigest())


@dataclass(frozen=True)
class CurrentExtension:
    artifact: Path


@dataclass(frozen=True)
class StaleExtension:
    artifact: Path


def assess(
    artifact: Path, inputs: ExtensionInputs
) -> CurrentExtension | StaleExtension:
    if not artifact.is_file():
        raise FreshnessError(f"the imported extension does not exist: {artifact}")
    if has_build_evidence(inputs.root, inputs.fingerprint(), artifact):
        return CurrentExtension(artifact)
    return StaleExtension(artifact)


def imported_extension(root: Path) -> Path:
    resolved = _run(
        root,
        sys.executable,
        "-c",
        "import batchalign_core.batchalign_core as ext; print(ext.__file__)",
    ).strip()
    if not resolved:
        raise FreshnessError("Python did not identify its imported extension")
    return Path(resolved)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        inputs = ExtensionInputs.inspect(root)
        if sys.argv[1:] == ["--prepare-wheel"]:
            wheel = InstalledWheel.inspect(root).require_current(inputs)
            removed = wheel.clear_shadowing_extensions()
            print(
                f"Removed {len(removed)} shadowing in-tree extension(s); wheel owns {wheel.extension}"
            )
        elif sys.argv[1:]:
            raise FreshnessError("only --prepare-wheel or no arguments are supported")
        result = assess(imported_extension(root), inputs)
        if isinstance(result, CurrentExtension):
            print(f"Current imported extension: {result.artifact}")
            return 0
        print(
            "ERROR: imported extension has no successful build matching its "
            f"content and current source/build inputs:\n  artifact: {result.artifact}",
            file=sys.stderr,
        )
    except (
        FreshnessError,
        KeyError,
        ValueError,
        OSError,
        importlib.metadata.PackageNotFoundError,
        sqlite3.Error,
    ) as error:
        print(f"ERROR: cannot establish extension freshness: {error}", file=sys.stderr)
    print(
        "Rebuild with make batchalign-develop (development) or "
        "make batchalign-python-prepare (wheel).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
