"""Check the imported extension against its local Cargo dependency closure.

Cargo owns the dependency graph. The default check never builds or installs;
the developer and wheel recipes own those transitions. Explicit --prepare-wheel
first admits an installed wheel, then removes only shadowing in-tree extensions.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast


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
        verdict = assess(self.extension, inputs)
        if isinstance(verdict, StaleExtension):
            raise FreshnessError(
                f"installed wheel is older than {verdict.newer_input}; no in-tree extensions removed"
            )
        return CurrentInstalledWheel(verdict.artifact)


@dataclass(frozen=True)
class CurrentInstalledWheel:
    """Only a current wheel authorizes removal of shadowing development copies."""

    extension: Path

    def clear_shadowing_extensions(self, root: Path) -> tuple[Path, ...]:
        directory = root / "batchalign_core"
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
        return cls(tuple(sorted(files)))


@dataclass(frozen=True)
class CurrentExtension:
    artifact: Path


@dataclass(frozen=True)
class StaleExtension:
    artifact: Path
    newer_input: Path


def assess(
    artifact: Path, inputs: ExtensionInputs
) -> CurrentExtension | StaleExtension:
    if not artifact.is_file():
        raise FreshnessError(f"the imported extension does not exist: {artifact}")
    newest = max(inputs.files, key=lambda path: path.stat().st_mtime_ns)
    if artifact.stat().st_mtime_ns < newest.stat().st_mtime_ns:
        return StaleExtension(artifact, newest)
    return CurrentExtension(artifact)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        inputs = ExtensionInputs.inspect(root)
        if sys.argv[1:] == ["--prepare-wheel"]:
            wheel = InstalledWheel.inspect(root).require_current(inputs)
            removed = wheel.clear_shadowing_extensions(root)
            print(
                f"Removed {len(removed)} shadowing in-tree extension(s); wheel owns {wheel.extension}"
            )
        elif sys.argv[1:]:
            raise FreshnessError("only --prepare-wheel or no arguments are supported")
        resolved = _run(
            root,
            sys.executable,
            "-c",
            "import batchalign_core.batchalign_core as ext; print(ext.__file__)",
        ).strip()
        if not resolved:
            raise FreshnessError("Python did not identify its imported extension")
        result = assess(Path(resolved), inputs)
        if isinstance(result, CurrentExtension):
            print(f"Current imported extension: {result.artifact}")
            return 0
        print(
            f"ERROR: imported extension is older than its inputs:\n"
            f"  artifact: {result.artifact}\n  newer input: {result.newer_input}",
            file=sys.stderr,
        )
    except (
        FreshnessError,
        KeyError,
        ValueError,
        OSError,
        importlib.metadata.PackageNotFoundError,
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
