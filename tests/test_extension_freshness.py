"""Freshness follows extension inputs, not unrelated server activity."""

import subprocess
from pathlib import Path

import pytest

from scripts.check_extension_freshness import (
    CurrentExtension,
    ExtensionInputs,
    FreshnessError,
    InstalledWheel,
    StaleExtension,
    assess,
    local_crate_directories,
)
from scripts.extension_build_evidence import CompletedBuild, artifact_digest


def package(
    root: Path, name: str, dependencies: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "name": name,
        "manifest_path": str(root / name / "Cargo.toml"),
        "dependencies": dependencies,
    }


def test_cargo_local_closure_includes_transitive_but_not_dev_or_server_edges(
    tmp_path: Path,
) -> None:
    metadata = {
        "packages": [
            package(
                tmp_path,
                "batchalign-pyo3",
                [
                    {"path": str(tmp_path / "shared"), "kind": None},
                    {"path": str(tmp_path / "dev"), "kind": "dev"},
                ],
            ),
            package(
                tmp_path, "shared", [{"path": str(tmp_path / "leaf"), "kind": "build"}]
            ),
            package(tmp_path, "leaf", []),
            package(tmp_path, "dev", []),
            package(tmp_path, "server", []),
        ]
    }
    assert local_crate_directories(metadata) == tuple(
        sorted(tmp_path / name for name in ("batchalign-pyo3", "shared", "leaf"))
    )


@pytest.mark.parametrize("metadata", [{}, {"packages": []}, {"packages": [None]}])
def test_unestablished_input_graph_is_not_current(metadata: object) -> None:
    with pytest.raises((FreshnessError, KeyError)):
        local_crate_directories(metadata)


def test_exact_boundary_and_missing_artifact_are_distinct(tmp_path: Path) -> None:
    source = tmp_path / "source.rs"
    artifact = tmp_path / "extension.so"
    source.touch()
    inputs = ExtensionInputs(tmp_path, (source,))
    with pytest.raises(FreshnessError):
        assess(artifact, inputs)
    artifact.touch()
    assert isinstance(assess(artifact, inputs), StaleExtension)
    CompletedBuild.run(inputs.fingerprint, lambda: (artifact_digest(artifact),)).record(
        tmp_path
    )
    assert isinstance(assess(artifact, inputs), CurrentExtension)


def test_wheel_cleanup_is_cross_platform_and_keeps_non_extensions(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "batchalign_core"
    directory.mkdir()
    installed = tmp_path / "site-packages" / "batchalign_core.abi3.so"
    installed.parent.mkdir()
    installed.touch()
    source = tmp_path / "source.rs"
    source.touch()
    inputs = ExtensionInputs(tmp_path, (source,))
    CompletedBuild.run(
        inputs.fingerprint, lambda: (artifact_digest(installed),)
    ).record(tmp_path)
    for name in (
        "batchalign_core.abi3.so",
        "batchalign_core.cp313-win_amd64.pyd",
        "__init__.py",
    ):
        (directory / name).touch()
    removed = (
        InstalledWheel(installed).require_current(inputs).clear_shadowing_extensions()
    )
    assert {path.suffix for path in removed} == {".so", ".pyd"}
    assert (directory / "__init__.py").exists()
    assert installed.exists()


def test_stale_wheel_has_no_cleanup_transition(tmp_path: Path) -> None:
    artifact = tmp_path / "wheel.so"
    source = tmp_path / "changed.rs"
    artifact.touch()
    source.touch()
    shadow = tmp_path / "batchalign_core" / "batchalign_core.abi3.so"
    shadow.parent.mkdir()
    shadow.write_bytes(b"development copy")
    with pytest.raises(FreshnessError, match="no in-tree extensions removed"):
        InstalledWheel(artifact).require_current(ExtensionInputs(tmp_path, (source,)))
    assert shadow.read_bytes() == b"development copy"


def test_development_install_cannot_authorize_wheel_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    extension = tmp_path / "batchalign_core" / "batchalign_core.abi3.so"
    extension.parent.mkdir()
    extension.touch()
    metadata = tmp_path / "batchalign3.dist-info"
    metadata.mkdir()
    (metadata / "RECORD").write_text("batchalign_core/batchalign_core.abi3.so,,\n")
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: importlib.metadata.PathDistribution(metadata),
    )
    with pytest.raises(FreshnessError, match="editable"):
        InstalledWheel.inspect(tmp_path)
    assert extension.exists()


def test_inspection_includes_new_rust_inputs_and_excludes_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from scripts import check_extension_freshness as checker

    metadata = {
        "packages": [
            package(tmp_path, "batchalign-pyo3", []),
            package(tmp_path, "server", []),
        ]
    }
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("Cargo.toml", "batchalign-pyo3/new.rs", "server/new.rs"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    original = checker._run

    def run(root: Path, *argv: str) -> str:
        return json.dumps(metadata) if argv[0] == "cargo" else original(root, *argv)

    monkeypatch.setattr(checker, "_run", run)
    assert ExtensionInputs.inspect(tmp_path).files == tuple(
        sorted(
            [
                tmp_path / "Cargo.toml",
                tmp_path / "batchalign-pyo3/new.rs",
            ]
        )
    )
