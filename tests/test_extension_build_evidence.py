"""Build evidence survives mtime churn, not changes to the paired content."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from scripts.check_extension_freshness import (
    CurrentExtension,
    ExtensionInputs,
    StaleExtension,
    assess,
)
from scripts.extension_build_evidence import (
    BuildEvidenceError,
    CompletedBuild,
    artifact_digest,
    evidence_path,
)


def test_cached_backend_artifact_survives_checkout_mtime_churn(tmp_path: Path) -> None:
    source, artifact = tmp_path / "source.rs", tmp_path / "extension.so"
    source.write_text("source")
    artifact.write_bytes(b"cached extension")
    inputs = ExtensionInputs(tmp_path, (source,))
    completed = CompletedBuild.run(
        inputs.fingerprint, lambda: (artifact_digest(artifact),)
    )
    completed.record(tmp_path)
    os.utime(source, ns=(200, 200))
    os.utime(artifact, ns=(100, 100))
    assert isinstance(assess(artifact, inputs), CurrentExtension)
    # A real edit is refused even if it preserves or backdates timestamps.
    source.write_text("changed source")
    os.utime(source, ns=(50, 50))
    assert isinstance(assess(artifact, inputs), StaleExtension)


def test_artifact_substitution_and_unrecorded_bytes_are_refused(tmp_path: Path) -> None:
    source = tmp_path / "source.rs"
    source.write_text("unchanged source")
    artifact = tmp_path / "extension.so"
    artifact.write_bytes(b"original")
    inputs = ExtensionInputs(tmp_path, (source,))
    assert isinstance(assess(artifact, inputs), StaleExtension)
    assert not evidence_path(tmp_path).exists(), "a read cannot create evidence"
    CompletedBuild.run(inputs.fingerprint, lambda: (artifact_digest(artifact),)).record(
        tmp_path
    )
    artifact.write_bytes(b"replacement")
    assert isinstance(assess(artifact, inputs), StaleExtension)


@pytest.mark.parametrize("failure", ["backend", "inputs", "empty"])
def test_failed_or_racing_build_cannot_publish_evidence(
    tmp_path: Path, failure: str
) -> None:
    source = tmp_path / "source.rs"
    source.write_text("before")
    inputs = ExtensionInputs(tmp_path, (source,))

    def backend() -> tuple[str, ...]:
        if failure == "backend":
            raise subprocess.CalledProcessError(73, ["backend"])
        if failure == "inputs":
            source.write_text("during build")
        return () if failure == "empty" else (hashlib.sha256(b"artifact").hexdigest(),)

    with pytest.raises((BuildEvidenceError, subprocess.CalledProcessError)):
        CompletedBuild.run(inputs.fingerprint, backend).record(tmp_path)
    assert not evidence_path(tmp_path).exists()


def test_completed_build_cannot_be_constructed_without_running_a_producer() -> None:
    with pytest.raises(TypeError, match="successful producer"):
        CompletedBuild()


def test_build_configuration_and_new_files_change_the_fingerprint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.rs"
    source.write_text("source")
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.maturin]\nprofile = "dev"\n')
    inputs = ExtensionInputs(tmp_path, (source,))
    before = inputs.fingerprint()
    project.write_text('[tool.maturin]\nprofile = "release"\n')
    assert inputs.fingerprint() != before
    project.write_text('[tool.maturin]\nprofile = "dev"\n')
    assert inputs.fingerprint() == before
    new_source = tmp_path / "new.rs"
    new_source.write_text("new")
    assert ExtensionInputs(tmp_path, (source, new_source)).fingerprint() != before
