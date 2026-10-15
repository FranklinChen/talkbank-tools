"""Local build evidence, not release or installed-runtime attestation."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class BuildEvidenceError(Exception):
    """A build did not establish an unchanged input/artifact pairing."""


def artifact_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class InputFingerprint:
    digest: str

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or set(self.digest) - set("0123456789abcdef"):
            raise BuildEvidenceError("invalid input fingerprint")


def evidence_path(root: Path) -> Path:
    return root / "target" / "python-extension-builds.sqlite3"


@dataclass(frozen=True, init=False)
class CompletedBuild:
    """Only a successful producer over stable inputs can record build evidence."""

    inputs: InputFingerprint
    artifacts: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("completed builds require the successful producer")

    @classmethod
    def run(
        cls,
        inspect: Callable[[], InputFingerprint],
        build: Callable[[], tuple[str, ...]],
    ) -> CompletedBuild:
        before = inspect()
        # Exceptions, including a failed backend, cannot reach publication.
        artifacts = build()
        if not artifacts or any(
            len(digest) != 64 or set(digest) - set("0123456789abcdef")
            for digest in artifacts
        ):
            raise BuildEvidenceError("build did not produce extension digests")
        if inspect() != before:
            raise BuildEvidenceError("extension inputs changed during the build")
        completed = object.__new__(cls)
        object.__setattr__(completed, "inputs", before)
        object.__setattr__(completed, "artifacts", artifacts)
        return completed

    def record(self, root: Path) -> None:
        path = evidence_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS extension_builds ("
                "input_digest TEXT NOT NULL, artifact_digest TEXT NOT NULL, "
                "PRIMARY KEY (input_digest, artifact_digest))"
            )
            database.executemany(
                "INSERT OR IGNORE INTO extension_builds VALUES (?, ?)",
                [(self.inputs.digest, artifact) for artifact in self.artifacts],
            )


def has_build_evidence(root: Path, inputs: InputFingerprint, artifact: Path) -> bool:
    """Read-only: a check never creates a database or certifies arbitrary bytes."""
    path = evidence_path(root)
    if not path.is_file():
        return False
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        return (
            database.execute(
                "SELECT 1 FROM extension_builds "
                "WHERE input_digest = ? AND artifact_digest = ?",
                (inputs.digest, artifact_digest(artifact)),
            ).fetchone()
            is not None
        )
