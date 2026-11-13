"""Managed CLI admission shared by real-model morphotag journeys."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

from batchalign.inference._admission import _FactoryAdmission


class Tokenization(Enum):
    PRESERVE = "preserve"
    RETOKENIZE = "retokenize"


@dataclass(frozen=True, init=False)
class ManagedMorphotag(_FactoryAdmission):
    """An explicitly selected CLI and local managed endpoint.

    The CLI admits live build compatibility on each submission. This fixture
    never starts a server, builds a binary or exposes direct-execution flags.
    The caller owns the service and chooses its interpreter and model cache.
    """

    binary: Path
    server: str

    @classmethod
    def from_environment(cls) -> ManagedMorphotag:
        binary = Path(os.environ["BATCHALIGN_TEST_CLI"]).resolve(strict=True)
        server = os.environ["BATCHALIGN_SERVER"]
        endpoint = urlsplit(server)
        if endpoint.scheme != "http" or endpoint.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "golden morphotag requires an explicit local managed server"
            )
        if not binary.is_file():
            raise ValueError(f"golden CLI is not a file: {binary}")
        admitted = object.__new__(cls)
        object.__setattr__(admitted, "binary", binary)
        object.__setattr__(admitted, "server", server)
        return admitted

    def run(self, source: str, mode: Tokenization = Tokenization.PRESERVE) -> str:
        with tempfile.TemporaryDirectory(prefix="morphotag-golden-") as directory:
            root = Path(directory)
            input_path = root / "input.cha"
            input_path.write_text(source, encoding="utf-8")
            output = root / "output"
            arguments = [
                str(self.binary),
                "--no-open-dashboard",
                "--server",
                self.server,
                "--override-media-cache",
                "morphotag",
                str(input_path),
                "-o",
                str(output),
            ]
            if mode is Tokenization.RETOKENIZE:
                arguments.append("--retokenize")
            completed = subprocess.run(
                arguments, capture_output=True, text=True, timeout=300
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"managed morphotag exited {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            return (output / input_path.name).read_text(encoding="utf-8")
