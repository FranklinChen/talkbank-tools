"""Model-free workers must reach the real IPC loop without ML imports."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("profile", ["stanza", "gpu", "io"])
def test_cpu_echo_worker_does_not_import_model_stacks(profile: str) -> None:
    # A fresh interpreter makes this independent of pytest's already-imported
    # ML modules. Fail at the import boundary instead of asserting host timing.
    program = """
import importlib.abc
import runpy
import sys

class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'stanza'}:
            raise RuntimeError('model-free startup imported ' + fullname)

sys.meta_path.insert(0, NoModelImports())
sys.argv = ['batchalign.worker', '--test-echo', '--force-cpu', '--profile', sys.argv[1]]
runpy.run_module('batchalign.worker', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, profile],
        input='{"op":"health"}\n{"op":"shutdown"}\n',
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert responses[0]["ready"] is True
    assert responses[1]["op"] == "health"
    assert responses[1]["response"]["status"] == "ok"
    assert responses[2] == {"op": "shutdown"}
