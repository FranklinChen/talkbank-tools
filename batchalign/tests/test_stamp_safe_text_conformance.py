"""Python and Rust agree on which text a provenance stamp can hold.

An engine name a worker reports is written into provenance stamps and FA cache
namespaces byte for byte, so both languages decide whether it is admissible:
Python's ``reported_engine_name`` where the name is born, and Rust's
``StampSafeText`` (behind ``ReportedEngineName``) where it is read. Both are
held to the same cases in ``tests/fixtures/stamp_safe_text_cases.json``, which
the Rust unit tests read too, and the JSON Schema pattern Rust generates for
``ReportedEngineName`` must decide every case the same way.

The cases include the characters the two languages' own ideas of whitespace
disagree on (``str.isspace`` treats U+001C to U+001F as space; Unicode
``White_Space`` does not), so a copy of the rule that drifted to either
built-in would fail here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from batchalign.worker._types import InvalidReportedEngineName, reported_engine_name

# Find the repository root the same way the IPC conformance tests do.
_here = Path(__file__).resolve().parent
ROOT = _here
while ROOT != ROOT.parent:
    if (ROOT / "Cargo.toml").exists() and (ROOT / "ipc-schema").exists():
        break
    ROOT = ROOT.parent

_CASES: list[dict[str, object]] = json.loads(
    (ROOT / "tests" / "fixtures" / "stamp_safe_text_cases.json").read_text(
        encoding="utf-8"
    )
)


def _admitted_by_python(text: str) -> bool:
    try:
        reported_engine_name(text)
    except InvalidReportedEngineName:
        return False
    return True


def _reported_engine_name_patterns() -> set[str]:
    """Every ``ReportedEngineName`` pattern in the generated IPC schemas."""
    patterns: set[str] = set()
    for path in sorted((ROOT / "ipc-schema").rglob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        definition = schema.get("$defs", {}).get("ReportedEngineName")
        if definition is not None:
            patterns.add(definition["pattern"])
    return patterns


def test_the_shared_cases_cover_both_verdicts() -> None:
    verdicts = {case["admitted"] for case in _CASES}
    assert verdicts == {True, False}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: repr(case["text"]))
def test_python_admits_exactly_the_shared_cases(case: dict[str, object]) -> None:
    text = case["text"]
    assert isinstance(text, str)
    assert _admitted_by_python(text) is case["admitted"]


def test_the_generated_schema_pattern_decides_every_case_the_same_way() -> None:
    patterns = _reported_engine_name_patterns()
    # One pattern, generated once: a second spelling would be drift.
    assert len(patterns) == 1, patterns
    (pattern,) = patterns
    compiled = re.compile(pattern)
    for case in _CASES:
        text = case["text"]
        assert isinstance(text, str)
        # ``fullmatch``: JSON Schema's ``$`` does not match before a trailing
        # newline the way Python's does, so the whole text must match.
        assert (compiled.fullmatch(text) is not None) is case["admitted"], case
