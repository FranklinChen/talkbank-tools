from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_pattern(path: Path, pattern: str) -> list[tuple[int, str]]:
    regex = re.compile(pattern)
    matches: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if regex.search(line):
            matches.append((lineno, line.strip()))
    return matches


def _scan_paths(paths: list[Path], pattern: str) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in _find_pattern(path, pattern):
            found.append((rel, lineno, line))
    return found


def test_chat_ops_dp_calls_are_allowlisted() -> None:
    chat_ops_src = sorted((ROOT / "crates" / "batchalign-chat-ops" / "src").rglob("*.rs"))
    align_hits = _scan_paths(chat_ops_src, r"\bdp_align::align\s*\(")
    align_chars_hits = _scan_paths(chat_ops_src, r"\bdp_align::align_chars\s*\(")
    # Allowlisted dp_align::align call sites:
    # - benchmark.rs: WER evaluation
    # - compare.rs: transcript comparison (2 calls: window alignment + rotation)
    # - fa/utr.rs: UTR global alignment (correctness-critical, not avoidable)
    # - fa/utr/two_pass.rs: overlap-aware UTR timing recovery
    assert len(align_hits) == 5
    assert {rel for rel, _, _ in align_hits} == {
        "crates/batchalign-chat-ops/src/benchmark.rs",
        "crates/batchalign-chat-ops/src/compare.rs",
        "crates/batchalign-chat-ops/src/fa/utr.rs",
        "crates/batchalign-chat-ops/src/fa/utr/two_pass.rs",
    }
    assert not align_chars_hits
