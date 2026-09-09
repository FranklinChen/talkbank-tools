# affects: crates/batchalign/src/**, crates/batchalign-transform/src/**
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Allowlisted `dp_align::align` call sites, BY FILE AND COUNT.
#
# One mapping, not a count beside a set of names. The previous form asserted
# `len(...) == 5` next to the set of files, so the two had to be kept in step
# by hand and the failure said only "6 != 5", naming neither the file that
# gained a call nor the one that lost one. A per-file count carries strictly
# more (engine.rs legitimately holds two) and the diff points at the change.
#
# dp_align is O(n*m), so a new call site is a decision worth recording rather
# than an incident worth blocking:
#
# - benchmark.rs: WER evaluation.
# - compare/engine.rs: transcript comparison (window alignment + rotation).
# - compare/cross_run.rs: cross-run agreement metrics for `compare-runs`.
# - chat_ops/fa/utr.rs: UTR global alignment, correctness critical and not
#   avoidable.
# - chat_ops/fa/utr/two_pass.rs: overlap-aware UTR timing recovery.
ALLOWED_DP_ALIGN_CALLS = {
    "crates/batchalign-transform/src/benchmark.rs": 1,
    "crates/batchalign-transform/src/compare/cross_run.rs": 1,
    "crates/batchalign-transform/src/compare/engine.rs": 2,
    "crates/batchalign/src/chat_ops/fa/utr.rs": 1,
    "crates/batchalign/src/chat_ops/fa/utr/two_pass.rs": 1,
}

# Allowlisted `dp_align::align_chars` call sites, the same shape. Character
# level alignment is the costlier form (n and m are characters, not words), so
# a site earns its row only by bounding its own input:
#
# - chat_ops/fa/alignment/residue.rs: the residue remap between transcript
#   tokens and aligner labels (adopted 2026-09-07). Bounded BEFORE the call by
#   `MAX_RESIDUE_ALIGN_CHARS` on both sides; over budget it returns the typed
#   `UntimedReason::ResidueTooLongToAlign` rather than aligning.
ALLOWED_DP_ALIGN_CHARS_CALLS = {
    "crates/batchalign/src/chat_ops/fa/alignment/residue.rs": 1,
}


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
    # Batchalign-specific transforms moved from talkbank-transform (now in
    # chatter) into the local batchalign-transform crate during the
    # 2026-06-18 CHAT-core dedup; scan that crate, not the gone path.
    dp_call_roots = [
        ROOT / "crates" / "batchalign" / "src",
        ROOT / "crates" / "batchalign-transform" / "src",
    ]
    dp_call_src = sorted(path for root in dp_call_roots for path in root.rglob("*.rs"))
    align_hits = _scan_paths(dp_call_src, r"\bdp_align::align\s*\(")
    align_chars_hits = _scan_paths(dp_call_src, r"\bdp_align::align_chars\s*\(")

    actual = Counter(rel for rel, _, _ in align_hits)
    assert dict(sorted(actual.items())) == ALLOWED_DP_ALIGN_CALLS, (
        "dp_align::align call sites changed. This is not automatically a "
        "failure: it is a prompt to decide. If the new call is a comparison "
        "or evaluation path, add it to ALLOWED_DP_ALIGN_CALLS with a one "
        "line reason. If it is on a per-file CHAT-ops path, the O(n*m) cost "
        "is the problem and the call is what needs rethinking.\n"
        f"expected: {ALLOWED_DP_ALIGN_CALLS}\ngot:      {dict(sorted(actual.items()))}"
    )
    actual_chars = Counter(rel for rel, _, _ in align_chars_hits)
    assert dict(sorted(actual_chars.items())) == ALLOWED_DP_ALIGN_CHARS_CALLS, (
        "dp_align::align_chars call sites changed. Same rule as above, with a "
        "stricter bar: a character-level site is allowlisted only with the "
        "bound on its input named in the reason.\n"
        f"expected: {ALLOWED_DP_ALIGN_CHARS_CALLS}\ngot:      {dict(sorted(actual_chars.items()))}"
    )
