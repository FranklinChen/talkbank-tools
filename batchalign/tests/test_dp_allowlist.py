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
# more information and the diff points at the change.
#
# dp_align is O(n*m), so a new call site is a decision worth recording rather
# than an incident worth blocking:
#
# - compare/engine.rs: one whole-file transcript comparison, shared by WER
#   evaluation; window alignment and rotation no longer add separate calls.
# - compare/cross_run.rs: cross-run agreement metrics for `compare-runs`.
#
# UTR no longer calls `dp_align::align` directly: it reaches the same DP
# through `CorrespondenceAnalysis::observe`, tracked below.
ALLOWED_DP_ALIGN_CALLS = {
    "crates/batchalign-transform/src/compare/cross_run.rs": 1,
    "crates/batchalign-transform/src/compare/engine.rs": 1,
}

# Allowlisted `CorrespondenceAnalysis::observe` call sites. `observe` runs one
# `dp_align::align` (the selected path) and a budgeted common-correspondence
# admission over the same pair, so each call is the same O(n*m) decision:
#
# - chat_ops/fa/utr.rs: UTR alignment of one anchored region's words against
#   its ASR tokens, correctness critical and not avoidable; regions bound n
#   and m.
# - chat_ops/fa/utr/two_pass.rs: overlap-aware UTR timing recovery, one
#   windowed utterance at a time.
ALLOWED_CORRESPONDENCE_OBSERVE_CALLS = {
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
    observe_hits = _scan_paths(dp_call_src, r"\bCorrespondenceAnalysis::observe\s*\(")

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
    # Production call sites only: the type's own unit tests exercise it on
    # fixed toy inputs.
    actual_observe = Counter(
        rel
        for rel, _, _ in observe_hits
        if not rel.endswith("/tests.rs") and "/tests/" not in rel
    )
    assert (
        dict(sorted(actual_observe.items())) == ALLOWED_CORRESPONDENCE_OBSERVE_CALLS
    ), (
        "CorrespondenceAnalysis::observe call sites changed. Each runs the "
        "O(n*m) DP; same rule as dp_align::align above.\n"
        f"expected: {ALLOWED_CORRESPONDENCE_OBSERVE_CALLS}\n"
        f"got:      {dict(sorted(actual_observe.items()))}"
    )
