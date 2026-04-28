#!/usr/bin/env python
"""Emit the test-file subset affected by a git diff.

Phase C of the test-cost revamp. Reads ``# affects: <glob>`` /
``// affects: <glob>`` declarations from every test file, matches
the gitignore-style globs against the paths touched in a git diff,
and prints the relevant test files — plus every test file with no
declarations (the "runs always" set).

Usage::

    # What would run for the uncommitted + last-commit changes?
    python scripts/test-affected.py

    # What would run for this branch vs main?
    python scripts/test-affected.py --base origin/main

    # Limit scan to one directory:
    python scripts/test-affected.py --tests-root batchalign/tests/

Output: absolute paths, one per line, suitable for ``$(...)``
splicing into a pytest invocation::

    uv run pytest $(python scripts/test-affected.py)

Design notes:

* The "runs always" set ensures backward compatibility — a test file
  that hasn't opted in still runs. Adoption can be gradual.
* When the diff is empty (clean tree, no base divergence), we print
  the runs-always set only. Callers that want the full suite should
  not use this tool in that situation.
* If the diff resolution fails (no git, detached head, etc.), we
  print every test file — failing-open is the safe default for a
  tool the developer will reach for interactively.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# This script lives at ``scripts/test-affected.py``; its imports need
# the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from batchalign.tests._affects import select_tests  # noqa: E402


def _git_diff_names(base: str | None) -> list[str]:
    """Return changed paths relative to the repo root.

    If ``base`` is given, use ``git diff --name-only <base>...HEAD``
    plus uncommitted changes. Otherwise default to
    ``HEAD~1..HEAD`` + uncommitted.
    """
    args = ["git", "-C", str(_REPO_ROOT), "diff", "--name-only"]
    if base is None:
        spec = "HEAD~1"
    else:
        spec = f"{base}...HEAD"
    args.append(spec)

    try:
        committed = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=10
        ).stdout.splitlines()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []  # Fail-open caller will print everything.

    try:
        uncommitted = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "diff", "--name-only"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.splitlines()
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.splitlines()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        uncommitted = []
        untracked = []

    merged = [p for p in dict.fromkeys(committed + uncommitted + untracked) if p]
    return merged


def _find_test_files(tests_root: Path) -> list[Path]:
    """Collect every test file under ``tests_root``.

    Python: ``test_*.py`` files. Rust: ``*.rs`` files under a
    ``tests/`` directory OR files tagged with ``#[cfg(test)]``
    (the latter would require AST parsing; skip for now and rely
    on test-binary collocated files).
    """
    py_tests = sorted(tests_root.rglob("test_*.py"))
    # Restrict to Python for the initial roll-out. Rust test
    # annotation is a follow-up (xtask integration).
    return py_tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help="Diff spec: compare HEAD against this base. "
        "Default: HEAD~1 (one-commit window).",
    )
    parser.add_argument(
        "--tests-root",
        default=str(_REPO_ROOT / "batchalign" / "tests"),
        help="Directory to scan for test files.",
    )
    parser.add_argument(
        "--print-diff",
        action="store_true",
        help="Also print the diff path set to stderr (debugging).",
    )
    args = parser.parse_args()

    tests_root = Path(args.tests_root).resolve()
    if not tests_root.is_dir():
        print(f"error: tests-root does not exist: {tests_root}", file=sys.stderr)
        return 2

    changed = _git_diff_names(args.base)
    if args.print_diff:
        print(f"# {len(changed)} changed paths", file=sys.stderr)
        for p in changed:
            print(f"#   {p}", file=sys.stderr)

    test_files = _find_test_files(tests_root)
    if not test_files:
        return 0

    if not changed:
        # Fail-open: no diff info → print everything so callers get a
        # complete run. Scripts that want the "no diff → nothing"
        # behavior can check the caller-side diff first.
        for tf in test_files:
            print(tf)
        return 0

    selected, run_always = select_tests(test_files, changed)
    for tf in list(selected) + list(run_always):
        print(tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
