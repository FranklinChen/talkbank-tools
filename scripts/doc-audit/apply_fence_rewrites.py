#!/usr/bin/env python3
"""Apply fence rewrites to book markdown files based on a triage TSV.

Reads rows from a TSV produced by `triage_mdbook_test_failures.py`,
filters by `--class` (the triage classification), and rewrites the
opening fence on each matching row to a new language tag specified by
`--target`.

Idempotency rule: the rewrite is applied ONLY when the existing fence
matches one of the expected shapes (`` ``` `` bare, `` ```rust ``,
or `` ```rust,ignore `` etc.). Any other shape (e.g. an already-
correct tag, or a different tag entirely) is skipped with a warning,
so re-running the script after a partial application is safe.

Usage:
    python3 scripts/doc-audit/apply_fence_rewrites.py \\
      --triage /tmp/mdbook-triage.tsv \\
      --class text \\
      --target text

    python3 scripts/doc-audit/apply_fence_rewrites.py \\
      --triage /tmp/mdbook-triage.tsv \\
      --class pseudocode \\
      --target 'rust,ignore'

The `--target` string is inserted verbatim after `` ``` `` on the
fence line. Common targets: `text`, `yaml`, `json`, `rust,ignore`.

Indentation is preserved: if the original fence is indented (inside a
list item), the rewritten fence keeps the same leading whitespace.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

BOOK_SRC = Path("book/src")

# Recognized starting shapes that we accept for rewriting.
ACCEPTABLE_OLD = re.compile(r"^(\s*)```(?:\s*rust(?:,\s*\w+)?)?\s*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True, type=Path)
    parser.add_argument(
        "--class",
        dest="cls",
        required=True,
        help="Classification to filter (e.g. text, chat, yaml, json, pseudocode, rust)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="New language tag after ``` (e.g. 'text', 'rust,ignore')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files",
    )
    parser.add_argument(
        "--restrict-prefix",
        action="append",
        default=[],
        help="Only rewrite rows whose relpath starts with this prefix. "
             "May be repeated. Empty list = no restriction.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Skip rows whose relpath starts with this prefix. May be repeated.",
    )
    args = parser.parse_args()

    rows_by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with args.triage.open(newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if row["classification"] != args.cls:
                continue
            relpath = row["relpath"]
            if args.restrict_prefix and not any(
                relpath.startswith(p) for p in args.restrict_prefix
            ):
                continue
            if any(relpath.startswith(p) for p in args.exclude_prefix):
                continue
            rows_by_file[relpath].append((int(row["fence_line"]), row["classification"]))

    total_rewrites = 0
    total_skipped = 0
    for relpath, hits in sorted(rows_by_file.items()):
        abs_path = BOOK_SRC / relpath
        if not abs_path.exists():
            print(f"# missing file: {abs_path}")
            continue
        original = abs_path.read_text()
        lines = original.splitlines(keepends=True)

        # Multiple hits may share a line if there's a duplicate row.
        # De-dup the fence lines we'll touch.
        fence_lines = sorted({line for line, _ in hits})

        modified = list(lines)
        rewrites_here = 0
        for one_based in fence_lines:
            if one_based < 1 or one_based > len(modified):
                print(f"# {relpath}:{one_based} out of range")
                total_skipped += 1
                continue
            old = modified[one_based - 1]
            match = ACCEPTABLE_OLD.match(old)
            if not match:
                print(f"# {relpath}:{one_based} fence shape not recognized: {old.rstrip()!r}")
                total_skipped += 1
                continue
            indent = match.group(1)
            new_line = f"{indent}```{args.target}\n"
            if old == new_line:
                # already at target — idempotent skip
                continue
            modified[one_based - 1] = new_line
            rewrites_here += 1

        if rewrites_here == 0:
            continue
        new_text = "".join(modified)
        if args.dry_run:
            print(f"# DRY {relpath}: {rewrites_here} rewrite(s)")
        else:
            abs_path.write_text(new_text)
            print(f"{relpath}: {rewrites_here} rewrite(s)")
        total_rewrites += rewrites_here

    print()
    print(f"rewrites applied: {total_rewrites}")
    print(f"rows skipped:     {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
