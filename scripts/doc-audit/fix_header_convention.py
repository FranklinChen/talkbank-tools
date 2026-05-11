#!/usr/bin/env python3
"""Bulk-fix two header-convention drifts across book/src/ markdown.

1. Rename `**Last modified:**` to `**Last updated:**` on the same line.
2. Insert `**Status:** Current` after the H1 title on pages that lack
   a `**Status:**` line.

Both fixes target the documentation convention used by the canonical
Bucket A pages and described in the workspace CLAUDE.md.

Idempotent: re-running the script on a fixed tree is a no-op.

Usage:
    python3 scripts/doc-audit/fix_header_convention.py [--dry-run]
        [--root book/src]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

H1_LINE = re.compile(r"^# .+$")
STATUS_LINE = re.compile(r"^\*\*Status:\*\*")
LAST_MODIFIED_LINE = re.compile(r"^\*\*Last modified:\*\*(.+)$")


def fix_one(path: Path, dry_run: bool) -> tuple[bool, bool]:
    """Apply both fixes to `path`. Return (renamed, status_added)."""
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    renamed = False
    status_added = False

    # 1) Last modified: → Last updated:
    for i, line in enumerate(lines):
        match = LAST_MODIFIED_LINE.match(line)
        if match:
            tail = match.group(1)
            lines[i] = f"**Last updated:**{tail}\n"
            renamed = True
            break  # convention: at most one Last-modified line per doc

    # 2) Insert Status: Current after H1 if missing
    has_status = any(STATUS_LINE.match(line) for line in lines)
    if not has_status:
        # Find the H1 line index.
        h1_idx: int | None = None
        for i, line in enumerate(lines):
            if H1_LINE.match(line.rstrip("\n")):
                h1_idx = i
                break
        if h1_idx is not None:
            # Insert "**Status:** Current\n" + blank line if next line
            # isn't already blank. Looking for the conventional pattern
            # "# Title\n\n**Status:** Current\n**Last updated:** ...\n".
            insert_pos = h1_idx + 1
            # Skip a single blank line immediately after the H1 so the
            # Status header lands directly between H1 and existing
            # subsequent content. If the next line is blank, our insert
            # goes after it; if it's not blank, we add a blank line
            # before the Status to keep the title visually separated.
            if insert_pos < len(lines) and lines[insert_pos].strip() == "":
                # H1\n\n<insert>\n[rest]
                lines.insert(insert_pos + 1, "**Status:** Current\n")
            else:
                # H1\n<insert blank+Status+blank>\n[rest]
                lines.insert(insert_pos, "\n")
                lines.insert(insert_pos + 1, "**Status:** Current\n")
                lines.insert(insert_pos + 2, "\n")
            status_added = True

    if renamed or status_added:
        new_text = "".join(lines)
        if not dry_run:
            path.write_text(new_text)
        return renamed, status_added
    return False, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="book/src", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    renamed_count = 0
    status_added_count = 0
    touched: list[Path] = []

    for md in sorted(args.root.rglob("*.md")):
        if not md.is_file():
            continue
        renamed, status_added = fix_one(md, args.dry_run)
        if renamed or status_added:
            touched.append(md)
            if renamed:
                renamed_count += 1
            if status_added:
                status_added_count += 1
            tag = "DRY" if args.dry_run else "FIX"
            ops = []
            if renamed:
                ops.append("Last modified → Last updated")
            if status_added:
                ops.append("+ Status: Current")
            print(f"{tag} {md}: {', '.join(ops)}")

    print()
    print(f"Last-modified→Last-updated renames: {renamed_count}")
    print(f"Status: Current insertions:         {status_added_count}")
    print(f"Files touched:                      {len(touched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
