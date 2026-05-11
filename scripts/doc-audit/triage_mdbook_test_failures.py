#!/usr/bin/env python3
"""Classify every ```rust block flagged by mdbook test as failing.

Reads /tmp/mdbook-failures.txt (one `relpath:line` per failing block,
where `line` is the line number of the fenced-block heading reported
by mdbook — typically the line of the first content line, not the
fence itself). For each location, opens the file, walks BACK to the
nearest preceding `` ```rust `` (the fence) and FORWARD to the
closing `` ``` ``. The block content is the lines between those.

For each block, produces a classification in one of these buckets:

  yaml       — content looks like YAML (top-level `key: value`, no Rust kw).
  json       — content starts with `{` or `[`, structured JSON-ish.
  toml       — content has `[section]` headers + `key = value`, no Rust kw.
  chat       — content has CHAT markers (`*SPEAKER:`, `@Header:`, `%mor:`).
  text       — content has none of the above and no Rust keyword either
               (likely shell, prose, or output-snippet).
  pseudocode — Rust-flavored but has `...` placeholder, undefined
               types-as-trait-bounds, or refers to identifiers that
               aren't present anywhere in `crates/` (likely a sketch).
  rust       — looks like genuine Rust that should compile. The mdbook
               test failure here represents either genuine API drift
               OR missing doctest scaffolding (no `fn main()`, no
               imports). These are the deep-vet backlog.

Output: TSV to stdout with columns
    relpath  line  classification  first_content_line

The script does NOT modify any files. The operator reviews the TSV,
groups by classification, and applies fence rewrites in a deliberate
second pass.

Usage:
    cd talkbank-tools/book && mdbook test 2>&1 \\
      | rg '^\\s+([\\w/.-]+\\.md) - .* \\(line (\\d+)\\)' -r '$1:$2' \\
      | sort -u > /tmp/mdbook-failures.txt
    cd talkbank-tools && \\
      python3 scripts/doc-audit/triage_mdbook_test_failures.py \\
        /tmp/mdbook-failures.txt > /tmp/mdbook-triage.tsv
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

BOOK_SRC = Path("book/src")

RUST_KEYWORDS = {
    "fn", "let", "use", "impl", "trait", "struct", "enum", "pub", "mod",
    "match", "if", "else", "for", "while", "loop", "return", "async",
    "await", "move", "const", "static", "where", "type", "Self",
}

# Heuristic: a code block contains "real Rust" if it has at least one
# of these signals.
RUST_SIGNAL = re.compile(
    r"\b(?:fn\s+\w|let\s+\w|use\s+\w|impl\b|pub\s+(?:fn|struct|enum|trait|mod|use|const)|"
    r"#\[(?:derive|cfg|test|allow|deny)|\\w+::\\w+|->\s*\w|;\s*$)",
    re.MULTILINE,
)

YAML_LIKE = re.compile(r"^[a-zA-Z_][\w-]*\s*:\s+\S", re.MULTILINE)
# Real JSON starts with { or [ AND contains : or , inside the body.
# Bare `[ba3 align | ... ]` provenance tags caught the old regex.
JSON_LIKE = re.compile(r'^\s*[{\[]\s*"[\w-]+"\s*:')
TOML_SECTION = re.compile(r"^\[[\w\.\-]+\]\s*$", re.MULTILINE)
TOML_KV = re.compile(r"^\w+\s*=\s*\S", re.MULTILINE)
CHAT_MARKER = re.compile(r"^[*%@](?:[A-Z]{2,4}:|\w+:|[Bb]egin|[Ee]nd)", re.MULTILINE)
PLACEHOLDER_DOTS = re.compile(r"\.\.\.(?:[\s,)\]]|$)")


@dataclass
class Block:
    """One fenced code block in a markdown file."""

    relpath: str
    fence_line: int  # 1-based line of the `` ```rust `` opening fence
    content_first_line: int  # 1-based line of first content line (fence + 1)
    content_last_line: int  # 1-based line of last content line (closing - 1)
    content: str  # the block's body, newline-joined


def find_enclosing_block(file_lines: list[str], reported_line: int) -> Block | None:
    """Walk backward from `reported_line` to find the nearest ```rust fence.

    mdbook test reports lines as the FIRST content line (not the fence).
    But some reports may point inside the block. We walk back until we
    see ``` (possibly with a language tag) and treat that as the fence.
    """
    # 1-based -> 0-based for list access; reported_line refers to a
    # content line, so the fence is at some prior index.
    if reported_line < 1 or reported_line > len(file_lines):
        return None

    # mdbook treats `` ``` `` (no tag) AND `` ```rust `` as runnable
    # Rust by default. Walk back until we find any opening fence that
    # is not the close of a previous block.
    fence_idx: int | None = None
    open_re = re.compile(r"^```(?:\s*(rust|edition\w*|ignore|no_run|should_panic|compile_fail)\b)?")
    for idx in range(reported_line - 1, -1, -1):
        line = file_lines[idx].rstrip("\n")
        stripped = line.strip()
        if open_re.match(stripped) and stripped != "```" or stripped == "```":
            # Could be an opener or a closer. Distinguish by walking
            # back further: if the count of triple-backticks before
            # this point is even, this is an opener.
            preceding = sum(
                1 for prior in file_lines[:idx]
                if prior.strip().startswith("```")
            )
            if preceding % 2 == 0:
                fence_idx = idx
                break
            else:
                # this is a closing fence; the reported line isn't in a block
                return None
    if fence_idx is None:
        return None

    # Walk forward to find the closing fence.
    close_idx: int | None = None
    for idx in range(fence_idx + 1, len(file_lines)):
        if file_lines[idx].rstrip("\n").strip() == "```":
            close_idx = idx
            break
    if close_idx is None:
        return None

    content_lines = file_lines[fence_idx + 1 : close_idx]
    return Block(
        relpath="",  # filled by caller
        fence_line=fence_idx + 1,
        content_first_line=fence_idx + 2,
        content_last_line=close_idx,  # last content line is close_idx - 1 + 1 = close_idx
        content="\n".join(line.rstrip("\n") for line in content_lines),
    )


def classify(content: str) -> str:
    """Bucket the content into yaml / json / toml / chat / text / pseudocode / rust."""
    body = content.strip()
    if not body:
        return "text"

    # CHAT comes first because *CHI: looks YAML-like at a glance but isn't.
    if CHAT_MARKER.search(body):
        return "chat"

    # JSON-ish: leading brace or bracket.
    if JSON_LIKE.match(body):
        return "json"

    # TOML: at least one [section] header AND key = value lines and no
    # Rust keywords other than `use` (which would conflict).
    if TOML_SECTION.search(body) and TOML_KV.search(body):
        if not _has_rust_keyword_outside_strings(body):
            return "toml"

    # YAML-like: at least one `key: value` line at column 0, no Rust kw.
    if YAML_LIKE.search(body):
        if not _has_rust_keyword_outside_strings(body):
            return "yaml"

    # If it carries strong Rust signals, it's either genuine rust or
    # pseudocode-flavored rust. Distinguish on `...` placeholder.
    if RUST_SIGNAL.search(body):
        if PLACEHOLDER_DOTS.search(body):
            return "pseudocode"
        return "rust"

    # Has nothing recognizable. If it has bare identifiers and arrows
    # or other Rust-ish syntax, call it pseudocode; otherwise text.
    if "::" in body or "->" in body:
        return "pseudocode"
    return "text"


def _has_rust_keyword_outside_strings(body: str) -> bool:
    """Crude check: any whole-word Rust keyword anywhere in the body."""
    for kw in RUST_KEYWORDS:
        if re.search(rf"\b{kw}\b", body):
            return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    failures_path = Path(argv[1])
    if not failures_path.exists():
        print(f"input file not found: {failures_path}", file=sys.stderr)
        return 2

    # Cache file contents so we read each markdown file once even when
    # multiple failures point at the same file.
    file_cache: dict[str, list[str]] = {}

    print("relpath\tfence_line\tclassification\tfirst_content_line")
    for raw in failures_path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            relpath, line_str = raw.rsplit(":", 1)
            reported_line = int(line_str)
        except ValueError:
            print(f"# skip malformed line: {raw}", file=sys.stderr)
            continue

        abs_path = BOOK_SRC / relpath
        if abs_path.as_posix() not in file_cache:
            try:
                file_cache[abs_path.as_posix()] = abs_path.read_text().splitlines(
                    keepends=True
                )
            except OSError as exc:
                print(f"# read failed: {abs_path}: {exc}", file=sys.stderr)
                continue
        lines = file_cache[abs_path.as_posix()]

        block = find_enclosing_block(lines, reported_line)
        if block is None:
            print(f"# no enclosing fence: {relpath}:{reported_line}", file=sys.stderr)
            continue

        first_content_line = lines[block.fence_line].rstrip("\n") if block.fence_line < len(lines) else ""
        first_content_line_trimmed = first_content_line[:80]
        kind = classify(block.content)
        print(f"{relpath}\t{block.fence_line}\t{kind}\t{first_content_line_trimmed}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
