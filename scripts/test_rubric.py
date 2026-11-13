"""Check that every test carries the TalkBank test rubric.

A test is the last resort, after designing types, typestate and consuming
transitions that make invalid setup or operations unrepresentable. Every retained test
says why it exists, in five labelled lines:

    Guards:      what observable behaviour it protects, and the defect, report or
                 person that made it necessary
    Kind:        boundary | wire | policy | measurement | contract | journey | transcript
    Not a type:  why no type, typestate or state machine can make the failure
                 unrepresentable
    State:       none | what it builds (disk, git, processes) and why a pure
                 observation will not do
    Concurrency: none | what it shares (process, descriptors, environment, ports,
                 files) and how it is isolated

In Rust they are `///` lines on the test function; in Python, lines of the test
function's docstring. A field may continue on following lines indented under it.

Tests that existed before the policy are listed in a baseline file with a hash of
their body. A listed test may stay as it is; once its body changes it must carry
the rubric, and it leaves the baseline. A test not in the baseline must carry the
rubric. The baseline only shrinks.

This file is the single source: repositories carry an exact copy (the workspace's
`scripts/test_rubric/sync.py` keeps them identical). Standard library only.

Usage:
    python3 test_rubric.py check  <repo> --baseline <file> [--json]
    python3 test_rubric.py init   <repo> --baseline <file>
    python3 test_rubric.py shrink <repo> --baseline <file>
    python3 test_rubric.py recognize <repo> --baseline <file>   (only after a checker upgrade)

Exit codes: 0 every test passes; 1 a test fails the rubric or the baseline is stale;
2 the command line or a file is unusable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

VERSION = "2"

FIELDS = ("Guards", "Kind", "Not a type", "State", "Concurrency")
KINDS = frozenset(
    {"boundary", "wire", "policy", "measurement", "contract", "journey", "transcript"}
)
PLACEHOLDERS = frozenset(
    {"n/a", "na", "todo", "tbd", "trivial", "-", "?", "same as above", "obvious", "see above"}
)
NONE_ALLOWED = frozenset({"State", "Concurrency"})
MIN_WORDS = {"Guards": 5, "Not a type": 8, "State": 4, "Concurrency": 4}

# `#[test]`, `#[tokio::test]`, `#[rstest]` and `#[test_case(...)]` each mark a test function.
RUST_TEST_ATTR = re.compile(r"^\s*#\[(?:tokio::test|test|rstest|test_case)\b")
# unittest base classes: a class deriving from one is a test class whatever its name.
PYTHON_TEST_BASES = frozenset({"TestCase", "IsolatedAsyncioTestCase"})
RUST_FN = re.compile(r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
FIELD_LINE = re.compile(r"^(Guards|Kind|Not a type|State|Concurrency)\s*:\s*(.*)$")


class Verdict(Enum):
    """What the checker concludes about one test."""

    PASSES = "passes the rubric"
    GRANDFATHERED = "listed in the baseline, unchanged"
    NEW_WITHOUT_RUBRIC = "is not in the baseline and has no valid rubric"
    EDITED_WITHOUT_RUBRIC = "changed since the baseline and has no valid rubric"


@dataclass(frozen=True)
class FoundTest:
    """One test as found in the source: where it is, its doc text, its body's hash."""

    path: str
    name: str
    line: int
    doc: str
    body_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


@dataclass(frozen=True)
class Judged:
    test: FoundTest
    verdict: Verdict
    problems: tuple[str, ...]


class UsageError(Exception):
    """The command line or an input file cannot be used (exit 2)."""


def body_hash(text: str) -> str:
    """Hash of a test body with all whitespace removed, so reformatting is not an edit."""
    return hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()[:16]


def after_attribute(line: str) -> str:
    """What follows the attribute that opens `line`, by bracket depth (arguments may nest)."""
    depth = 0
    for index, char in enumerate(line):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return line[index + 1 :]
    return ""


def rust_tests(path: str, text: str) -> list[FoundTest]:
    """Every test function in a Rust file (see `RUST_TEST_ATTR`), with its `///` text.

    The `fn` may follow the attribute on the same line or on a later one. The body runs
    from the `fn` to the first later line that is exactly the `fn` line's indentation
    followed by `}` (rustfmt's layout), or ends on its own line when that closes it.
    """
    lines = text.split("\n")
    found: list[FoundTest] = []
    seen: dict[str, int] = {}
    for at, line in enumerate(lines):
        if not RUST_TEST_ATTR.match(line):
            continue
        doc: list[str] = []
        before = at - 1
        while before >= 0 and lines[before].strip().startswith(("///", "#[")):
            if lines[before].strip().startswith("///"):
                doc.insert(0, lines[before].strip()[3:])
            before -= 1
        after = at + 1
        same_line = RUST_FN.match(after_attribute(line).lstrip())
        if same_line is not None:
            after = at
        while same_line is None and after < len(lines) and not RUST_FN.match(lines[after]):
            if lines[after].strip().startswith("///"):
                doc.append(lines[after].strip()[3:])
            elif not lines[after].strip().startswith("#["):
                break
            after += 1
        match = same_line or (RUST_FN.match(lines[after]) if after < len(lines) else None)
        if match is None:
            continue
        name = match.group(2)
        indent = lines[after][: len(lines[after]) - len(lines[after].lstrip())]
        end = after
        if not lines[after].rstrip().endswith("}"):
            end = next(
                (i for i in range(after + 1, len(lines)) if lines[i].rstrip() == f"{indent}}}"),
                len(lines) - 1,
            )
        seen[name] = seen.get(name, 0) + 1
        unique = name if seen[name] == 1 else f"{name}#{seen[name]}"
        found.append(
            FoundTest(
                path=path,
                name=unique,
                line=after + 1,
                doc="\n".join(d.removeprefix(" ") for d in doc),
                body_hash=body_hash("\n".join(lines[after : end + 1])),
            )
        )
    return found


def python_tests(path: str, text: str) -> list[FoundTest]:
    """Every test function: `test_*` at module level, or in a class named `Test*` or
    deriving from a unittest base (`PYTHON_TEST_BASES`), whatever the class's name."""
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise UsageError(f"{path}: cannot be parsed: {error}") from error
    lines = text.split("\n")
    found: list[FoundTest] = []

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef) and (
                node.name.startswith("Test") or is_unittest_class(node)
            ):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                body = node.body
                has_doc = (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                )
                code = body[1:] if has_doc else body
                start = code[0].lineno if code else node.lineno
                end = node.end_lineno or node.lineno
                found.append(
                    FoundTest(
                        path=path,
                        name=f"{prefix}{node.name}",
                        line=node.lineno,
                        doc=ast.get_docstring(node) or "",
                        body_hash=body_hash(
                            "\n".join(lines[node.lineno - 1 : node.lineno])
                            + "\n".join(lines[start - 1 : end])
                        ),
                    )
                )

    visit(tree.body, "")
    return found


def is_unittest_class(node: ast.ClassDef) -> bool:
    """Whether a class derives from a unittest base, written bare or as `unittest.X`."""
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name in PYTHON_TEST_BASES:
            return True
    return False


def is_python_test_file(path: str) -> bool:
    """A Python test file by its name (`test_*.py`, `*_test.py`), never under `fixtures`:
    a fixture may be deliberately invalid Python, and is data, not tests."""
    parts = Path(path).parts
    name = parts[-1]
    named = name.startswith("test_") or name.endswith("_test.py")
    return named and "fixtures" not in parts[:-1]


def rubric_fields(doc: str) -> dict[str, str]:
    """The rubric's labelled fields in a doc text; continuation lines join the field above."""
    fields: dict[str, str] = {}
    current: str | None = None
    for raw in doc.split("\n"):
        line = raw.strip()
        match = FIELD_LINE.match(line)
        if match:
            label: str = match.group(1)
            fields[label] = match.group(2).strip()
            current = label
        elif current is not None and line and raw[:1].isspace():
            fields[current] = f"{fields[current]} {line}".strip()
        else:
            current = None
    return fields


def rubric_problems(doc: str) -> list[str]:
    """Why a doc text fails the rubric (empty when it passes)."""
    fields = rubric_fields(doc)
    problems: list[str] = []
    for field in FIELDS:
        value = fields.get(field)
        if value is None:
            problems.append(f"no `{field}:` line")
            continue
        lowered = value.lower().rstrip(".")
        if lowered in PLACEHOLDERS or not lowered:
            problems.append(f"`{field}:` is a placeholder")
            continue
        if field == "Kind":
            kind = re.split(r"[\s,;.(]", lowered, maxsplit=1)[0]
            if kind not in KINDS:
                problems.append(f"`Kind:` is `{kind}`, not one of {', '.join(sorted(KINDS))}")
            continue
        if lowered == "none" and field in NONE_ALLOWED:
            continue
        if lowered == "none":
            problems.append(f"`{field}:` cannot be none")
            continue
        if len(value.split()) < MIN_WORDS[field]:
            problems.append(f"`{field}:` needs at least {MIN_WORDS[field]} words")
    return problems


def tracked_files(repo: Path) -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.rs", "*.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise UsageError(f"{repo}: git ls-files failed: {error}") from error
    return sorted(listed.split())


def find_tests(repo: Path) -> list[FoundTest]:
    found: list[FoundTest] = []
    for path in tracked_files(repo):
        try:
            text = (repo / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if path.endswith(".rs"):
            found.extend(rust_tests(path, text))
        elif is_python_test_file(path):
            found.extend(python_tests(path, text))
    return found


BASELINE_VERSION = re.compile(r"^# checker version: (\d+)$")


def baseline_version(file: Path) -> int:
    """The checker version a baseline was made or last recognized with (1 when unstated)."""
    if not file.exists():
        return int(VERSION)
    for line in file.read_text(encoding="utf-8").splitlines():
        match = BASELINE_VERSION.match(line)
        if match:
            return int(match.group(1))
    return 1


def read_baseline(file: Path) -> dict[str, str]:
    """`path::name` to body hash, from a tab-separated baseline file."""
    if not file.exists():
        return {}
    entries: dict[str, str] = {}
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise UsageError(f"{file}:{number}: expected `path::name<TAB>hash`")
        entries[parts[0]] = parts[1]
    return entries


def write_baseline(file: Path, entries: dict[str, str], version: int) -> None:
    header = (
        "# Tests from before the test rubric (2026-11-13). This list only shrinks:\n"
        "# a test leaves it when deleted or when it gains the rubric. See test_rubric.py.\n"
        f"# checker version: {version}\n"
    )
    body = "".join(f"{key}\t{entries[key]}\n" for key in sorted(entries))
    file.write_text(header + body, encoding="utf-8")


def judge(tests: list[FoundTest], baseline: dict[str, str]) -> list[Judged]:
    """Each test's verdict; copied rubric text is a problem for every test sharing it."""
    judged: list[Judged] = []
    seen_text: dict[str, list[str]] = {}
    for test in tests:
        fields = rubric_fields(test.doc)
        for field in ("Guards", "Not a type"):
            text = " ".join(fields.get(field, "").lower().split())
            if text:
                seen_text.setdefault(f"{field}\0{text}", []).append(test.key)
    copied = {
        key: field.split("\0")[0]
        for field, keys in seen_text.items()
        if len(keys) > 1
        for key in keys
    }
    for test in tests:
        problems = rubric_problems(test.doc)
        if test.key in copied:
            problems.append(f"`{copied[test.key]}:` is copied from another test")
        if not problems:
            judged.append(Judged(test, Verdict.PASSES, ()))
        elif baseline.get(test.key) == test.body_hash:
            judged.append(Judged(test, Verdict.GRANDFATHERED, ()))
        elif test.key in baseline:
            judged.append(Judged(test, Verdict.EDITED_WITHOUT_RUBRIC, tuple(problems)))
        else:
            judged.append(Judged(test, Verdict.NEW_WITHOUT_RUBRIC, tuple(problems)))
    return judged


def stale_entries(judged: list[Judged], baseline: dict[str, str]) -> list[str]:
    """Baseline entries for tests that are gone or now pass: the baseline must shrink."""
    live = {j.test.key for j in judged if j.verdict is not Verdict.PASSES}
    return sorted(key for key in baseline if key not in live)


def command_check(repo: Path, baseline_file: Path, as_json: bool) -> int:
    baseline = read_baseline(baseline_file)
    judged = judge(find_tests(repo), baseline)
    failures = [
        j
        for j in judged
        if j.verdict in (Verdict.NEW_WITHOUT_RUBRIC, Verdict.EDITED_WITHOUT_RUBRIC)
    ]
    stale = stale_entries(judged, baseline)
    counts = {v.name.lower(): sum(1 for j in judged if j.verdict is v) for v in Verdict}
    if as_json:
        print(json.dumps({"version": VERSION, "tests": len(judged), **counts, "stale": len(stale)}))
    else:
        for j in failures:
            print(f"{j.test.path}:{j.test.line}: {j.test.name} {j.verdict.value}:")
            for problem in j.problems:
                print(f"    {problem}")
        for key in stale:
            print(f"{baseline_file}: {key} is listed but is gone or now passes; run shrink")
        print(
            f"test rubric: {len(judged)} tests; {counts['passes']} pass, "
            f"{counts['grandfathered']} grandfathered, {len(failures)} fail; "
            f"{len(stale)} stale baseline entries"
        )
    return 1 if failures or stale else 0


def command_init(repo: Path, baseline_file: Path) -> int:
    if baseline_file.exists():
        raise UsageError(f"{baseline_file} exists; the baseline is made once, then only shrinks")
    judged = judge(find_tests(repo), {})
    entries = {j.test.key: j.test.body_hash for j in judged if j.verdict is not Verdict.PASSES}
    write_baseline(baseline_file, entries, int(VERSION))
    print(f"test rubric: baseline of {len(entries)} tests written to {baseline_file}")
    return 0


def command_shrink(repo: Path, baseline_file: Path) -> int:
    baseline = read_baseline(baseline_file)
    stale = set(stale_entries(judge(find_tests(repo), baseline), baseline))
    write_baseline(
        baseline_file,
        {k: v for k, v in baseline.items() if k not in stale},
        baseline_version(baseline_file),
    )
    print(f"test rubric: {len(stale)} entries removed; {len(baseline) - len(stale)} remain")
    return 0


def command_recognize(repo: Path, baseline_file: Path) -> int:
    """Admit tests a newer checker recognizes that an older one did not see.

    The one way a baseline may grow: only when this checker's version is newer than the
    baseline's, and only in a commit that changes nothing but the checker copy and the
    baseline, so every added test is one that existed before and was merely unseen.
    """
    made_with = baseline_version(baseline_file)
    if made_with >= int(VERSION):
        raise UsageError(
            f"{baseline_file} is already at checker version {made_with}; "
            "a baseline grows only when the checker recognizes more tests"
        )
    baseline = read_baseline(baseline_file)
    judged = judge(find_tests(repo), baseline)
    unseen = {
        j.test.key: j.test.body_hash
        for j in judged
        if j.verdict is Verdict.NEW_WITHOUT_RUBRIC and j.test.key not in baseline
    }
    write_baseline(baseline_file, {**baseline, **unseen}, int(VERSION))
    print(
        f"test rubric: checker version {made_with} to {VERSION}: "
        f"{len(unseen)} newly recognized tests admitted to {baseline_file}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the TalkBank test rubric.")
    parser.add_argument("command", choices=["check", "init", "shrink", "recognize"])
    parser.add_argument("repo", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return command_check(args.repo, args.baseline, args.json)
        if args.command == "init":
            return command_init(args.repo, args.baseline)
        if args.command == "recognize":
            return command_recognize(args.repo, args.baseline)
        return command_shrink(args.repo, args.baseline)
    except UsageError as error:
        print(f"test rubric: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
