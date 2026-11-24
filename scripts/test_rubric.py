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
function's docstring. A field continues on the following lines until a blank line
or the next field.

Kind is exactly one of the kind words. Guards, Not a type, State and Concurrency
are each written for their own test: a line identical to another test's, or
identical once its identifiers (CamelCase, snake_case, `backticked`, paths) are
replaced, is refused. `none` is exempt.

Tests that honestly share a State or Concurrency (one fixture, one isolation)
name a shared description instead of repeating it. It is declared once, anywhere
in the repository, in an ordinary `//` comment (`#` in Python) that begins with
the words Test state, or Test concurrency, then a lowercase name and a colon, for
example `Test state git-origin: a private committed repository and a local bare
origin ...`; following comment lines continue it. A test refers to it by name and
may add what is its own: `State: [git-origin] This test also corrupts one blob.` A reference to
an unknown name, a name declared twice, a declaration no test uses, and a
declaration copied from another are refused.

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

VERSION = "3"

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
COPY_CHECKED = ("Guards", "Not a type", "State", "Concurrency")
SHAREABLE = {"State": "state", "Concurrency": "concurrency"}
# `// Test state <name>: text` or `# Test concurrency <name>: text`.
DECLARATION = re.compile(
    r"^\s*(?://|#)\s*Test (state|concurrency) ([a-z0-9][a-z0-9-]*)\s*:\s*(.*)$"
)
COMMENT = re.compile(r"^\s*(?://(?!/)|#)\s?(.*)$")
REFERENCE = re.compile(r"^\[([a-z0-9][a-z0-9-]*)\]\s*(.*)$")
# What names code rather than prose; replaced to find copies with a name swapped.
IDENTIFIER = re.compile(
    r"`[^`]+`"
    r"|\b[A-Za-z_]+::[A-Za-z_:]+\b"
    r"|\b[a-z]+(?:_[a-z0-9]+)+\b"
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"
)


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
class Problem:
    """One reason a test fails, and the checker version whose rules found it. A
    newer checker's baseline may admit only tests whose every problem is new."""

    text: str
    since: int


@dataclass(frozen=True)
class Judged:
    test: FoundTest
    verdict: Verdict
    problems: tuple[Problem, ...]


@dataclass(frozen=True)
class Declaration:
    """A shared State or Concurrency description, declared once by name."""

    field: str
    name: str
    text: str
    path: str
    line: int


@dataclass(frozen=True)
class Declarations:
    """Every declaration in a repository, keyed by (field, name), with the problems
    found while reading them (a name declared twice)."""

    by_name: dict[tuple[str, str], Declaration]
    problems: tuple[str, ...]

    @staticmethod
    def of(found: list[Declaration]) -> Declarations:
        """Admit declarations in order; a name declared again is a problem, never a
        replacement of the first."""
        by_name: dict[tuple[str, str], Declaration] = {}
        problems: list[str] = []
        for declaration in found:
            key = (SHAREABLE[declaration.field], declaration.name)
            earlier = by_name.get(key)
            if earlier is not None:
                problems.append(
                    f"{declaration.path}:{declaration.line}: `Test {key[0]} {key[1]}` is "
                    f"already declared at {earlier.path}:{earlier.line}"
                )
                continue
            by_name[key] = declaration
        return Declarations(by_name, tuple(problems))


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


# Only lexical regions that can conceal a test attribute are masked. Source
# validity and macro expansion remain the Rust compiler's responsibility.
RUST_NON_CODE = re.compile(
    r'//|/\*|r(?P<hashes>\#*)"|(?P<string>")|'
    r"(?P<character>'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|[^\n])|[^'\\\r\n])')"
)
RUST_BLOCK_MARKER = re.compile(r"/\*|\*/")


def rust_code_only(source: str) -> str:
    """Mask literals/comments while preserving every character position and newline."""
    chunks: list[str] = []
    cursor = 0
    while token := RUST_NON_CODE.search(source, cursor):
        start = token.start()
        end = token.end()
        chunks.append(source[cursor:start])
        if token.group() == "//":
            newline = source.find("\n", end)
            end = len(source) if newline < 0 else newline
        elif token.group() == "/*":
            depth = 1
            while depth and (marker := RUST_BLOCK_MARKER.search(source, end)):
                depth += 1 if marker.group() == "/*" else -1
                end = marker.end()
            if depth:
                end = len(source)
        elif token.group("hashes") is not None:
            closing = '"' + token.group("hashes")
            closing_at = source.find(closing, end)
            end = len(source) if closing_at < 0 else closing_at + len(closing)
        elif token.group("string") is not None:
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                elif source[end] == '"':
                    end += 1
                    break
                else:
                    end += 1
            end = min(end, len(source))
        # Character tokens already include their closing quote. Lifetimes
        # have no closing character quote and do not match this token family.
        # Allocate padding per lexical region/line, without one list slot
        # per source character. Newline and character positions stay exact.
        chunks.append("\n".join(" " * len(line) for line in source[start:end].split("\n")))
        cursor = end
    chunks.append(source[cursor:])
    return "".join(chunks)


@dataclass(frozen=True, init=False)
class RustSource:
    """One raw source and its position-preserving lexical code view.

    Construction performs masking once. Consumers cannot supply an unrelated
    code view or mutate either retained line sequence.
    """

    source_lines: tuple[str, ...]
    code_lines: tuple[str, ...]

    def __init__(self, source: str) -> None:
        object.__setattr__(self, "source_lines", tuple(source.split("\n")))
        object.__setattr__(self, "code_lines", tuple(rust_code_only(source).split("\n")))


def rust_tests(path: str, text: str) -> list[FoundTest]:
    """Every test function in a Rust file (see `RUST_TEST_ATTR`), with its `///` text.

    The `fn` may follow the attribute on the same line or on a later one. The body runs
    from the `fn` to the first later line that is exactly the `fn` line's indentation
    followed by `}` (rustfmt's layout), or ends on its own line when that closes it.
    """
    source = RustSource(text)
    lines = source.source_lines
    code_lines = source.code_lines
    found: list[FoundTest] = []
    seen: dict[str, int] = {}
    for at, code_line in enumerate(code_lines):
        if not RUST_TEST_ATTR.match(code_line):
            continue
        doc: list[str] = []
        before = at - 1
        while before >= 0 and lines[before].strip().startswith(("///", "#[")):
            if lines[before].strip().startswith("///"):
                doc.insert(0, lines[before].strip()[3:])
            before -= 1
        after = at + 1
        same_line = RUST_FN.match(after_attribute(code_line).lstrip())
        if same_line is not None:
            after = at
        while same_line is None and after < len(lines) and not RUST_FN.match(code_lines[after]):
            if lines[after].strip().startswith("///"):
                doc.append(lines[after].strip()[3:])
            elif not lines[after].strip().startswith("#["):
                break
            after += 1
        match = same_line or (RUST_FN.match(code_lines[after]) if after < len(lines) else None)
        if match is None:
            continue
        name = match.group(2)
        indent = lines[after][: len(lines[after]) - len(lines[after].lstrip())]
        end = after
        if not code_lines[after].rstrip().endswith("}"):
            end = next(
                (
                    i
                    for i in range(after + 1, len(lines))
                    if code_lines[i].rstrip() == f"{indent}}}"
                ),
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
        elif current is not None and line:
            fields[current] = f"{fields[current]} {line}".strip()
        else:
            current = None
    return fields


def own_text(field: str, value: str) -> str:
    """The part of a field the test wrote itself: a shared reference is not its own."""
    match = REFERENCE.match(value) if field in SHAREABLE else None
    return match.group(2) if match else value


def skeleton(text: str) -> str:
    """The text with every identifier replaced, so a name swap does not make it new."""
    return " ".join(IDENTIFIER.sub("X", text).lower().split())


def rubric_problems(doc: str, declarations: Declarations) -> list[Problem]:
    """Why a doc text fails the rubric on its own (empty when it passes). Copies
    across tests are judged separately, in `judge`."""
    fields = rubric_fields(doc)
    problems: list[Problem] = []

    def refuse(text: str, since: int = 2) -> None:
        problems.append(Problem(text, since))

    for field in FIELDS:
        value = fields.get(field)
        if value is None:
            refuse(f"no `{field}:` line")
            continue
        reference = REFERENCE.match(value) if field in SHAREABLE else None
        if reference is not None:
            if (SHAREABLE[field], reference.group(1)) not in declarations.by_name:
                refuse(
                    f"`{field}:` names [{reference.group(1)}], which no "
                    f"`Test {SHAREABLE[field]} {reference.group(1)}:` comment declares",
                    3,
                )
            continue
        lowered = value.lower().rstrip(".")
        if lowered in PLACEHOLDERS or not lowered:
            refuse(f"`{field}:` is a placeholder")
            continue
        if field == "Kind":
            if value not in KINDS:
                # Version 2 accepted a kind word followed by punctuation or more text.
                word = re.split(r"[\s,;.(]", lowered, maxsplit=1)[0]
                refuse(
                    f"`Kind:` is `{value}`; it must be exactly one of {', '.join(sorted(KINDS))}",
                    3 if word in KINDS else 2,
                )
            continue
        if lowered == "none" and field in NONE_ALLOWED:
            continue
        if lowered == "none":
            refuse(f"`{field}:` cannot be none")
            continue
        if len(value.split()) < MIN_WORDS[field]:
            refuse(f"`{field}:` needs at least {MIN_WORDS[field]} words")
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


def declarations_in(path: str, text: str) -> list[Declaration]:
    """Each `Test state|concurrency <name>:` comment in a file, with its continuation
    comment lines (up to a blank comment, a non-comment line or the next declaration)."""
    found: list[Declaration] = []
    lines = text.split("\n")
    at = 0
    while at < len(lines):
        match = DECLARATION.match(lines[at])
        if match is None:
            at += 1
            continue
        parts = [match.group(3).strip()]
        start = at
        at += 1
        while at < len(lines) and not DECLARATION.match(lines[at]):
            comment = COMMENT.match(lines[at])
            if comment is None or not comment.group(1).strip():
                break
            parts.append(comment.group(1).strip())
            at += 1
        field = "State" if match.group(1) == "state" else "Concurrency"
        found.append(Declaration(field, match.group(2), " ".join(parts), path, start + 1))
    return found


def find_declarations(repo: Path) -> Declarations:
    """Every shared description in the repository's tracked Rust and Python files."""
    found: list[Declaration] = []
    for path in tracked_files(repo):
        try:
            text = (repo / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found.extend(declarations_in(path, text))
    return Declarations.of(found)


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


def judge(
    tests: list[FoundTest], baseline: dict[str, str], declarations: Declarations
) -> list[Judged]:
    """Each test's verdict. Copied rubric text is a problem for every test sharing it:
    identical text, or identical once identifiers are replaced. A declaration counts
    as one more author, so pasting its text instead of naming it is a copy too."""
    judged: list[Judged] = []
    authors: dict[str, set[str]] = {}

    def wrote(field: str, text: str, author: str) -> None:
        if not text or text.lower().rstrip(".") == "none":
            return
        # Version 2 compared exact Guards and Not a type text; version 3 added the
        # identifier skeleton and the State and Concurrency fields.
        exact = 2 if field in ("Guards", "Not a type") else 3
        for form, since in ((" ".join(text.lower().split()), exact), (skeleton(text), 3)):
            authors.setdefault(f"{field}\0{since}\0{form}", set()).add(author)

    for declaration in declarations.by_name.values():
        wrote(declaration.field, declaration.text, f"declaration {declaration.name}")
    for test in tests:
        fields = rubric_fields(test.doc)
        for field in COPY_CHECKED:
            wrote(field, own_text(field, fields.get(field, "")), test.key)
    copied: dict[str, dict[str, int]] = {}
    for form, keys in authors.items():
        if len(keys) > 1:
            copied_field, version, _ = form.split("\0", 2)
            for key in keys:
                found = copied.setdefault(key, {})
                found[copied_field] = min(found.get(copied_field, int(version)), int(version))
    for test in tests:
        problems = rubric_problems(test.doc, declarations)
        for field, since in sorted(copied.get(test.key, {}).items()):
            problems.append(
                Problem(
                    f"`{field}:` is copied from another test or declaration; write this "
                    "test's own, or name a shared `Test state|concurrency` declaration",
                    since,
                )
            )
        if not problems:
            judged.append(Judged(test, Verdict.PASSES, ()))
        elif baseline.get(test.key) == test.body_hash:
            judged.append(Judged(test, Verdict.GRANDFATHERED, ()))
        elif test.key in baseline:
            judged.append(Judged(test, Verdict.EDITED_WITHOUT_RUBRIC, tuple(problems)))
        else:
            judged.append(Judged(test, Verdict.NEW_WITHOUT_RUBRIC, tuple(problems)))
    return judged


def declaration_problems(tests: list[FoundTest], declarations: Declarations) -> list[str]:
    """Problems of the declarations themselves: a name declared twice, a declaration
    no test names, one too short or a placeholder, and two with the same text."""
    used = set()
    for test in tests:
        fields = rubric_fields(test.doc)
        for field, kind in SHAREABLE.items():
            match = REFERENCE.match(fields.get(field, ""))
            if match:
                used.add((kind, match.group(1)))
    problems = list(declarations.problems)
    texts: dict[str, Declaration] = {}
    for key, declaration in sorted(declarations.by_name.items()):
        where = f"{declaration.path}:{declaration.line}: `Test {key[0]} {key[1]}`"
        if key not in used:
            problems.append(f"{where} is declared but no test names it")
        lowered = declaration.text.lower().rstrip(".")
        if lowered in PLACEHOLDERS or lowered == "none":
            problems.append(f"{where} is a placeholder")
        elif len(declaration.text.split()) < MIN_WORDS[declaration.field]:
            problems.append(f"{where} needs at least {MIN_WORDS[declaration.field]} words")
        form = f"{declaration.field}\0{skeleton(declaration.text)}"
        if form in texts:
            other = texts[form]
            problems.append(f"{where} repeats `Test {key[0]} {other.name}` at {other.path}")
        texts[form] = declaration
    return problems


def stale_entries(judged: list[Judged], baseline: dict[str, str]) -> list[str]:
    """Baseline entries for tests that are gone or now pass: the baseline must shrink."""
    live = {j.test.key for j in judged if j.verdict is not Verdict.PASSES}
    return sorted(key for key in baseline if key not in live)


def command_check(repo: Path, baseline_file: Path, as_json: bool) -> int:
    baseline = read_baseline(baseline_file)
    tests = find_tests(repo)
    declarations = find_declarations(repo)
    judged = judge(tests, baseline, declarations)
    shared_problems = declaration_problems(tests, declarations)
    failures = [
        j
        for j in judged
        if j.verdict in (Verdict.NEW_WITHOUT_RUBRIC, Verdict.EDITED_WITHOUT_RUBRIC)
    ]
    stale = stale_entries(judged, baseline)
    counts = {v.name.lower(): sum(1 for j in judged if j.verdict is v) for v in Verdict}
    if as_json:
        print(
            json.dumps(
                {
                    "version": VERSION,
                    "tests": len(judged),
                    **counts,
                    "stale": len(stale),
                    "declarations": len(declarations.by_name),
                    "declaration_problems": len(shared_problems),
                }
            )
        )
    else:
        for j in failures:
            print(f"{j.test.path}:{j.test.line}: {j.test.name} {j.verdict.value}:")
            for problem in j.problems:
                print(f"    {problem.text}")
        for key in stale:
            print(f"{baseline_file}: {key} is listed but is gone or now passes; run shrink")
        for shared in shared_problems:
            print(shared)
        print(
            f"test rubric: {len(judged)} tests; {counts['passes']} pass, "
            f"{counts['grandfathered']} grandfathered, {len(failures)} fail; "
            f"{len(stale)} stale baseline entries; "
            f"{len(declarations.by_name)} shared declarations, {len(shared_problems)} problems"
        )
    return 1 if failures or stale or shared_problems else 0


def command_init(repo: Path, baseline_file: Path) -> int:
    if baseline_file.exists():
        raise UsageError(f"{baseline_file} exists; the baseline is made once, then only shrinks")
    judged = judge(find_tests(repo), {}, find_declarations(repo))
    entries = {j.test.key: j.test.body_hash for j in judged if j.verdict is not Verdict.PASSES}
    write_baseline(baseline_file, entries, int(VERSION))
    print(f"test rubric: baseline of {len(entries)} tests written to {baseline_file}")
    return 0


def command_shrink(repo: Path, baseline_file: Path) -> int:
    baseline = read_baseline(baseline_file)
    stale = set(stale_entries(judge(find_tests(repo), baseline, find_declarations(repo)), baseline))
    write_baseline(
        baseline_file,
        {k: v for k, v in baseline.items() if k not in stale},
        baseline_version(baseline_file),
    )
    print(f"test rubric: {len(stale)} entries removed; {len(baseline) - len(stale)} remain")
    return 0


def command_recognize(repo: Path, baseline_file: Path) -> int:
    """Admit tests a newer checker sees, or newly refuses, that an older one passed.

    The one way a baseline may grow: only when this checker's version is newer than the
    baseline's, and only in a commit that changes nothing but the checker copy and the
    baseline, so every added test is one that existed before: merely unseen, or written
    under the older rules (version 3 refused copied State and Concurrency text). Each
    admitted test then awaits review and leaves the baseline once it passes.
    """
    made_with = baseline_version(baseline_file)
    if made_with >= int(VERSION):
        raise UsageError(
            f"{baseline_file} is already at checker version {made_with}; "
            "a baseline grows only when the checker recognizes more tests"
        )
    baseline = read_baseline(baseline_file)
    judged = judge(find_tests(repo), baseline, find_declarations(repo))
    # A test failing a rule older than this version is not admitted: an upgrade must
    # never excuse a rubric that was already missing or wrong.
    unseen = {
        j.test.key: j.test.body_hash
        for j in judged
        if j.verdict is Verdict.NEW_WITHOUT_RUBRIC
        and j.test.key not in baseline
        and all(problem.since > made_with for problem in j.problems)
    }
    write_baseline(baseline_file, {**baseline, **unseen}, int(VERSION))
    print(
        f"test rubric: checker version {made_with} to {VERSION}: "
        f"{len(unseen)} tests newly seen or newly refused admitted to {baseline_file}"
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
