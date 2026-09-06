#!/usr/bin/env python3
"""Fail if the receipt-producing local gate does not cover CI's named targets.

scripts/gate.sh runs the former pre-push checks before a push begins. The hook
then verifies its receipt against every actual pushed tree. This checker follows
make targets transitively from the gate script, so moving verification out of
the hook does not weaken CI coverage. Platform and wheel-only exemptions remain
explicit below. The receipt protocol is tested by make gate-receipts-test.

Usage: python3 scripts/check_push_gate_sync.py
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "gate.sh"
MAKEFILE = REPO / "Makefile"
WORKFLOWS = REPO / ".github" / "workflows"


@dataclass(frozen=True)
class HookExemption:
    """A CI target the local hook may skip for the stated reason."""

    reason: str


@dataclass(frozen=True)
class RecipeInvariantExemption:
    """A skipped target whose recipe must retain a promised property."""

    reason: str
    required_recipe_prefix: str


Exemption = HookExemption | RecipeInvariantExemption


#: Targets a workflow may run that the hook is not expected to.
#:
#: Each needs a REASON, because an unexplained exemption is how the drift
#: starts again. Keep this list short; if it grows, the hook is the thing that
#: should change.
#:
#: This list is also the honest statement of what a green hook does NOT prove.
#: Everything here is covered only once CI runs, which is why anything touching
#: these areas goes to a branch first (docs/contributing/pushing.md).
EXEMPT: dict[str, Exemption] = {
    # Builds the dashboard's JS bundle. Needs `npm ci` against the network and
    # is not a correctness gate on committed Rust.
    "batchalign-dashboard-build": HookExemption(
        "network npm install, not a correctness gate"
    ),
    # Runs in its own workflow job with its own Linux toolchain setup; the hook
    # covers the Rust side that can fail from committed content.
    "batchalign-build-pyo3": HookExemption("separate job with its own toolchain setup"),
    # Everything below needs `batchalign-python-prepare`, i.e. a maturin release
    # wheel built and installed into the dev environment. That is minutes, so
    # putting it in the hook would recreate the pressure that produced the
    # hand-written subset the hook used to be. The SOURCE-only half of the
    # Python gate (`batchalign-lint-python-source`) is deliberately split out
    # and IS in the hook.
    "batchalign-ci-python": HookExemption("needs a built wheel; minutes, not seconds"),
    "batchalign-lint-python": HookExemption(
        "needs a built wheel; use -source in the hook"
    ),
    "batchalign-typecheck-python": HookExemption(
        "needs a built wheel; mypy against the install"
    ),
    "batchalign-test-python": HookExemption("needs a built wheel"),
    "batchalign-python-prepare": HookExemption("builds the wheel these depend on"),
    "batchalign-build-wheel": RecipeInvariantExemption(
        "maturin release build",
        required_recipe_prefix="uv run --frozen --only-dev maturin build --release",
    ),
    "batchalign-build-ci-wheel": HookExemption(
        "CI-only wheel requires the same-commit binary artifact"
    ),
    # NOT here any more: `batchalign-ipc-schema-check`. It was exempt for
    # "needs the built binary", which is true in isolation and irrelevant in
    # the hook, where `batchalign-ci-rust` has already built it and the check
    # costs 0 s. It is in the hook now, after a docstring edit regenerated six
    # IPC schemas and CI caught what every local gate had passed.
    #
    # Drift checks against generated artifacts that need the built binary or the
    # npm-installed frontend, and run in their own jobs with that setup.
    # NARROWED 2026-08-27. Only the `npx openapi-typescript` half needs npm.
    # The `openapi.json` half reuses the current development binary and is in
    # the hook as
    # `batchalign-dashboard-schema-check`; a stale openapi.json reached main
    # through this exemption, the same way `batchalign-ipc-schema-check` did
    # before it was un-exempted above.
    "batchalign-dashboard-api-check": HookExemption(
        "TypeScript half needs the npm-installed frontend"
    ),
    "batchalign-runtime-check": HookExemption("runs in the wheel-installed job"),
}


def make_targets(text: str) -> set[str]:
    """Read direct make commands, including YAML run lines and env assignments.

    This deliberately handles the repository's direct-command convention, not
    arbitrary shell execution. Quoted prose and comments confer no coverage.
    """
    targets: set[str] = set()
    for line in text.splitlines():
        command = line.strip().removeprefix("run:").strip().lstrip("@")
        command = command.replace("$(MAKE)", "make")
        try:
            words = shlex.split(command, comments=True)
        except ValueError:
            continue  # YAML prose and continued shell lines are not commands.
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words.pop(0)
        if not words or words.pop(0) != "make":
            continue
        for word in words:
            if word in {";", "&&", "||", "|"}:
                break
            if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", word):
                targets.add(word)
    return targets


def _is_comment(line: str) -> bool:
    """Whether a Makefile, recipe, or shell line is wholly a comment."""
    stripped = line.strip()
    if stripped.startswith("@#"):
        return True
    return stripped.startswith("#")


class Uncovered(NamedTuple):
    """A gate CI runs on a push that the gate does not reach.

    Named rather than a bare pair because both fields are strings and the
    reporting loop had them in scope beside a `Path` of the same name.
    """

    workflow: str
    target: str


@dataclass(frozen=True)
class Recipe:
    """Keep prerequisite evidence separate from executable recipe text."""

    prerequisites: frozenset[str]
    commands: tuple[str, ...]


def makefile_recipes(text: str | None = None) -> dict[str, Recipe]:
    """Parse this Makefile's single-target rules and direct prerequisites."""
    recipes: dict[str, Recipe] = {}
    current: str | None = None
    if text is None:
        text = MAKEFILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("\t"):
            if current is not None:
                recipe = recipes[current]
                recipes[current] = Recipe(
                    recipe.prerequisites,
                    (*recipe.commands, line.strip().lstrip("@")),
                )
            continue
        match = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*:(?!=)", line)
        current = match.group(1) if match else None
        if current is not None:
            prerequisites = line.partition(":")[2].partition("#")[0].split()
            previous = recipes.get(current, Recipe(frozenset(), ()))
            recipes[current] = Recipe(
                previous.prerequisites | frozenset(prerequisites), previous.commands
            )
    return recipes


def reachable(roots: set[str], recipes: dict[str, Recipe]) -> set[str]:
    """Every target reached from `roots`, following recipes and prerequisites.

    A prerequisite is named bare (`foo: bar`) while a recursive call is written
    `$(MAKE) bar`, and `make_targets` only sees the latter, so prerequisites are
    folded in from the header line by `makefile_recipes`.
    """
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        target = pending.pop()
        if target in seen:
            continue
        seen.add(target)
        body = recipes.get(target)
        if body is None:
            continue
        pending.extend(make_targets("\n".join(body.commands)) & recipes.keys())
        pending.extend(body.prerequisites & recipes.keys())
    return seen


def invalid_exemption_invariants(recipes: dict[str, Recipe]) -> list[str]:
    """Return exemptions whose executable recipe no longer earns its reason.

    Whole-line comments are deliberately excluded and a real recipe line must
    start with the promised command. Otherwise a comment or `echo` can keep
    this gate green after the actual build regresses to a debug PEP 517 profile.
    """
    invalid: list[str] = []
    for target, exemption in EXEMPT.items():
        if not isinstance(exemption, RecipeInvariantExemption):
            continue
        required = exemption.required_recipe_prefix
        recipe = recipes.get(target, Recipe(frozenset(), ()))
        executable_lines = [
            line.strip() for line in recipe.commands if not _is_comment(line)
        ]
        if not any(line.startswith(required) for line in executable_lines):
            invalid.append(
                f"make {target}: exemption requires executable recipe prefix "
                f"{required!r}"
            )
    return invalid


def is_push_triggered(text: str) -> bool:
    """Does this workflow run on a push to main, or on a PR against it?

    Those are the runs a developer can turn red by pushing, so they are exactly
    the ones the hook is supposed to predict. Read as text rather than parsed:
    the alternative is a YAML dependency in a script whose whole job is to be
    runnable from a git hook on any checkout, and the two trigger keys are
    unambiguous at the top level of every workflow here.
    """
    header = text.split("\njobs:", 1)[0]
    return "\n  push:" in header or "\n  pull_request:" in header


def main() -> int:
    if not GATE.is_file():
        print(f"missing {GATE}", file=sys.stderr)
        return 2

    recipes = makefile_recipes()
    invalid_invariants = invalid_exemption_invariants(recipes)
    if invalid_invariants:
        print("CI exemption invariants are false:", file=sys.stderr)
        for invalid in invalid_invariants:
            print(f"  {invalid}", file=sys.stderr)
        return 1

    hook_roots = make_targets(GATE.read_text(encoding="utf-8")) & recipes.keys()
    if not hook_roots:
        print(
            "the local gate invokes no make target at all; either it was "
            "rewritten to call cargo directly (in which case this check needs "
            "updating) or it is not gating anything",
            file=sys.stderr,
        )
        return 1
    hook_targets = reachable(hook_roots, recipes)

    missing: list[Uncovered] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        # EVERY push-triggered workflow, not just the Rust one. Scoping this to
        # `batchalign-ci-rust` is what let a `ruff format --check` failure reach
        # main on 2026-08-14: the Python workflow was outside the scan, so its
        # gates were never compared against the hook at all. A workflow that
        # does not run on a push (release, the scheduled dependency audit) is
        # correctly out of scope, and that is now decided by its own triggers
        # rather than by which target it happens to name.
        if not is_push_triggered(text):
            continue
        # Intersected with the Makefile's own targets, so English prose in a
        # workflow comment ("make target", "make each ...") is not read as an
        # invocation. A workflow naming a target that does not exist fails in
        # CI on the first run and is not the silent drift this file is for.
        for target in sorted(make_targets(text) & recipes.keys()):
            if target in hook_targets or target in EXEMPT:
                continue
            missing.append(Uncovered(workflow=workflow.name, target=target))

    if missing:
        print("local gate does not run what CI runs:", file=sys.stderr)
        for gap in missing:
            print(
                f"  {gap.workflow} runs 'make {gap.target}', the gate does not",
                file=sys.stderr,
            )
        print(
            "\nAdd it to scripts/gate.sh, or add it to EXEMPT in this file "
            "WITH a reason.",
            file=sys.stderr,
        )
        return 1

    covered = ", ".join(sorted(hook_targets))
    print(f"local gate runs CI's targets ({covered})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
