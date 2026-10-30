"""Regression checks for evidence accepted by the local CI coverage guard."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_push_gate_sync as checker
from check_push_gate_sync import make_targets, makefile_recipes, reachable


class CoverageEvidenceTests(unittest.TestCase):
    def test_delegated_release_needs_command_and_reached_contract_tests(self):
        command = (
            "uv run --frozen --only-dev python -m "
            "scripts.build_python_extension wheel-release"
        )
        exemption = checker.RecipeInvariantExemption(
            "delegated optimized build", command, "workflow-contract"
        )
        for recipe, gate_targets, valid in [
            (command, {"workflow-contract"}, True),
            (command, set(), False),
            (f"# {command}", {"workflow-contract"}, False),
            (f'echo "{command}"', {"workflow-contract"}, False),
            (
                command.replace("wheel-release", "wheel-dev"),
                {"workflow-contract"},
                False,
            ),
            ("uv build", {"workflow-contract"}, False),
        ]:
            with self.subTest(recipe=recipe, gate_targets=gate_targets):
                recipes = makefile_recipes(f"wheel:\n\t{recipe}\n")
                with patch.object(checker, "EXEMPT", {"wheel": exemption}):
                    errors = checker.invalid_exemption_invariants(recipes, gate_targets)
                self.assertEqual(not errors, valid)

    def test_checker_refuses_ci_target_mentioned_only_in_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.sh"
            makefile = root / "Makefile"
            workflows = root / "workflows"
            workflows.mkdir()
            gate.write_text("make root\n")
            makefile.write_text(
                'root:\n\t@echo "make missing"\n\t@# missing\nmissing:\n'
            )
            (workflows / "ci.yml").write_text(
                "on:\n  push:\njobs:\n  test:\n    run: make missing\n"
            )
            with patch.multiple(
                checker,
                GATE=gate,
                MAKEFILE=makefile,
                WORKFLOWS=workflows,
                EXEMPT={},
                RAW_VERIFIERS={},
            ):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(checker.main(), 1)
                self.assertIn("runs 'make missing'", errors.getvalue())
                gate.write_text("make root\nmake missing\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(checker.main(), 0)

    def test_recipe_prose_cannot_claim_a_missing_gate(self):
        recipes = makefile_recipes(
            "root: prerequisite # missing\n"
            "\t@# missing is normally run by make missing\n"
            '\t@echo "make missing"\n'
            "\t@echo missing\n"
            "\t@$(MAKE) actual\n"
            "prerequisite:\n"
            "actual:\n"
            "missing:\n"
        )
        self.assertEqual(
            reachable({"root"}, recipes), {"root", "prerequisite", "actual"}
        )

    def test_gate_and_workflow_command_forms(self):
        self.assertEqual(
            make_targets(
                "  run: make lint-shell\n"
                'BATCHALIGN_BIN="$PWD/target/debug/batchalign3" make schema\n'
                "@$(MAKE) _python rust # make never\n"
                'echo "make imaginary"\n'
                "# make commented\n"
            ),
            {"lint-shell", "schema", "_python", "rust"},
        )

    def test_raw_verifier_step_is_refused(self):
        """The step that let a stale stub reach main on 2026-10-21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.sh"
            makefile = root / "Makefile"
            workflows = root / "workflows"
            workflows.mkdir()
            gate.write_text("make typecheck\n")
            makefile.write_text("typecheck:\n\tuv run --no-sync mypy\n")
            workflow = workflows / "python.yml"
            workflow.write_text(
                "on:\n  push:\njobs:\n  types:\n    steps:\n"
                "      - name: Strict mypy\n"
                "        run: uv run --no-sync mypy\n"
            )
            with patch.multiple(
                checker,
                GATE=gate,
                MAKEFILE=makefile,
                WORKFLOWS=workflows,
                EXEMPT={},
                RAW_VERIFIERS={},
            ):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(checker.main(), 1)
                self.assertIn("python.yml: uv run --no-sync mypy", errors.getvalue())
                workflow.write_text(
                    workflow.read_text().replace(
                        "uv run --no-sync mypy", "make typecheck"
                    )
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(checker.main(), 0)

    def test_stale_raw_verifier_exemption_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "gate.sh"
            makefile = root / "Makefile"
            workflows = root / "workflows"
            workflows.mkdir()
            gate.write_text("make root\n")
            makefile.write_text("root:\n")
            (workflows / "ci.yml").write_text(
                "on:\n  push:\njobs:\n  test:\n    run: make root\n"
            )
            gone = {"pytest gone": checker.HookExemption("was here once")}
            with patch.multiple(
                checker,
                GATE=gate,
                MAKEFILE=makefile,
                WORKFLOWS=workflows,
                EXEMPT={},
                RAW_VERIFIERS=gone,
            ):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(checker.main(), 1)
                self.assertIn("pytest gone", errors.getvalue())

    def test_verifiers_are_found_by_the_program_invoked(self):
        for command, verifies in [
            ("uv run --no-sync mypy", True),
            ("uv run --no-sync python -m pytest batchalign", True),
            ("python3 -m unittest scripts/test_x.py", True),
            ("cargo test -p crate", True),
            ("cargo build -p crate", False),
            ("mdbook test", True),
            ("mdbook build", False),
            ("uv run --no-sync batchalign3 ipc-schema --check --output x/", True),
            ("uv run --no-sync python scripts/check_runtime_drift.py", True),
            ('curl -fsSL "https://x/shellcheck.tar.xz" -o shellcheck.tar.xz', False),
            ('sudo install "shellcheck-1/shellcheck" /usr/local/bin/shellcheck', False),
            ("SERVER=1 ./target/debug/batchalign3 serve start", False),
            ("batchalign3 version", True),
            ("bash scripts/run_react_dashboard_smoke.sh", True),
            ("./scripts/server_startup_smoke.sh", True),
            ("uv run --no-sync python -m scripts.verify_hub_selection_loads", True),
            ("bash -c 'echo scripts/x'", False),
            ("cp installers/install-batchalign3.sh dist/", False),
        ]:
            with self.subTest(command=command):
                self.assertEqual(checker.is_verifier(command), verifies)

    def test_run_blocks_skip_heredoc_bodies_and_comments(self):
        text = (
            "    steps:\n"
            "      - run: uv run --no-sync mypy\n"
            "      - run: |\n"
            "          # pytest in a comment\n"
            "          python3 - <<'PY'\n"
            "          import pytest\n"
            "          PY\n"
            "          cargo test -p x\n"
            "      - name: next\n"
        )
        self.assertEqual(
            checker.run_commands(text),
            ["uv run --no-sync mypy", "python3 - <<'PY'", "cargo test -p x"],
        )

    def test_prerequisite_cycle_terminates(self):
        recipes = makefile_recipes("first: second\nsecond: first\n")
        self.assertEqual(reachable({"first"}, recipes), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
