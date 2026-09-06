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
                checker, GATE=gate, MAKEFILE=makefile, WORKFLOWS=workflows, EXEMPT={}
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

    def test_prerequisite_cycle_terminates(self):
        recipes = makefile_recipes("first: second\nsecond: first\n")
        self.assertEqual(reachable({"first"}, recipes), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
