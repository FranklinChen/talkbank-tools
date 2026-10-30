# Pushing without CI churn

**Status:** Current
**Last updated:** 2026-10-21 17:09 EDT

Run the verification once, then push the content it verified:

```bash
make install-hooks  # once, and after another tool replaces Git's hook
make gate
git add <changed-files>
git commit
git push
```

`make gate` records a receipt only after all checks succeed and the source tree
still matches the tree captured before checking. A new attempt invalidates any
previous receipt. Committing the same content preserves the receipt; editing
content requires another gate. Add new scripts to the index before running the
gate so the tracked-script ShellCheck inventory includes them.

The pre-push hook performs no compilation. It checks the current content and
every actual pushed commit tree against the receipt, including annotated tags.
A verified dirty tree cannot authorize an older, different commit. The receipt
covers Git content, including nonignored new files, rather than timestamps or a
list of changed paths. It does not attest to external tool versions or prevent
an edit followed by restoration while a check is running.

## Hook installation and local checks

`make install-hooks` installs `scripts/pre-push.sh` in `.git/hooks/pre-push`.
If `core.hooksPath` is configured, make sure Git resolves that installed hook.
An executable `.git/hooks/pre-push.local` runs after receipt verification and
receives Git's original ref stream unchanged. A failed receipt check stops the
push before the local hook runs.

## The gate invokes CI's targets

`scripts/gate.sh` invokes `make batchalign-ci-rust` and the source-only Python,
schema, shell, workflow, and book checks previously run during every push.
This avoids repeating successful verification during the commit/push cycle;
it does not replace those checks with a smaller subset.

`scripts/check_push_gate_sync.py`, run by `make lint`, follows actual Makefile
prerequisites and direct recursive make commands. It checks all push/PR-triggered
workflows. Comments and echoed suggestions cannot establish coverage.

It enforces two rules, so neither depends on review:

- **A workflow target the gate does not reach fails**, unless `EXEMPT` gives
  the reason (today: checks that need a built wheel or the npm frontend).
- **A workflow step that runs a verifier directly fails.** A verifier is found
  by the program a command invokes (mypy, stubtest, pytest, ruff, shellcheck,
  actionlint, lychee, unittest, `cargo test`/`clippy`/`fmt`/`llvm-cov`/`deny`,
  `mdbook test`), by a `--check` flag, by running any script under `scripts/`
  (directly or through `bash`, `sh`, `python` or `python -m scripts.*`), or
  by running `batchalign3` itself other than `serve` (which starts a fixture
  a later step checks), looking through `VAR=value`, `sudo`, `uv run` and
  `python -m`. Such a step must call a
  make target instead, so the command has one owner the gate can be compared
  against, or be listed in `RAW_VERIFIERS` by exact text with its reason. An
  entry no workflow uses any more also fails.

The second rule exists because a raw step is invisible to the first. A raw
`ruff` step let a formatting failure reach main on 2026-08-14, and a raw
`uv run mypy` step let a stub missing a new PyO3 function reach main on
2026-10-21; each time the gate was green. Until 2026-10-30 only
`scripts/check_*` scripts counted, so `bash scripts/run_react_dashboard_smoke.sh`,
the whole Playwright dashboard suite, ran in CI with neither gate coverage nor
an exemption, and a failure in it reached main on 2026-10-25.

`make gate-receipts-test` exercises both the receipt protocol in temporary Git
repositories and the coverage checker. These checks also run in CI without
building a Rust test binary. Exemptions in the checker document wheel-installed
Python and frontend checks that require their own CI setup; a local receipt is
not evidence that those jobs or other platforms have passed.

**A change to the PyO3 extension** (`crates/batchalign-pyo3`, or anything a
`#[pyfunction]` exposes) needs `make batchalign-test-python` before the push as
well as the gate. It builds the wheel, runs `make batchalign-stubtest`, which
checks the hand-written `stubs/batchalign_core/` against the built extension,
and runs the Python suite against it. The gate's strict mypy trusts the stub;
stubtest is what makes the stub true.

## What local checks cannot catch: the runner is Linux

CI runs `ubuntu-latest`; development happens on macOS. Anything behind
`cfg(target_os)` is invisible to every local command, and two of the three
failures were exactly that:

- The Tauri desktop crate depends on `glib` on Linux and not on macOS, so
  `cargo clippy --workspace` passed locally and failed on the runner. (Fixed by
  scoping the lint to the crates CI gates, which is why every other CI target
  in the Makefile enumerates crates instead of using `--workspace`.)
- `dashboard_auto_open_enabled` is called only inside a
  `cfg(target_os = "macos")` block, so it is live locally and dead code on the
  runner, which `-D warnings` rejects.

**No amount of local gating catches these.** If you want to reproduce one, flip
the target conditions in the file and run clippy: changing `target_os = "macos"`
to `"linux"` makes the macOS branch inactive and reproduces the runner's view.
That is how the second failure above was diagnosed and its fix verified.

## So: let CI see it before `main` does

For any change that touches platform-conditional code, build configuration,
lint configuration, or the workspace's crate set:

1. Push the work to a branch, not `main`.
2. Let the Rust workflow run.
3. Fast-forward `main` only once it is green.

`main` then only ever receives commits CI has already validated, which is what
keeps its history free of "fix CI" commits. For ordinary changes to crate
internals, the hook is sufficient and a direct push to `main` is fine.

## A gate must test the commit, not the day

The fourth red run that afternoon was different in kind from the other three,
and it is the one worth remembering. Nothing in the push was wrong: an advisory
had been published against a transitive dependency since the previous run, and
`pip-audit` exits non-zero on any advisory outside its allowlist.

An advisory can appear without a source change. Treat it as a security finding
that needs assessment and an upgrade when a fix is available, rather than
assuming that a successful source gate proves the dependencies safe. Pinned
model revisions constrain which checkpoints load; they do not make a vulnerable
checkpoint-loading path unreachable.

So dependency advisories moved to `.github/workflows/dependency-audit.yml`,
weekly plus `workflow_dispatch`. The information is unchanged and Dependabot
still opens PRs for anything with a real fix; what is gone is the blocking.

**The test to apply before adding anything to `ci.yml`:** can a developer make
this pass before pushing? If the answer depends on the state of the world rather
than the state of the commit, it is a report, not a gate, and it belongs on a
schedule.

## Doctests are a separate compilation

`cargo test --lib` and `cargo clippy --all-targets` do not see them.
`make batchalign-ci-rust` does, via `batchalign-test-rust`. The third failure
was a doctest still passing `&["cha"]` to a function whose parameter had become
a typed `InputKind`, which no `--lib` or `--all-targets` run could have caught.
