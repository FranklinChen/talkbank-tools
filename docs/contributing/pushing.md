# Pushing without CI churn

**Status:** Current
**Last updated:** 2026-09-06 03:40 EDT

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
workflows. Comments and echoed suggestions cannot establish coverage. This
checker follows the repository's direct-command convention; arbitrary shell
control flow or raw tool invocations need explicit review when changing CI.

`make gate-receipts-test` exercises both the receipt protocol in temporary Git
repositories and the coverage checker. These checks also run in CI without
building a Rust test binary. Exemptions in the checker document wheel-installed
Python and frontend checks that require their own CI setup; a local receipt is
not evidence that those jobs or other platforms have passed.

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
