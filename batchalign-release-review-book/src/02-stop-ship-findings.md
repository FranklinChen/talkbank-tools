# Chapter 2: Stop-Ship Findings

These findings should block public release.

## 1. Clean Wheel Builds Can Ship a Broken CLI

The public package entrypoint is defined as:

- `pyproject.toml:83-106`
- `batchalign/_cli.py:52-90`

The entrypoint requires a packaged binary under `batchalign/_bin/batchalign3`,
but the release workflow builds wheels using `maturin-action` against
`pyo3/Cargo.toml`:

- `.github/workflows/release.yml:44-63`

In a clean exported source snapshot, I reproduced this exact failure:

1. export tracked source with `git archive`
2. build wheel
3. install wheel into a clean `uv tool` sandbox
4. run `batchalign3 --help`

Observed result:

```text
batchalign3 CLI binary not found. Reinstall batchalign3 or, in a source checkout, run `cargo build -p batchalign-cli`.
```

That is a release blocker. It means the artifact contract is not actually
tested or guaranteed.

### Action Items

- Make the wheel build self-contained:
  - either build and embed the Rust CLI during the wheel build
  - or remove the packaged-binary dependency and use a single supported entry path
- Add a release smoke test that installs the built wheel and runs:
  - `batchalign3 --help`
  - `batchalign3 version`
- Make that smoke test mandatory in CI, not a manual script.

## 2. CI Does Not Gate Ordinary Development

The main CI workflow triggers on tag pushes and manual dispatch:

- `.github/workflows/test.yml:3-7`

That means the project can accumulate breakage on normal branches and even on
mainline development without the primary CI pipeline running automatically.

For software that must be stable for researchers, this is not acceptable.

### Action Items

- Add `pull_request` and ordinary branch push triggers to the main CI workflow.
- Make merge-to-main impossible without the required checks passing.
- Split optional heavy ML lanes from mandatory correctness lanes, but do not
  leave the mandatory lanes disabled by default.

## 3. The Repository Is Not Independently Releasable

The Rust workspace depends on sibling local paths from `talkbank-tools`:

- `crates/batchalign-chat-ops/Cargo.toml:22-24`
- `README.md:215-223`
- repeated workflow steps cloning `../talkbank-tools`

This is not just a developer convenience issue. It means:

- builds are not hermetic
- release versioning is split across repos
- licensing/compliance tracking is split across repos
- reproducibility depends on the state of another unreleased codebase

If `talkbank-tools` is not independently released, `batchalign3` is not really
independently releasable either.

### Action Items

- Choose one release boundary:
  - release `talkbank-tools` crates properly and depend on versioned artifacts
  - or vendor the required crates into this repository
  - or move to pinned git dependencies with an explicit release manifest
- Document the compatibility contract between the two repos.
- Require synchronized release notes when cross-repo boundary changes happen.

## 4. Release Signals Contradict Each Other

Examples:

- `batchalign/version:1-3` says `1.0.0` and `First public release`
- `pyproject.toml:19-28` claims `Production/Stable`
- `installers/README.md:51-57` says public release work remains on hold
- `book/src/developer/api-stability.md:25-39` says compatibility is still
  architecture-first and not a frozen public contract

This is dangerous because it produces false confidence inside the team as well
as outside it.

### Action Items

- Publish one authoritative release-readiness state machine:
  - internal experimental
  - internal releasable
  - public beta
  - public stable
- Require metadata, docs, and workflows to agree on the current state.
- Remove or correct contradictory "stable" and "public release" claims until
  the release gate truly exists.

## 5. Cross-Platform Claims Outrun Evidence

The project says it runs on macOS, Windows, and Linux, but:

- the main CI test workflow is Linux-only by default
- much tooling is Bash-based
- core process-lifecycle code uses Unix-only hooks such as `pre_exec`,
  `setsid`, and `killpg`
- the desktop app is explicitly documented as dormant and not functional for
  end users

This is not a reason to abandon platform support. It is a reason not to claim
release-grade support yet.

### Action Items

- Define supported platform tiers explicitly.
- Add smoke lanes for macOS and Windows that install and run the actual
  supported artifact.
- Do not advertise a platform as supported until artifact build, install,
  startup, and one minimal command pass in CI.
