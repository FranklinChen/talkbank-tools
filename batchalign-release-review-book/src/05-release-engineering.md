# Chapter 5: Packaging, Versioning, and Release Engineering

## This Is Currently the Weakest Part of the Project

The code is ahead of the release engineering.

That is the reverse of what a public research tool needs.

## Broken Packaging Contract

The package metadata says:

- `pyproject.toml:83-106`

The runtime entrypoint says:

- `batchalign/_cli.py:55-90`

The release workflow says:

- `.github/workflows/release.yml:44-63`

These pieces do not currently compose into a trustworthy artifact pipeline.

The clean-source reproduction proved that a wheel can be built, installed, and
still fail at the first user command because the packaged Rust CLI binary is
missing.

That single issue is enough to block release.

## Versioning Is Incoherent

Examples:

- `pyproject.toml` is `1.0.0`
- `batchalign/version` says `First public release`
- desktop package metadata is `0.1.0`
- several internal crates remain `0.1.0`
- a stale script still expects `cli-pyproject.toml`:
  `scripts/check_cli_version_sync.py:14-26`

Internal crate versions being lower is not inherently wrong. The problem is
that the repository does not clearly distinguish:

- internal package versions
- public product versions
- dormant product-surface versions
- synchronized release candidates

## License Metadata Is Incoherent

Examples:

- top-level LICENSE is BSD-3-Clause
- `pyproject.toml:10` says BSD-3-Clause
- `Cargo.toml:13-16` says workspace license `MIT`

This is unacceptable for public release. Even if there is a good reason, the
reason is not encoded in a release-grade way.

## Public Release State Is Self-Contradictory

Examples:

- `pyproject.toml:19-22` claims `Production/Stable` and `OS Independent`
- `batchalign/version:1-3` says first public release
- `installers/README.md:51-57` says public release remains on hold
- `installers/README.md:64-67` says the desktop surface is dormant and not functional

The release narrative is currently being written from multiple places with
different assumptions.

## There Is No Serious Internal Release Workflow Yet

This matches your own assessment.

The repo has pieces of a release process:

- workflows
- installer scripts
- smoke-test scripts
- version files
- package metadata

But it does not yet have one enforceable internal release workflow with:

- a required checklist
- required gates
- rollback criteria
- version bump procedure
- release notes source of truth
- artifact verification
- cross-platform signoff

## Action Items

- Create a release checklist that is required before any tag or PyPI publish.
- Make the wheel build self-contained and smoke-tested.
- Unify version ownership:
  - one source of truth for public product version
  - documented policy for internal component versions
- Unify license ownership across Python, Rust workspace, installers, and docs.
- Remove or fix stale scripts that refer to missing files or dead workflows.
- Add automated artifact verification jobs:
  - install wheel
  - run CLI
  - validate basic server startup
  - verify metadata
