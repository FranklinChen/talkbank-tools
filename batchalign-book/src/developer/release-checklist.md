# Release Checklist

**Status:** Current
**Last updated:** 2026-04-02 07:31 EDT

This checklist must be completed in full before any git tag or PyPI publish
of batchalign3. No gate may be skipped. If a gate cannot be satisfied,
the release is blocked until it is resolved.

## Pre-Release Gates

### 1. Version Consistency

- [ ] `pyproject.toml` version matches target release version
- [ ] `batchalign/version` file matches (version, date, description)
- [ ] `crates/batchalign-cli/Cargo.toml` version matches
- [ ] `scripts/check_cli_version_sync.py` passes
- [ ] Desktop metadata versions updated if desktop is included in the release

### 2. CI Green

- [ ] All CI jobs pass on the release branch/tag
- [ ] `uv run mypy` passes locally
- [ ] `cargo check --workspace` passes
- [ ] `uv run pytest batchalign -q` passes

### 3. Artifact Verification

- [ ] Wheel builds successfully: `uv build --wheel`
- [ ] Clean install works: install wheel in a fresh venv, run `batchalign3 --help`
- [ ] `batchalign3 version` shows correct version string
- [ ] Server starts: `batchalign3 serve` responds to `/health`

### 4. Cross-Platform

- [ ] Release workflow builds wheels for all 5 targets
- [ ] At least one smoke test per platform tier (see `PLATFORM-SUPPORT.md`)

### 5. License and Metadata

- [ ] `pyproject.toml` classifiers match actual release state
- [ ] License metadata consistent across: `pyproject.toml`, Cargo.toml workspace, `LICENSE` file
- [ ] README accurate (no overclaiming of features or platform support)

### 6. Dependencies

- [ ] `talkbank-tools` dependency pinned to a git SHA or version (not a floating path)
- [ ] `pip-audit` / `cargo deny` clean (no known vulnerabilities)
- [ ] No yanked or deprecated dependencies

### 7. Documentation

- [ ] `RELEASE-CONTRACT.md` up to date
- [ ] `PLATFORM-SUPPORT.md` up to date
- [ ] CHANGELOG or release notes drafted for this version
- [ ] API stability documentation reflects current state

## Release Procedure

1. Create release branch: `release/vX.Y.Z`
2. Complete every gate above (all boxes checked)
3. Tag: `git tag vX.Y.Z`
4. Push tag: triggers release workflow
5. Verify published artifacts (download wheel, install in clean venv, run smoke test)
6. Update `batchalign/version` for next development cycle (bump to next dev version)

## Rollback

If a release artifact is found to be broken after publish:

1. Yank the PyPI release: `uv run twine yank batchalign3==X.Y.Z`
2. Delete the GitHub release (set to draft state)
3. Fix the issue on a patch branch
4. Re-tag with an incremented patch version (never reuse a yanked version number)
5. Re-release following the full procedure above
