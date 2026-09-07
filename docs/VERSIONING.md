# Versioning Policy

**Status:** Current
**Last updated:** 2026-09-06

This repository owns the Batchalign product. Chatter owns the CHAT core in
[TalkBank/chatter](https://github.com/TalkBank/chatter); its versions and release
contracts are independent. This workspace consumes the Chatter release tags
recorded in `Cargo.toml` and `Cargo.lock`.

## Canonical version sources

| Surface | Authority | Relationship |
|---|---|---|
| `batchalign3` Python package, bundled CLI, local server and dashboard | `pyproject.toml` `[project].version` | Public preview product version, distributed through GitHub Releases |
| Product Rust build metadata | `Cargo.toml` `[workspace.package].version` and `crates/batchalign/Cargo.toml` | Must match the product version; workspace-inheriting crates follow it automatically |
| Runtime banner | `batchalign/version` | Mirrors the product version; build identity distinguishes different source builds |
| Internal Rust/PyO3 crates | Their `Cargo.toml` manifests | Unpublished implementation crates; independent versions stay independent |
| Experimental desktop shell | `apps/dashboard-desktop/package.json` and its `src-tauri/Cargo.toml` | Separate experimental version; normal BA3 releases do not bump it |

Read current versions from these sources instead of maintaining another table
of version literals. Matching versions do not create a public Rust API contract.

## Pre-1.0 policy

The Batchalign product remains public preview. No surface is promoted to 1.0
without an explicit compatibility decision and coordinated changes to
[the release contract](RELEASE-CONTRACT.md) and release procedures.

- **Patch (`0.x.y`):** bug fixes, packaging corrections and documentation updates
  that do not require consumers to change their integrations.
- **Minor (`0.y.0`):** new user-visible features, integration-relevant behavior
  shifts or breaking changes. Describe affected commands and data formats in
  release notes.
- **1.0:** an explicit stability promise, not a consequence of accumulating
  features or successful builds.

After a surface is explicitly stable at 1.0, patches contain fixes, minor
versions add backward-compatible features, and major versions signal breaking
changes with a documented migration path.

## Preparing a product release

1. Choose the version based on the changed public surface.
2. Update `pyproject.toml`, the workspace version, the explicit `batchalign`
   crate version, and `batchalign/version` together. Refresh Cargo and uv locks.
   Do not mechanically bump independently versioned internal crates or desktop
   metadata.
3. Keep Chatter dependencies pinned to a published upstream tag. Local path
   patches are for co-development and must not enter a release commit.
4. Review generated schemas and documentation. Version evidence formats when
   their consumers need to distinguish a changed interpretation or shape.
5. Complete the [release checklist](../book/src/batchalign/developer/release-checklist.md),
   including a prospective-tag dry run, wheel smoke tests and published-download
   verification. BA3 releases are GitHub Releases, never PyPI publications.
6. Squash commits since the last push, push rarely, and preserve published
   history. Never move or reuse a published release tag.

Use the build identity, rather than the product version alone, to compare
installed source builds. A package version and a source-build identity answer
different questions.
