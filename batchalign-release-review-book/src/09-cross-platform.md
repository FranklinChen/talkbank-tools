# Chapter 9: Cross-Platform Story

## Claimed Support vs Demonstrated Support

The project claims support for Windows, macOS, and Linux.

That goal is reasonable and should be preserved. But right now the repo does
not demonstrate that level of support convincingly enough for public release.

## Current Evidence Gap

### CI

The main test workflow runs on Ubuntu by default:

- `.github/workflows/test.yml:11-62`

Release builds include platform wheel targets, but the release workflow itself
is manual:

- `.github/workflows/release.yml:3-20`

That is not the same as an always-on cross-platform validation story.

### Process Lifecycle

Core lifecycle code uses Unix-specific process hooks:

- `worker/handle/mod.rs:191-204`
- `worker/handle/mod.rs:1026-1071`
- `cli/daemon.rs:477-485`
- `cli/serve_cmd.rs:211-220`

Again, this may be correct where it applies. It is still evidence that the
cross-platform semantics are not yet unified or sufficiently proven.

### Tooling

A large amount of developer/release tooling is Bash-first:

- installer test scripts
- drift checks
- dashboard build helpers
- release-prep scripts

That is workable for maintainers. It is not a strong foundation for a
cross-platform release pipeline unless the Windows/macOS equivalents are
actually exercised.

### Metadata Drift

`pyproject.toml` currently claims `Operating System :: OS Independent`:

- `pyproject.toml:18-28`

That classifier is not credible for the current system.

## Desktop Surface

The desktop path is explicitly documented as dormant:

- `installers/README.md:64-67`

The Tauri desktop metadata is also still at `0.1.0`, not aligned with the main
product release:

- `apps/dashboard-desktop/package.json`
- `apps/dashboard-desktop/src-tauri/tauri.conf.json`

That is fine for an experimental side surface. It is not fine if the team wants
to treat it as a near-term release channel without saying so clearly.

## Action Items

- Define platform support tiers:
  - Tier A: fully supported and CI-gated
  - Tier B: build-only
  - Tier C: experimental
- Remove the `OS Independent` classifier until it is justified.
- Add platform smoke jobs that install and run the public artifact.
- Treat process lifecycle and shutdown behavior as a platform-owned test area.
- Keep the desktop app clearly experimental until its versioning, functionality,
  and update path are real.
