# Contributing to talkbank-tools

**Status:** Current
**Last updated:** 2026-09-06 03:40 EDT

Thank you for contributing.

This repository owns Batchalign3: the ML pipeline, Python package,
`batchalign-*` crates, dashboard, and PyO3 bridge. The native CLI lives in
`crates/batchalign/`.

CHAT grammar, specification, parsers, model, validation, Chatter CLI, and CLAN
commands belong to the separate [Chatter repository](https://github.com/TalkBank/chatter).
Batchalign consumes its release-pinned crates. Make CHAT-format changes and
regenerate grammar walkers there; do not recreate that toolchain here.

Start with [README.md](README.md) and the [pushing guide](docs/contributing/pushing.md).

## Development Setup
1. Install Rust (stable).
2. Install Node.js (for frontend tooling).
3. Install `uv` for the Python/Batchalign surfaces.

Core commands:
```bash
make check
make test
make verify
make batchalign-check
make batchalign-test-rust
make batchalign-test-integration
make batchalign-test-python
make batchalign-typecheck-python
make batchalign-ci-python
make ci-local
make ci-full
make gate
make chat-anchors-check
```

For the full target list:

```bash
make help
```

For xtask helpers:

```bash
cargo run -q -p xtask -- help
```

`make chat-anchors-check` validates all `CHAT.html#...` links in `crates/`, `schema/`, and `docs/` against the published CHAT manual at `https://talkbank.org/0info/manuals/CHAT.html` by default.
To validate against a local mirror instead, pass:

```bash
CHAT_HTML_PATH=/abs/path/to/CHAT.html make chat-anchors-check
```

Without `CHAT_HTML_PATH`, the script fetches from:

```bash
CHAT_HTML_URL=https://talkbank.org/0info/manuals/CHAT.html make chat-anchors-check
```

This check is now part of required CI gates.

## Before Opening a PR

Run `make gate`, commit the verified content, then push. The installed hook
verifies that each pushed tree has a matching successful receipt. Changes after
the gate require another run. For packaging and Python runtime changes, also run
`make batchalign-ci-python`, which builds and installs the wheel for its checks.
Platform, build, lint-configuration, and crate-set changes go through branch CI
before advancing `main`; see the pushing guide for the local gate's limits.

If you changed the dashboard frontend, also run:

```bash
make batchalign-dashboard-api-check  # Verify API types are in sync
make batchalign-dashboard-e2e        # Quick mock-server e2e tests
```

For a comprehensive confidence check before a dashboard PR, also run:

```bash
make batchalign-dashboard-build      # Verify build completes
make batchalign-dashboard-e2e-real   # Integration tests with real server
```

If you need broader confidence for cross-cutting changes, also run:
```bash
make ci-local
make ci-full
```

## Generated Files
Do not hand-edit generated artifacts.
Regenerate them from their source inputs and include the generated updates in the same PR.

## Pull Request Expectations
Include:
- what changed and why,
- which subsystem(s) were touched,
- tests run,
- whether generated files changed.
- whether docs were updated (or why not),
- whether integrator/API behavior changed (or why not).

## Documentation Expectations
Update docs in the same PR when behavior, workflows, or contracts change.

Key doc surfaces:

- `book/`: the Batchalign developer and user documentation under `book/src/`.
- crate READMEs for component-specific entrypoints

## Reporting Bugs
Open an issue with:
- minimal reproduction,
- expected behavior,
- actual behavior,
- relevant files and commands.
