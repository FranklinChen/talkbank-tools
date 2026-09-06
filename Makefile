.PHONY: gate gate-receipts-test help hooks-check lint fmt-check lint-shell lint-actionlint test test-affected batchalign-check batchalign-test-rust batchalign-test-integration batchalign-test-ml-golden batchalign-build-pyo3 batchalign-build-wheel batchalign-build-ci-wheel batchalign-python-prepare batchalign-test-python batchalign-typecheck-python batchalign-ci-python batchalign-runtime-check batchalign-dashboard-api-check batchalign-dashboard-schema-check batchalign-dashboard-build batchalign-dashboard-e2e batchalign-dashboard-e2e-real batchalign-ci-rust build clean check check-affected lint-affected verify book-check book book-serve smoke ci-local ci-full install-hooks _batchalign-test-python _batchalign-typecheck-python audit-status audit-streak audit-scan audit-flag-staleness audit-prose-references

help:
	@echo "talkbank-tools task index (batchalign3 workspace)"
	@echo ""
	@echo "Core workflow:"
	@echo "  make check                 Fast compile check for the workspace"
	@echo "  make test                  Rust workspace tests + doctests"
	@echo "  make verify                Canonical pre-merge gate (compile + batchalign + book)"
	@echo "  make ci-local              Fast local CI approximation"
	@echo "  make gate                  Verify source and record a pre-push receipt"
	@echo "  make ci-full               Stricter local CI approximation"
	@echo "  make smoke CRATE=x         Fast compile check + one crate test"
	@echo ""
	@echo "Batchalign workflow:"
	@echo "  make batchalign-check              Imported Batchalign compile checks"
	@echo "  make batchalign-test-rust          Imported Batchalign Rust library suites"
	@echo "  make batchalign-test-integration   Imported Batchalign focused integration gates"
	@echo "  make batchalign-build-pyo3         Imported standalone PyO3 crate"
	@echo "  make batchalign-build-wheel        Imported Batchalign wheel"
	@echo "  make batchalign-build-ci-wheel     Development-profile ABI3 wheel for push CI"
	@echo "  make batchalign-test-python        Imported Batchalign Python pytest gate"
	@echo "  make batchalign-lint-python        Imported Batchalign Python ruff gate"
	@echo "  make batchalign-ipc-schema-check   IPC schema vs Rust types drift gate"
	@echo "  make batchalign-typecheck-python   Imported Batchalign Python typecheck gate"
	@echo "  make batchalign-ci-python          Imported Batchalign Python wheel/test/type gate"
	@echo "  make batchalign-dashboard-build       Imported dashboard frontend build"
	@echo "  make batchalign-dashboard-api-check  Imported dashboard API drift gate"
	@echo "  make batchalign-dashboard-e2e        Dashboard e2e tests (mock server)"
	@echo "  make batchalign-dashboard-e2e-real   Dashboard e2e tests (real server)"
	@echo "  make batchalign-runtime-check        Imported runtime constants check"
	@echo "  make batchalign-ci-rust              Imported Batchalign Rust/PyO3 CI gate"
	@echo ""
	@echo "  make lint                  Structural lints (core purity, wide structs)"
	@echo ""
	@echo "Docs and developer helpers:"
	@echo "  make install-hooks          Install the git pre-push hook"
	@echo "  make book                   Build the unified TalkBank Toolchain book"
	@echo "  make book-serve             Serve the unified book locally"
	@echo "  cargo run -q -p xtask -- help"
	@echo "                              List xtask audit/helper commands"
	@echo "  make clean                  Clean build artifacts"
	@echo ""

# Warn if the pre-push hook isn't installed. Not a hard failure
# users may intentionally push without hooks in rare cases (e.g.,
# re-pushing an already-verified commit after a remote hiccup).
hooks-check:
	@if [ ! -e .git/hooks/pre-push ]; then \
	  echo "warning: .git/hooks/pre-push is not installed, run 'make install-hooks'" >&2; \
	fi

# Run all tests.
#
# `cargo test`, never `cargo nextest`: nextest is banned and uninstalled in this
# workspace (it execs every test binary up front merely to enumerate tests, which
# saturates macOS's notarization-assessment path). This target and `smoke` below
# still named it long after the removal, so `make test` could only fail with
# "no such command".
test:
	@echo "==> Testing Rust workspace..."
	cargo test --workspace
	@echo "==> Testing doctests..."
	cargo test --doc

test-affected:
	cargo run -q -p xtask -- affected-rust test

BATCHALIGN_PYTEST_ARGS ?= batchalign --disable-pytest-warnings -k "not test_whisper_fa_pipeline"

# The crate list is explicit rather than `--workspace` so the CI job stays
# scoped, which means a NEW workspace member is invisible here until it is added
# by hand. `batchalign-fa-core` sat outside both this and the test gate below,
# so its 7 library tests ran only when someone typed its name; added along with
# the new `batchalign-core`.
batchalign-check:
	@echo "==> Checking imported Batchalign Rust crates..."
	cargo check -p batchalign-types -p batchalign-core -p batchalign-fa-core -p batchalign -p batchalign-transform -p batchalign-pyo3 --all-targets

batchalign-test-rust:
	@echo "==> Testing imported batchalign-types..."
	cargo test -p batchalign-types --lib -q
	@echo "==> Testing batchalign-fa-core (forced-alignment core)..."
	cargo test -p batchalign-fa-core --lib -q
	@echo "==> Testing batchalign-transform (ML over chatter's generic transform)..."
	cargo test -p batchalign-transform --lib -q
	@echo "==> Testing imported batchalign..."
	cargo test -p batchalign --lib -q
	@echo "==> Testing xtask (the audits and lint gates themselves)..."
	cargo test -p xtask -q
	@$(MAKE) batchalign-test-doc

# Doctests, which the CI chain did not run. `make test` does (see `test:`), but
# `batchalign-ci-rust` did not, and that is the chain used before a deploy.
#
# A doctest is not a cargo target, so neither `--lib` nor `--all-targets`
# reaches one; a public example can therefore break with every gate green.
#
# `--workspace`, never a hand-written crate list: the list is the drift this
# file already warns about above, and a first attempt at this target spelled out
# four crates and silently omitted three workspace members.
batchalign-test-doc:
	@echo "==> Testing doctests (not cargo targets; --lib and --all-targets miss them)..."
	cargo test --doc --workspace -q

# The ML golden suite: real engines, real model weights, real network.
#
# Deliberately NOT part of `make test` or `make verify`. It needs multi-GB
# model downloads, live credentials for the hosted ASR engines, and minutes of
# wall clock, so it is opt-in and feature-gated: a plain `cargo test` cannot
# reach it (see `required-features = ["ml-golden"]` on the target).
#
# WHY THIS TARGET EXISTS. Until 2026-07-28 the suite had NO entry point at all
# in the Makefile or CI, and was reachable only through nextest's
# `--profile ml`, which was retired with nextest itself. It had therefore
# become unrunnable, and its tests additionally SKIPPED SILENTLY when a live
# session could not be acquired, so they could report `ok` without executing.
# Both are fixed: the skip now panics, and this is the entry point.
batchalign-test-ml-golden:
	@echo "==> ML golden suite (real engines; multi-GB models; needs credentials)"
	cargo test -p batchalign --features ml-golden --test ml_golden -- --test-threads=1

batchalign-test-integration:
	@echo "==> Running imported Batchalign CI hygiene..."
	cargo run -q -p xtask -- lint-ci-hygiene
	@echo "==> Testing imported batchalign focused integration gates..."
	cargo test -p batchalign --test contract_suite --locked -q

batchalign-build-pyo3:
	@echo "==> Building imported standalone PyO3 crate..."
	cargo build --manifest-path crates/batchalign-pyo3/Cargo.toml -q

batchalign-build-wheel:
	@echo "==> Building imported Batchalign wheel..."
	@# Default: always rebuild the native binary so the wheel never
	@# bundles a stale one. The 2026-04-29 deploy postmortem (cancel-
	@# cascade) was caused by a previous guard that silently reused
	@# whatever was at batchalign/_bin/batchalign3, even when the
	@# sources had changed.
	@#
	@# Two known-safe skip paths:
	@#   1. Windows cross-compile pre-stages batchalign3.exe (existing).
	@#   2. CI's build-wheel job downloads the cli-binary artifact (the
	@#      OUTPUT of build-cli, compiled FROM THIS COMMIT'S SOURCES) and
	@#      sets BATCHALIGN_PRESTAGED_BIN=1 to declare provenance. In
	@#      that one path the prestaged binary is guaranteed fresh and
	@#      rebuilding duplicates the optimized native compile.
	@#
	@# Local invocations (no env var, no .exe) fall through to the safe
	@# always-rebuild path.
	@if [ -f batchalign/_bin/batchalign3.exe ]; then \
	  echo "==> Using pre-staged Windows binary (.exe)"; \
	elif [ "$$BATCHALIGN_PRESTAGED_BIN" = "1" ] && [ -x batchalign/_bin/batchalign3 ]; then \
	  echo "==> Using pre-staged Linux binary from CI build-cli artifact"; \
	else \
	  echo "==> Building native batchalign3 binary..."; \
	  cargo build --release -p batchalign; \
	  mkdir -p batchalign/_bin; \
	  cp target/release/batchalign3 batchalign/_bin/batchalign3; \
	fi
	rm -rf dist
	mkdir -p dist
	@# `pyproject.toml` deliberately keeps Maturin's dev profile for the fast
	@# `maturin develop` loop. Deployment wheels must override it explicitly;
	@# PEP 517's `uv build` used that dev profile and shipped an unoptimized
	@# batchalign_core extension even though the bundled CLI above was release.
	@# `--only-dev` provisions Maturin from the frozen lock even in a fresh CI
	@# checkout; `--no-sync` is invalid there because no tool exists to spawn.
	uv run --frozen --only-dev maturin build --release --out dist/

# Push CI proves packaging and behavior with a development-profile extension.
# The tag release workflow separately builds and smoke-tests the optimized
# artifacts on all five platforms. Repeating release-mode native compilation
# on every push added about 15 minutes to feedback without checking a distinct
# source or Python compatibility property.
batchalign-build-ci-wheel:
	@echo "==> Building development-profile Batchalign ABI3 wheel for push CI..."
	@if [ "$$BATCHALIGN_PRESTAGED_BIN" != "1" ] || [ ! -x batchalign/_bin/batchalign3 ]; then \
		echo "ERROR: CI wheel requires this commit's pre-staged native binary" >&2; \
		exit 1; \
	fi
	rm -rf dist
	mkdir -p dist
	uv run --frozen --only-dev maturin build --out dist/

batchalign-python-prepare: batchalign-build-wheel
	@echo "==> Syncing imported Batchalign dev dependencies..."
	uv sync --group dev --no-install-project
	@echo "==> Installing imported Batchalign wheel into the dev environment..."
	uv pip install --reinstall --no-deps dist/*.whl
	@# THE INVARIANT: after this target, exactly one compiled extension is
	@# importable, and it is the one this target just built.
	@#
	@# `uv run maturin develop` writes batchalign_core/batchalign_core.abi3.so
	@# into the working tree, and because python-source is "." the repo-root
	@# batchalign_core package shadows site-packages for anything run from here.
	@# Leaving that file behind means this target installs a wheel that nothing
	@# imports, and the suite silently reports on whatever develop last built.
	@# That is not hypothetical: it happened on 2026-08-21 and made a session of
	@# PyO3 verification meaningless.
	@#
	@# Removing it is not extra damage. `uv pip install` above has already
	@# replaced the editable install that file belonged to, so it is orphaned
	@# either way. Verified the same day: with it gone, the shim's extend_path
	@# resolves to the wheel's extension in site-packages.
	@# Globbed rather than naming one file: the invariant is "no in-tree
	@# extension shadows the wheel", not "this filename does not". The tree also
	@# accumulates fossils under other ABI tags, e.g. a cpython-312 build left
	@# from before the move to 3.13, which a filename-specific rm would miss.
	@count=$$(ls batchalign_core/*.so 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$count" != "0" ]; then \
	  echo "==> Cleared $$count in-tree extension(s); the wheel is what runs"; \
	  rm -f batchalign_core/*.so; \
	else \
	  echo "==> No in-tree extension present; the wheel is what runs"; \
	fi

_batchalign-test-python:
	@# Proves the invariant rather than assuming it: resolves the extension
	@# Python will actually import and refuses if it is older than the Rust it
	@# is built from. Covers both install modes and the no-extension case, so
	@# it holds whether you came via batchalign-python-prepare or maturin
	@# develop. Fails closed.
	@echo "==> Checking the extension Python will import is current..."
	bash scripts/check_extension_freshness.sh
	@echo "==> Running imported Batchalign Python tests..."
	uv run --no-sync pytest $(BATCHALIGN_PYTEST_ARGS)

batchalign-test-python: batchalign-python-prepare
	@$(MAKE) _batchalign-test-python

batchalign-ipc-schema-check:
	@echo "==> Verifying IPC schema matches the Rust types..."
	bash scripts/check_ipc_type_drift.sh

# ruff reads only SOURCE: no wheel, no editable install, no maturin build. That
# is what lets the pre-push hook run it, so it is split out from the targets
# that depend on `batchalign-python-prepare` (which builds a release wheel and
# costs minutes). CI calls this same target, so there is one owner of the two
# commands rather than a workflow that repeats them.
batchalign-lint-python-source:
	@echo "==> Running imported Batchalign Python lint (ruff)..."
	uv run --no-sync ruff check .
	@echo "==> Checking imported Batchalign Python formatting (ruff)..."
	uv run --no-sync ruff format --check .

_batchalign-lint-python:
	@$(MAKE) batchalign-lint-python-source

batchalign-lint-python: batchalign-python-prepare
	@$(MAKE) _batchalign-lint-python

_batchalign-typecheck-python:
	@$(MAKE) batchalign-ipc-schema-check
	@echo "==> Verifying imported Batchalign retirement gates..."
	test ! -e batchalign/cli/cli.py
	test ! -e batchalign/serve/app.py
	test ! -e batchalign/serve/job_store.py
	@echo "==> Running imported Batchalign CI hygiene..."
	cargo run -q -p xtask -- lint-ci-hygiene
	@$(MAKE) batchalign-runtime-check
	@echo "==> Running imported Batchalign Python typecheck..."
	uv run --no-sync mypy

batchalign-typecheck-python: batchalign-python-prepare
	@$(MAKE) _batchalign-typecheck-python

batchalign-ci-python: batchalign-python-prepare
	@$(MAKE) _batchalign-test-python
	@$(MAKE) _batchalign-lint-python
	@$(MAKE) _batchalign-typecheck-python

batchalign-runtime-check:
	@echo "==> Verifying imported runtime constants..."
	uv run --no-sync python scripts/check_runtime_drift.py

batchalign-dashboard-api-check:
	@echo "==> Verifying imported dashboard API artifacts..."
	bash scripts/check_dashboard_api_drift.sh

batchalign-dashboard-schema-check:
	@echo "==> Verifying generated openapi.json (no network)..."
	bash scripts/check_dashboard_schema_drift.sh

batchalign-dashboard-build:
	@echo "==> Building imported dashboard frontend..."
	cd frontend && npm ci && npm run build

batchalign-dashboard-e2e:
	@echo "==> Running dashboard e2e tests (mock server)..."
	bash scripts/run_react_dashboard_smoke.sh

batchalign-dashboard-e2e-real:
	@echo "==> Running dashboard e2e tests (real server)..."
	cd frontend && npm ci
	cd frontend/e2e && npm ci && npm run install:browsers
	BATCHALIGN_REAL_SERVER_E2E=1 bash scripts/run_react_dashboard_smoke.sh

# The structural lints: the checks that state properties no compiler can.
#
# One target rather than a line per lint per gate. Before 2026-07-30 each lint
# was named in every target that wanted it, and the cost showed up immediately:
# `ci-full` invoked the purity gate twice (once directly, once through
# batchalign-ci-rust), and `lint-wide-structs` reached only `ci-local`, which
# nobody runs, so its table drifted for two months. Adding a sixth lint is now
# one line here.
# The three CI hygiene jobs, each a NAMED TARGET so `ci.yml` can invoke it by
# name. That is the whole point: `scripts/check_push_gate_sync.py` exists to
# fail when the pre-push hook does not run everything CI runs, but it can only
# see CI steps written as `make <target>`. These three jobs used to be raw
# commands, so the intersection it computes was EMPTY and it compared `ci.yml`
# against nothing and printed success -- which is exactly how an unformatted
# commit reached CI on 2026-08-25 in a repo that already owns an anti-drift
# checker. Naming them puts them under that check mechanically, instead of
# under a hand-written mirror somebody has to remember to update.
fmt-check:
	cargo fmt --all -- --check

lint-shell:
	bash scripts/lint/shellcheck-all.sh

lint-actionlint:
	actionlint

lint:
	@echo "==> rustfmt"
	@# Safe to run from here even though CI invokes `lint` via
	@# `batchalign-ci-rust` on a cargo-only runner: rustfmt is a cargo
	@# component. shellcheck and actionlint are NOT, which is why they are
	@# separate targets the hook runs directly rather than prerequisites here.
	@$(MAKE) fmt-check
	@echo "==> clippy (CI-gated crates)"
	@# CI runs `lint` via `batchalign-ci-rust`, and until now nothing in that
	@# chain ran clippy, so every `#![deny(clippy::...)]` in the tree fired
	@# only on a developer's machine.
	@#
	@# Scoped to the same crates `batchalign-check` names, NOT `--workspace`:
	@# the workspace includes `apps/dashboard-desktop`, whose Tauri stack needs
	@# GTK/glib system libraries the CI runner does not have. A `--workspace`
	@# clippy therefore passed locally and failed CI with a `glib-sys` build
	@# error, which is why every other CI-gated target here enumerates crates.
	cargo clippy -p batchalign-types -p batchalign-core -p batchalign-fa-core \
		-p batchalign -p batchalign-transform -p batchalign-pyo3 \
		--all-targets -- -D warnings
	@echo "==> push gate covers CI"
	@# Static check that scripts/gate.sh runs what the workflow runs. The
	@# hook drifted into a weaker subset and reported success for three pushes
	@# CI rejected on 2026-08-14.
	@python3 scripts/check_push_gate_sync.py
	@echo "==> batchalign-core purity gate"
	@cargo run -q -p xtask -- lint-core-purity
	@echo "==> wide struct audit"
	@cargo run -q -p xtask -- lint-wide-structs
	@echo "==> prose references"
	@cargo run -q -p xtask -- audit-prose-references

# Lints run FIRST: they are the cheapest gates here.
batchalign-ci-rust:
	@# THIS TARGET IS INVOKED BY CI ITSELF (the `Batchalign Rust` workflow), so
	@# it may only use tools that runner installs: cargo and its components.
	@# Shellcheck and actionlint are separate CI JOBS with their own setup, and
	@# calling them here made the Rust workflow die with `Error 127` on a runner
	@# that has neither. The developer-facing target that mirrors ALL of CI is
	@# `make ci-local`; use that before pushing.
	@$(MAKE) lint
	@$(MAKE) batchalign-check
	@$(MAKE) batchalign-test-rust
	@$(MAKE) batchalign-test-integration
	@$(MAKE) batchalign-build-pyo3

# Build all components
build:
	@echo "==> Building Rust workspace..."
	cargo build --workspace --release

# Fast compile check
check:
	@echo "==> Checking Rust workspace..."
	cargo check --workspace --all-targets

check-affected:
	cargo run -q -p xtask -- affected-rust check

lint-affected:
	cargo run -q -p xtask -- affected-rust clippy

# Canonical pre-merge verification gates
# The CHAT-format gates (parser signature guardrail, spec tools, CHAT manual
# anchors, parser-equivalence / golden / corpus / generated-check / fuzz) moved
# to chatter, which is now the single home for the CHAT core (model, parser,
# transform, spec, grammar). talkbank-tools verifies the batchalign layer it
# still owns; CHAT-format verification lives in the chatter repo.
verify:
	@$(MAKE) hooks-check
	@echo "==> [G1] Rust workspace compile check"
	cargo check --workspace --all-targets
	@echo "==> [G2] Batchalign Rust check (types, transform, batchalign, pyo3)"
	@$(MAKE) batchalign-check
	@echo "==> [G3] Batchalign Rust lib tests"
	@$(MAKE) batchalign-test-rust
	@echo "==> [G4] mdBook build + linkcheck"
	@$(MAKE) book-check

# Build the unified TalkBank mdBook and link-check it with lychee.
#
# mermaid must be a preprocessor (it rewrites ```mermaid blocks). The pair
# (mdBook + mdbook-mermaid) is pinned to one version each, the SAME versions
# chatter's book pins, and both are inventoried in the private workspace's
# book-toolchain pin list so a bump lands in every repo at once. Until
# 2026-09-01 this repo stayed on the 0.4.x line because no mdbook-mermaid
# release spoke mdBook 0.5's wire format; 0.17 does. Link-checking is
# decoupled onto lychee, which runs on the built HTML and is independent of
# mdBook's wire format (the old mdbook-linkcheck2 renderer was tied to it).
# lychee still catches SUMMARY-unreachable targets like the 2026-05-22
# batchalign/introduction.md regression. `--offline` skips web links;
# `--root-dir` resolves the 404 page's leading '/'.
#
# The git-dates preprocessor (book.toml) stamps every page with git-derived
# "last changed" dates; its tests run first, and `verify` then proves the
# rendered front page carries the same dates git reports, so a build in which
# the preprocessor silently did not run cannot pass. Needs full git history
# (a shallow checkout is refused by the script itself, not by this target).
book-check:
	@command -v mdbook >/dev/null || { \
		echo "ERROR: mdbook not found on PATH."; \
		echo "Install: cargo install mdbook@0.5.4 mdbook-mermaid@0.17.1 lychee"; \
		exit 1; \
	}
	@command -v lychee >/dev/null || { \
		echo "ERROR: lychee not found on PATH."; \
		echo "Install: cargo install lychee"; \
		exit 1; \
	}
	python3 -m unittest $(CURDIR)/scripts/test_mdbook_git_dates.py
	cd $(CURDIR)/book && mdbook build
	python3 $(CURDIR)/scripts/mdbook_git_dates.py verify --book-root $(CURDIR)/book --page introduction.md $(CURDIR)/book/build/index.html
	lychee --offline --root-dir "$(CURDIR)/book/build" "$(CURDIR)/book/build"
	@# The fence-shape regression guard, mirroring book.yml's third step.
	@# rustdoc compiles every UNTAGGED ``` block as a Rust doctest, so a
	@# bare fence around sample output fails the build. This target ran
	@# only build + lychee until 2026-07-31, so it passed locally while
	@# CI was red on three such blocks; a local gate that omits a CI step
	@# is worse than no local gate, because it is believed.
	cd $(CURDIR)/book && mdbook test

# Fast iteration: compile-check workspace + test a single crate
# Usage: make smoke CRATE=talkbank-model
smoke:
	@echo "==> Compile check (workspace)..."
	cargo check --workspace --all-targets
	@echo "==> Testing $(CRATE)..."
	cargo test -p $(CRATE) --no-fail-fast

# Fast local CI: fmt + dependency-aware compile checks + structural lints.
ci-local:
	@# Calls the SAME targets `ci.yml` invokes, rather than restating their
	@# commands. An earlier version of this recipe spelled them out and called
	@# itself "the pre-push target", which was false twice over: the hook runs
	@# `batchalign-ci-rust`, and a hand-written mirror is the drift the named
	@# targets above exist to eliminate.
	@$(MAKE) lint-shell
	@$(MAKE) lint-actionlint
	@echo "==> affected compile check"
	cargo run -q -p xtask -- affected-rust check
	@$(MAKE) lint
	@echo "✓ ci-local passed"

# Full local CI: mirrors the stricter CI-style gate.
ci-full:
	@# Build the assets that the default `embed-dashboard` feature compiles into
	@# the Rust binary. The GitHub Rust job does this before batchalign-ci-rust;
	@# omitting it locally made two embedded-dashboard tests fail only after a
	@# clean worktree removed frontend/dist.
	@$(MAKE) batchalign-dashboard-build
	@# `batchalign-ci-rust` below owns clippy and check for the imported BA3
	@# crates. Run clippy only for the two remaining workspace members here;
	@# the former workspace-wide clippy + check compiled the BA3 graph twice
	@# before running the same gate again.
	@echo "==> clippy (remaining workspace members)"
	cargo clippy -p batchalign-dashboard-desktop -p batchalign-whisper-pilot \
		--all-targets -- -D warnings
	@echo "==> runtime_constants.toml drift check"
	@cargo run -p xtask --quiet -- gen-runtime-toml --check
	@echo "==> imported Batchalign Rust/PyO3 gate"
	@$(MAKE) batchalign-ci-rust
	@echo "✓ ci-full passed"

# Verify the complete local pre-push gate once; the hook only checks its receipt.
gate:
	bash scripts/gate.sh

gate-receipts-test:
	python3 scripts/test_push_gate_sync.py
	bash scripts/test-gate-receipts.sh

# Install git hooks (pre-push).
install-hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "✓ pre-push hook installed"

# Clean build artifacts
clean:
	cargo clean

# Build the documentation book
#
# Run with the book directory as cwd (never `mdbook build book/` from here):
# mdBook 0.4.x sets no cwd of its own for preprocessor subprocesses and just
# inherits whatever directory `mdbook` itself was started from, so the
# book.toml preprocessor commands' `../scripts/...` paths only resolve when
# that directory is the book directory. mdBook 0.5.x always uses the book
# directory regardless, so this is a no-op for it.
book:
	cd book && mdbook build

# Serve the documentation book locally
book-serve:
	cd book && mdbook serve

# ---------------------------------------------------------------------------
# Doc audit (talkbank-tools only)
# ---------------------------------------------------------------------------
#
# The catalog DB is auditing tooling, not user content; it lives in
# the meta-repo's release-doc-audit/ working dir. Default path assumes
# the workspace layout `<workspace>/talkbank-tools` alongside
# `<workspace>/docs/release-doc-audit/audit.db`. Operators with a
# different layout override TB_AUDIT_DB.
#
# Daily-cadence: `make audit-status` is the session-start command
# prints Bucket A progress, streak, and the next 5 unvetted sections.
# See `<workspace>/docs/release-doc-audit/audit-method.md`.
TB_AUDIT_DB ?= ../docs/release-doc-audit/audit.db
TB_AUDIT_TT_ROOT ?= $(CURDIR)

audit-status:
	@TB_AUDIT_DB="$(TB_AUDIT_DB)" cargo run -q -p xtask -- audit-docs status

audit-streak:
	@TB_AUDIT_DB="$(TB_AUDIT_DB)" cargo run -q -p xtask -- audit-docs streak

audit-scan:
	TB_AUDIT_DB="$(TB_AUDIT_DB)" TB_AUDIT_TT_ROOT="$(TB_AUDIT_TT_ROOT)" \
		cargo run -q -p xtask -- audit-docs scan

audit-flag-staleness:
	TB_AUDIT_DB="$(TB_AUDIT_DB)" TB_AUDIT_TT_ROOT="$(TB_AUDIT_TT_ROOT)" \
		cargo run -q -p xtask -- audit-docs flag-staleness

# Layer 1 CI gate. Catalog-independent, walks every .md file under the
# repo root and exits non-zero if any high-severity prose-reference
# pattern (deleted crate, moved book path) is found outside the
# allow-list. Designed for ci.yml use where audit.db is not present.
audit-prose-references:
	cargo run -q -p xtask -- audit-prose-references
