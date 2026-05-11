.PHONY: help symbols-gen generated-check test-gen mine-candidates test test-affected test-grammar test-generated test-fragment-semantics test-legacy-fragment-parity test-parity batchalign-check batchalign-test-rust batchalign-test-integration batchalign-build-pyo3 batchalign-build-wheel batchalign-python-prepare batchalign-test-python batchalign-typecheck-python batchalign-ci-python batchalign-runtime-check batchalign-dashboard-api-check batchalign-dashboard-build batchalign-dashboard-e2e batchalign-dashboard-e2e-real batchalign-ci-rust build clean check check-affected verify parser-guard chat-anchors-check fuzz-check hooks-check book book-serve coverage smoke check-specs ci-local ci-full install-hooks lint-affected _batchalign-test-python _batchalign-typecheck-python audit-status audit-streak audit-scan audit-flag-staleness audit-prose-references

help:
	@echo "talkbank-tools task index"
	@echo ""
	@echo "Core chat/parser/model workflow:"
	@echo "  make check                 Fast compile check for core workspaces"
	@echo "  make test                  Core Rust tests + doctests + spec tools"
	@echo "  make verify                Canonical core pre-merge gate"
	@echo "  make ci-local              Fast local CI approximation"
	@echo "  make ci-full               Stricter local CI approximation"
	@echo "  make smoke CRATE=x         Fast compile check + one crate test"
	@echo ""
	@echo "Batchalign workflow:"
	@echo "  make batchalign-check              Imported Batchalign compile checks"
	@echo "  make batchalign-test-rust          Imported Batchalign Rust library suites"
	@echo "  make batchalign-test-integration   Imported Batchalign focused integration gates"
	@echo "  make batchalign-build-pyo3         Imported standalone PyO3 crate"
	@echo "  make batchalign-build-wheel        Imported Batchalign wheel"
	@echo "  make batchalign-test-python        Imported Batchalign Python pytest gate"
	@echo "  make batchalign-typecheck-python   Imported Batchalign Python typecheck gate"
	@echo "  make batchalign-ci-python          Imported Batchalign Python wheel/test/type gate"
	@echo "  make batchalign-dashboard-build       Imported dashboard frontend build"
	@echo "  make batchalign-dashboard-api-check  Imported dashboard API drift gate"
	@echo "  make batchalign-dashboard-e2e        Dashboard e2e tests (mock server)"
	@echo "  make batchalign-dashboard-e2e-real   Dashboard e2e tests (real server)"
	@echo "  make batchalign-runtime-check        Imported runtime constants check"
	@echo "  make batchalign-ci-rust              Imported Batchalign Rust/PyO3 CI gate"
	@echo ""
	@echo "Spec, grammar, and generated artifacts:"
	@echo "  make symbols-gen            Generate shared symbol sets"
	@echo "  make test-gen               Regenerate spec-driven tests/docs"
	@echo "  make generated-check        Verify generated artifacts are in sync"
	@echo "  make test-grammar           Tree-sitter grammar corpus tests"
	@echo "  make test-generated         Spec-generated parser/validation tests"
	@echo "  make test-legacy-fragment-parity Legacy word-fragment parity audit"
	@echo "  make test-parity            Full-file parser equivalence"
	@echo "  make coverage               Reference corpus node-type coverage"
	@echo "  make check-specs            Verify every error code has a spec file"
	@echo "  make mine-candidates        Mine valid CHAT candidates from ../data"
	@echo ""
	@echo "Docs and developer helpers:"
	@echo "  make chat-anchors-check     Verify CHAT manual anchors referenced in source/docs"
	@echo "  make parser-guard           Enforce ErrorSink/Option parser guardrail"
	@echo "  make install-hooks          Install the git pre-push hook"
	@echo "  make book                   Build the unified TalkBank Toolchain book"
	@echo "  make book-serve             Serve the unified book locally"
	@echo "  cargo run -q -p xtask -- help"
	@echo "                              List xtask audit/helper commands"
	@echo "  make clean                  Clean build artifacts"
	@echo ""

# Guardrail: disallow introducing new ErrorSink + Option parser signatures.
parser-guard:
	@scripts/check-errorsink-option-signatures.sh

# Validate that CHAT manual anchors referenced in source/docs resolve in CHAT.html.
# Optional local mirror path: CHAT_HTML_PATH=/abs/path/to/CHAT.html make chat-anchors-check
chat-anchors-check:
	@scripts/check-chat-manual-anchors.sh

# Cheap gate that mirrors the Fuzz Smoke Test CI job's first action
# (cargo metadata on fuzz/). Catches workspace-isolation breakage
# without compiling anything. No nightly toolchain needed.
fuzz-check:
	@echo "==> fuzz workspace isolation check"
	@cd fuzz && cargo metadata --no-deps --format-version 1 >/dev/null

# Warn if the pre-push hook isn't installed. Not a hard failure —
# users may intentionally push without hooks in rare cases (e.g.,
# re-pushing an already-verified commit after a remote hiccup).
hooks-check:
	@if [ ! -e .git/hooks/pre-push ]; then \
	  echo "warning: .git/hooks/pre-push is not installed — run 'make install-hooks'" >&2; \
	fi

# Generate shared symbol sets used by grammar.
symbols-gen:
	@echo "==> Generating shared symbol sets..."
	node spec/symbols/validate_symbol_registry.js
	node scripts/generate-symbol-sets.js
	node spec/symbols/generate_rust_symbol_sets.js
	rustfmt crates/talkbank-model/src/generated/symbol_sets.rs spec/tools/src/generated/symbol_sets.rs

# Generate tests from specs
test-gen:
	@$(MAKE) symbols-gen
	@echo "==> Generating tree-sitter tests from spec..."
	cd spec/tools && cargo run --bin gen_tree_sitter_tests -- \
		--spec-dir ../constructs \
		--error-dir ../errors \
		--template-dir templates \
		--output-dir ../../grammar/test/corpus
	@echo "==> Generating Rust tests from spec..."
	cd spec/tools && cargo run --bin gen_rust_tests -- \
		--construct-dir ../constructs \
		--error-dir ../errors \
		--output-dir ../../crates/talkbank-parser-tests/tests/generated \
		--test-error-path talkbank_parser_tests::test_error::TestError
	@echo "==> Generating error documentation..."
	cd spec/tools && cargo run --bin gen_error_docs -- \
		--error-dir ../errors \
		--output-dir ../../docs/errors

# Mine candidate files from real data (staging-only, for curation input)
# Usage: make mine-candidates LANGUAGES=eng,spa TOP=50 MAX_LINES=200 MAX_FILES=20000 DATA_DIR=../data
mine-candidates:
	@mkdir -p spec/tmp/mined
	cargo run --manifest-path spec/runtime-tools/Cargo.toml --bin extract_corpus_candidates -- \
		--data-dir $${DATA_DIR:-../data} \
		--languages $${LANGUAGES:-eng} \
		--node-types grammar/src/node-types.json \
		--max-lines $${MAX_LINES:-200} \
		--max-files $${MAX_FILES:-20000} \
		--top $${TOP:-50} \
		--require-rust-parse=true \
		--require-rust-validation=true \
		--validate-alignment=true \
		--json \
		--output spec/tmp/mined/candidates.$${LANGUAGES:-eng}.json

# Regenerate and verify all generated artifacts are committed.
generated-check:
	@$(MAKE) symbols-gen
	@$(MAKE) test-gen
	@echo "==> Checking for uncommitted generated changes..."
	git diff --exit-code -- \
		spec/symbols/symbol_registry.json \
		spec/symbols/validate_symbol_registry.js \
		spec/symbols/generate_rust_symbol_sets.js \
		crates/talkbank-model/src/generated/symbol_sets.rs \
		spec/tools/src/generated/symbol_sets.rs \
		crates/talkbank-parser-tests/tests/generated \
		docs/errors
	@echo "==> [Phase β] runtime_constants.toml drift check"
	@cargo run -p xtask --quiet -- gen-runtime-toml --check

# Run all tests
test:
	@echo "==> Testing Rust workspace..."
	cargo nextest run --workspace
	@echo "==> Testing doctests..."
	cargo test --doc
	@echo "==> Testing spec tools..."
	cd spec/tools && cargo test
	@echo "==> Testing spec runtime tools..."
	cargo test --manifest-path spec/runtime-tools/Cargo.toml

test-grammar:
	cd grammar && tree-sitter test

test-generated:
	cargo nextest run -p talkbank-parser-tests --test generated
	cargo nextest run -p talkbank-parser-tests --test generated_tests

test-fragment-semantics:
	cargo nextest run -p talkbank-parser-tests --test golden_words_validation --test golden_tiers_validation

test-legacy-fragment-parity:
	cargo nextest run -p talkbank-parser-tests --test parser_equivalence_words

test-parity:
	cargo nextest run -p talkbank-parser-tests --test parser_equivalence_files

test-affected:
	cargo run -q -p xtask -- affected-rust test

BATCHALIGN_PYTEST_ARGS ?= batchalign --disable-pytest-warnings -k "not test_whisper_fa_pipeline"

batchalign-check:
	@echo "==> Checking imported Batchalign Rust crates..."
	cargo check -p batchalign-types -p batchalign --all-targets

batchalign-test-rust:
	@echo "==> Testing imported batchalign-types..."
	cargo test -p batchalign-types --lib -q
	@echo "==> Testing imported batchalign..."
	cargo test -p batchalign --lib -q

batchalign-test-integration:
	@echo "==> Running imported Batchalign CI hygiene..."
	cargo run -q -p xtask -- lint-ci-hygiene
	@echo "==> Testing imported batchalign focused integration gates..."
	cargo test -p batchalign --test json_compat --test workflow_helpers -q

batchalign-build-pyo3:
	@echo "==> Building imported standalone PyO3 crate..."
	cargo build --manifest-path crates/batchalign-pyo3/Cargo.toml -q

batchalign-build-wheel:
	@echo "==> Building imported Batchalign wheel..."
	@# Default: always rebuild the native binary so the wheel never
	@# bundles a stale one. The 2026-04-29 deploy postmortem (cancel-
	@# cascade) was caused by a previous guard that silently reused
	@# whatever was at batchalign/_bin/batchalign3 — even when the
	@# sources had changed.
	@#
	@# Two known-safe skip paths:
	@#   1. Windows cross-compile pre-stages batchalign3.exe (existing).
	@#   2. CI's build-wheel job downloads the cli-binary artifact (the
	@#      OUTPUT of build-cli, compiled FROM THIS COMMIT'S SOURCES) and
	@#      sets BATCHALIGN_PRESTAGED_BIN=1 to declare provenance. In
	@#      that one path the prestaged binary is guaranteed fresh and
	@#      rebuilding duplicates ~9 min of fat-LTO compile.
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
	uv build --wheel --out-dir dist/

batchalign-python-prepare: batchalign-build-wheel
	@echo "==> Syncing imported Batchalign dev dependencies..."
	uv sync --group dev --no-install-project
	@echo "==> Installing imported Batchalign wheel into the dev environment..."
	uv pip install --reinstall --no-deps dist/*.whl

_batchalign-test-python:
	@echo "==> Running imported Batchalign Python tests..."
	uv run --no-sync pytest $(BATCHALIGN_PYTEST_ARGS)

batchalign-test-python: batchalign-python-prepare
	@$(MAKE) _batchalign-test-python

_batchalign-typecheck-python:
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
	@$(MAKE) _batchalign-typecheck-python

batchalign-runtime-check:
	@echo "==> Verifying imported runtime constants..."
	python3 scripts/check_runtime_drift.py

batchalign-dashboard-api-check:
	@echo "==> Verifying imported dashboard API artifacts..."
	bash scripts/check_dashboard_api_drift.sh

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

batchalign-ci-rust:
	@$(MAKE) batchalign-check
	@$(MAKE) batchalign-test-rust
	@$(MAKE) batchalign-test-integration
	@$(MAKE) batchalign-build-pyo3

# Build all components
build:
	@$(MAKE) symbols-gen
	@echo "==> Building Rust workspace..."
	cargo build --workspace --release
	@echo "==> Building spec tools..."
	cd spec/tools && cargo build --release
	@echo "==> Building spec runtime tools..."
	cargo build --manifest-path spec/runtime-tools/Cargo.toml --release

# Fast compile check
check:
	@echo "==> Checking Rust workspace..."
	cargo check --workspace --all-targets
	@echo "==> Checking spec tools..."
	cd spec/tools && cargo check --all-targets
	@echo "==> Checking spec runtime tools..."
	cargo check --manifest-path spec/runtime-tools/Cargo.toml --all-targets

check-affected:
	cargo run -q -p xtask -- affected-rust check

lint-affected:
	cargo run -q -p xtask -- affected-rust clippy

# Canonical pre-merge verification gates
verify:
	@$(MAKE) hooks-check
	@echo "==> [G0] Parser signature guardrail"
	@$(MAKE) parser-guard
	@echo "==> [G1] Rust workspace compile check"
	cargo check --workspace --all-targets
	@echo "==> [G2] Spec tools compile check"
	cd spec/tools && cargo check --all-targets
	@echo "==> [G3] Spec runtime tools compile check"
	cargo check --manifest-path spec/runtime-tools/Cargo.toml --all-targets
	@echo "==> [G4] CHAT manual anchor links"
	@$(MAKE) chat-anchors-check
	@echo "==> [G5] Generated parser corpus equivalence suite"
	cargo nextest run -p talkbank-parser-tests --test generated
	@echo "==> [G6] Golden fragment validity (words + tiers)"
	@$(MAKE) test-fragment-semantics
	@echo "==> [G7] Bare-timestamp regression gate"
	cargo nextest run --test bare_timestamp_regression
	@echo "==> [G8] Reference corpus semantic equivalence"
	cargo nextest run -p talkbank-parser-tests --test parser_equivalence_files
	@echo "==> [G9] %wor tier parsing and alignment"
	cargo nextest run -p talkbank-parser-tests --test wor_terminator_alignment
	@echo "==> [G10] Golden tier roundtrip (%mor, %gra, %pho, %wor)"
	cargo nextest run -p talkbank-parser-tests --test parser_suite
	@echo "==> [G11] Reference corpus node coverage"
	@$(MAKE) coverage
	@echo "==> [G12] Generated artifacts match committed sources"
	@$(MAKE) generated-check
	@echo "==> [G13] Fuzz workspace isolation"
	@$(MAKE) fuzz-check
	@echo "==> [G14] Imported Batchalign Rust/PyO3 gate"
	@$(MAKE) batchalign-ci-rust

# Reference corpus grammar node type coverage
coverage:
	cd spec/tools && cargo run --bin corpus_node_coverage -- \
		--corpus-dir ../../corpus/reference \
		--node-types ../../grammar/src/node-types.json

# Fast iteration: compile-check workspace + test a single crate
# Usage: make smoke CRATE=talkbank-model
smoke:
	@echo "==> Compile check (workspace)..."
	cargo check --workspace --all-targets
	@echo "==> Testing $(CRATE)..."
	cargo nextest run -p $(CRATE) --no-fail-fast

# Verify every error code in the Rust enum has a corresponding spec file
check-specs:
	@scripts/check-error-specs.sh

# Fast local CI: fmt + dependency-aware compile checks + structural lints.
ci-local:
	@echo "==> fmt check (main workspace)"
	cargo fmt --all -- --check
	@echo "==> fmt check (spec/tools)"
	cd spec/tools && cargo fmt --all -- --check
	@echo "==> fmt check (spec/runtime-tools)"
	cargo fmt --manifest-path spec/runtime-tools/Cargo.toml --all -- --check
	@echo "==> affected compile check"
	cargo run -q -p xtask -- affected-rust check
	@echo "==> parser guardrail"
	@scripts/check-errorsink-option-signatures.sh
	@echo "==> wide struct audit"
	cargo run -q -p xtask -- lint-wide-structs
	@echo "==> docs sync"
	cargo run -q -p xtask -- lint-docs-sync
	@echo "✓ ci-local passed"

# Full local CI: mirrors the stricter CI-style gate.
ci-full:
	@echo "==> fmt check (main workspace)"
	cargo fmt --all -- --check
	@echo "==> fmt check (spec/tools)"
	cd spec/tools && cargo fmt --all -- --check
	@echo "==> fmt check (spec/runtime-tools)"
	cargo fmt --manifest-path spec/runtime-tools/Cargo.toml --all -- --check
	@echo "==> clippy"
	cargo clippy --all-targets -- -D warnings
	@echo "==> compile check (main workspace)"
	cargo check --workspace --all-targets
	@echo "==> compile check (spec/tools)"
	cd spec/tools && cargo check --all-targets
	@echo "==> compile check (spec/runtime-tools)"
	cargo check --manifest-path spec/runtime-tools/Cargo.toml --all-targets
	@echo "==> parser guardrail"
	@scripts/check-errorsink-option-signatures.sh
	@echo "==> generated artifacts check"
	@$(MAKE) generated-check
	@echo "==> imported Batchalign Rust/PyO3 gate"
	@$(MAKE) batchalign-ci-rust
	@echo "✓ ci-full passed"

# Install git hooks (pre-push).
install-hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "✓ pre-push hook installed"

# Clean build artifacts
clean:
	cargo clean
	cd spec/tools && cargo clean

# Build the documentation book
book:
	mdbook build book/

# Serve the documentation book locally
book-serve:
	mdbook serve book/

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
# Daily-cadence: `make audit-status` is the session-start command —
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

# Layer 1 CI gate. Catalog-independent — walks every .md file under the
# repo root and exits non-zero if any high-severity prose-reference
# pattern (deleted crate, moved book path) is found outside the
# allow-list. Designed for ci.yml use where audit.db is not present.
audit-prose-references:
	cargo run -q -p xtask -- audit-prose-references
