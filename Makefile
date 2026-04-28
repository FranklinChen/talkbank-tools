.PHONY: help symbols-gen generated-check test-gen mine-candidates test test-affected test-grammar test-generated test-fragment-semantics test-legacy-fragment-parity test-parity batchalign-check batchalign-test-rust batchalign-test-integration batchalign-build-pyo3 batchalign-build-wheel batchalign-runtime-check batchalign-dashboard-api-check batchalign-dashboard-build batchalign-book-build batchalign-ci-rust build clean check check-affected verify parser-guard chat-anchors-check fuzz-check hooks-check book book-serve coverage smoke check-specs ci-local ci-full install-hooks lint-affected

help:
	@echo "TalkBank Core Library Tasks"
	@echo ""
	@echo "  make test-gen       Generate tests from specs"
	@echo "  make symbols-gen    Generate shared symbol sets"
	@echo "  make generated-check Regenerate and verify generated artifacts are in sync"
	@echo "  make mine-candidates Mine valid CHAT candidates from ../data into spec/tmp/"
	@echo "  make test           Run all tests"
	@echo "  make test-affected  Run dependency-aware tests for changed code"
	@echo "  make test-grammar   Run tree-sitter grammar corpus tests"
	@echo "  make test-generated Run spec-generated parser/validation tests"
	@echo "  make test-legacy-fragment-parity Run legacy word-fragment parity audit"
	@echo "  make test-parity    Run full-file parser parity tests"
	@echo "  make batchalign-check Fast compile check for imported Batchalign Rust crates"
	@echo "  make batchalign-test-rust Run imported Batchalign Rust library suites"
	@echo "  make batchalign-test-integration Run imported Batchalign focused integration gates"
	@echo "  make batchalign-build-pyo3 Build the imported standalone PyO3 crate"
	@echo "  make batchalign-build-wheel Build the imported Batchalign wheel"
	@echo "  make batchalign-runtime-check Verify imported runtime constants"
	@echo "  make batchalign-dashboard-api-check Verify dashboard API artifacts"
	@echo "  make batchalign-dashboard-build Build the imported dashboard frontend"
	@echo "  make batchalign-book-build Build the imported Batchalign book"
	@echo "  make batchalign-ci-rust Run the imported Batchalign Rust/PyO3 CI command set"
	@echo "  make build          Build all components"
	@echo "  make check          Fast compile check"
	@echo "  make check-affected Fast dependency-aware compile check"
	@echo "  make verify         Canonical pre-merge verification gates"
	@echo "  make coverage       Report grammar node type coverage for reference corpus"
	@echo "  make parser-guard   Enforce parser ErrorSink/Option signature guardrail"
	@echo "  make chat-anchors-check Verify CHAT manual anchors referenced in source/docs"
	@echo "  make smoke CRATE=x  Fast check + test a single crate"
	@echo "  make check-specs    Verify every error code has a spec file"
	@echo "  make clean          Clean all build artifacts"
	@echo "  make book           Build the documentation book"
	@echo "  make book-serve     Serve the documentation book locally"
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

test-legacy-fragment-parity:
	cargo nextest run -p talkbank-parser-tests --test parser_equivalence_words

test-parity:
	cargo nextest run -p talkbank-parser-tests --test parser_equivalence_files

test-affected:
	cargo run -q -p xtask -- affected-rust test

batchalign-check:
	@echo "==> Checking imported Batchalign Rust crates..."
	cargo check -p batchalign-types -p batchalign-revai -p batchalign-chat-ops -p batchalign-app -p batchalign-cli --all-targets

batchalign-test-rust:
	@echo "==> Testing imported batchalign-types..."
	cargo test -p batchalign-types --lib -q
	@echo "==> Testing imported batchalign-revai..."
	cargo test -p batchalign-revai --lib -q
	@echo "==> Testing imported batchalign-chat-ops..."
	cargo test -p batchalign-chat-ops --lib -q
	@echo "==> Testing imported batchalign-app..."
	cargo test -p batchalign-app --lib -q
	@echo "==> Testing imported batchalign-cli..."
	cargo test -p batchalign-cli --lib -q

batchalign-test-integration:
	@echo "==> Running imported Batchalign CI hygiene..."
	cargo run -q -p xtask -- lint-ci-hygiene
	@echo "==> Testing imported batchalign-cli CI proxy..."
	cargo test -p batchalign-cli --test ci_checks -q
	@echo "==> Testing imported batchalign-app focused integration gates..."
	cargo test -p batchalign-app --test json_compat --test workflow_helpers -q

batchalign-build-pyo3:
	@echo "==> Building imported standalone PyO3 crate..."
	cargo build --manifest-path pyo3/Cargo.toml -q

batchalign-build-wheel:
	@echo "==> Building imported Batchalign wheel..."
	@if [ ! -x batchalign/_bin/batchalign3 ] && [ ! -f batchalign/_bin/batchalign3.exe ]; then \
	  echo "==> Staging bundled batchalign3 binary..."; \
	  cargo build --release -p batchalign-cli; \
	  mkdir -p batchalign/_bin; \
	  cp target/release/batchalign3 batchalign/_bin/batchalign3; \
	fi
	uv build --wheel --out-dir dist/

batchalign-runtime-check:
	@echo "==> Verifying imported runtime constants..."
	python3 scripts/check_runtime_drift.py

batchalign-dashboard-api-check:
	@echo "==> Verifying imported dashboard API artifacts..."
	bash scripts/check_dashboard_api_drift.sh

batchalign-dashboard-build:
	@echo "==> Building imported dashboard frontend..."
	cd frontend && npm ci && npm run build

batchalign-book-build:
	@echo "==> Building imported Batchalign book..."
	mdbook build batchalign-book

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
	@echo "==> [G6] Fragment parsing semantics"
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
