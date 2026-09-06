# Building & Development

**Status:** Current
**Last updated:** 2026-09-05 20:24 EDT

Development is supported on **Windows, macOS, and Linux**. The instructions below use Unix shell syntax; on Windows, use PowerShell or Git Bash equivalently.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** -- Python package manager (all platforms). Used for all dependency management and running commands.
- **Rust (stable)** via [rustup](https://rustup.rs/) (all platforms) -- needed for the Rust CLI and PyO3 extension.
- **Node.js + npm** -- needed for `make build` and `make build-dashboard`, which rebuild the embedded dashboard bundled into the Rust binary.
- **[maturin](https://www.maturin.rs/)** -- Required only if you modify the Rust `batchalign_core` extension.
- **Python 3.13 or 3.14** for development. Installers and deployments default
  to 3.13, while CI builds and tests both standard interpreter versions.
  Free-threaded 3.14t is **not** a supported install or deployment target; see
  [Python Versioning](python-versioning.md).
- **Platform note:** On macOS, `python` and `python3` may not exist outside a venv. Always use `uv run` to execute Python commands, which handles this automatically on all platforms.

## Rust compiler policy

Development, CI, and releases use current stable Rust. We do not maintain an
older minimum supported Rust version: update with `rustup update stable` when
needed, and evaluate dependencies against the compiler our supported builds use.

The former workspace `rust-version = "1.89.0"` was neither inherited by any
package nor tested by CI. It has been removed. An older-compiler commitment
would require a concrete consumer need and a CI job proving that commitment.

## Development Install

Batchalign source lives as the `batchalign-*` sibling crates inside
this `talkbank-tools` repo (the standalone `batchalign3` repo was
decommissioned 2026-04-28; there are no longer two siblings to clone).
A development checkout is one repo:

```bash
git clone https://github.com/TalkBank/talkbank-tools.git
cd talkbank-tools
make build
```

`make build` rebuilds the embedded dashboard, then runs `cargo build
--workspace --release`, which compiles every Rust crate (including the
PyO3 bridge `batchalign-pyo3`). For PyO3-specific work, the dedicated
target is:

```bash
make batchalign-build-wheel       # build the maturin wheel
make batchalign-python-prepare    # build + install the wheel into the dev env
```

`uv run batchalign3` then uses the installed wheel. Most contributors
skip the wheel step and rely on the dev fallback in
`batchalign/_cli.py`, which execs `target/{debug,release}/batchalign3`
when no packaged binary is present.

Never use `pip install` directly; `uv` manages the `.venv` and every
Python dependency.

## Running the CLI

In a source checkout, `uv run batchalign3` is the normal way to invoke
the installed console script. When no packaged binary is present
(`batchalign/_bin/batchalign3`), `batchalign/_cli.py` falls back to
`target/{debug,release}/batchalign3` and then to `cargo run -p
batchalign` as a last resort, so a single `cargo build -p batchalign`
up front gives you a fast iteration loop:

```bash
cargo build -p batchalign
uv run batchalign3 --help     # uses the debug target via the wrapper fallback
```

Reserve `uv run` for Python tools (pytest, mypy, maturin) when you are
not invoking the CLI.

```bash
make build
./target/debug/batchalign3 --help
./target/debug/batchalign3 transcribe input_dir -o output_dir --lang eng
./target/debug/batchalign3 morphotag input_dir -o output_dir
./target/debug/batchalign3 align input_dir -o output_dir

# Or let Cargo rebuild the Rust binary incrementally for you:
cargo run -p batchalign -- transcribe input_dir -o output_dir --lang eng
```

## What to Rebuild After Changes

Use the repo-native build targets so the Rust CLI, the shared
`batchalign` crate, and the `batchalign_core` PyO3 extension stay in
sync:

| What changed | What to rebuild |
| --- | --- |
| Python code only (`batchalign/`) | Nothing; the next worker process picks up the change |
| Rust CLI / server (`crates/batchalign/`) | `cargo build -p batchalign` |
| Shared chat logic (any `crates/`) or PyO3 bridge (`crates/batchalign-pyo3/`) | `make batchalign-python-prepare` (rebuilds the maturin wheel and reinstalls it into the dev env). For the fastest CLI loop, also build the CLI once (`cargo build -p batchalign`) so the wrapper can fall back to `target/debug/batchalign3`. |
| Command/orchestrator changes (`crates/batchalign/src/commands/`, `compare.rs`, `benchmark.rs`, `transcribe/`, `fa/`, `morphosyntax/`, `command_family.rs`, `text_batch.rs`) | `cargo build -p batchalign`, and `make batchalign-python-prepare` if the PyO3 bridge surface changed |
| Cross-cutting or dashboard changes | `make build` (requires Node.js + npm because it rebuilds the embedded dashboard, then runs `cargo build --workspace --release`) |

## Rebuilding the Rust Extension

The `batchalign_core` Python package is a PyO3 Rust extension built by
maturin. The repo-native rebuild path is:

```bash
make batchalign-build-wheel       # build the maturin wheel
make batchalign-python-prepare    # depends on batchalign-build-wheel; reinstalls into the dev env
```

The PyO3 crate (`crates/batchalign-pyo3/`) has no feature gates beyond
`extension-module`: no heavy CLI or Rev.AI dependencies. In a source
checkout, `batchalign/_cli.py` falls back to
`target/{debug,release}/batchalign3` when the packaged binary isn't
present, so most contributors do not need to install the wheel
during iteration.

To exercise the installed-package experience locally, build the CLI
once (`cargo build -p batchalign`) and copy it into
`batchalign/_bin/batchalign3` before running `make
batchalign-python-prepare`; the maturin `include` directive in
`pyproject.toml` will then bundle it into the wheel.

## CLI Binary Packaging (`batchalign/_bin/`)

batchalign3 ships two native artifacts in its wheel:

1. **`batchalign_core.so`**: the PyO3 extension (gives Python access to Rust
   CHAT parsing, alignment, etc.)
2. **`batchalign/_bin/batchalign3`**: the standalone Rust CLI binary (the
   server, job runner, and all commands)

The Python entry point (`batchalign/_cli.py`) locates and execs the native CLI
binary. It searches three locations in order:

1. **Packaged binary** at `batchalign/_bin/batchalign3`: this is what GitHub
   Release installers and downloaded wheels use. The binary is bundled inside
   the wheel.
2. **Dev checkout** at `target/{debug,release}/batchalign3`: for developers
   who built the CLI with `cargo build`.
3. **Cargo fallback**: execs `cargo run -p batchalign` to compile on the
   fly.

### Why `_bin/` is gitignored

The binary is a 50+ MB platform-specific build artifact, it must not be
tracked in git. Instead:

- **Locally:** copy `target/release/batchalign3` (or `target/debug/batchalign3`)
  into `batchalign/_bin/` before running `make batchalign-python-prepare`
  if you need the installed-package experience. Most developers skip this and
  rely on the dev-checkout fallback (`target/debug/batchalign3`).
- **CI:** A dedicated `build-cli` job compiles the CLI binary once (release
  mode), uploads it as an artifact, and each Python-version wheel build
  downloads it into `batchalign/_bin/` before maturin packages it.
- **Release:** The release workflow builds platform-specific CLI binaries
  (macOS ARM + Intel, Linux x86 + ARM, Windows x86) and packages each into
  the corresponding wheel.

### Maturin include directive

`pyproject.toml` tells maturin to include the binary in the wheel:

```toml
[tool.maturin]
include = [
    { path = "batchalign/_bin/batchalign3", format = "wheel" },
    { path = "batchalign/_bin/batchalign3.exe", format = "wheel" },
]
```

If the binary doesn't exist at build time, maturin silently skips it, the
wheel still builds but `batchalign3 --help` will fail at runtime with
"CLI binary not found." This is why CI must build and copy the binary
**before** running the maturin wheel build.

## Where Command Logic Should Live

If you are changing command behavior, the first stop should be the owning
command module in `crates/batchalign/src/commands/` and then the module
that actually owns the algorithmic or orchestration semantics (`compare.rs`,
`benchmark.rs`, `transcribe/`, `fa/`, `morphosyntax/`, etc.).

- `crates/batchalign/src/commands/` owns released-command identity, specs,
  and the top-level contributor-facing entrypoints.
- `crates/batchalign/src/command_family.rs` keeps the small command-shape
  enum used by command metadata.
- `crates/batchalign/src/text_batch.rs` keeps reusable text-batch helper
  types for commands such as `utseg`, `translate`, and `coref`.
- `crates/batchalign/src/runner/` owns job lifecycle, queueing, and shared
  dispatch machinery.
- `crates/batchalign/src/runner/dispatch/` (benchmark_pipeline.rs,
  fa_pipeline.rs, transcribe_pipeline.rs, infer_batched.rs, audio_task.rs,
  asr_media.rs, media_analysis_v2.rs, options.rs, plan.rs, utr.rs) should
  stay thin and focus on argument parsing, capability gating, and whether
  a command runs locally or through the server.
- `crates/batchalign-pyo3/` should stay a thin bridge, not the place where new command logic is
  invented.

Run the Rust test suite to verify your changes:

```bash
cargo test --manifest-path crates/batchalign-pyo3/Cargo.toml
```

## Type Checking

Run the current mypy gate before every commit:

```bash
uv run mypy                       # mypy only
make batchalign-typecheck-python  # mypy under the batchalign- target group used by CI
make lint-affected                # affected-Rust clippy + affected Python mypy
```

Strictness lives in `mypy.ini`, and CI runs the same repo-native
command shape.

Do not commit with mypy errors. Use `# type: ignore[<code>]` only when
necessary, and always include the specific error code.

## Type Annotation Rules

All new and modified code must include type annotations:

- Annotate all function parameters and return types.
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`.
- **`Any` and `object` are banned as type annotations.** Use specific types. For ML library types that are expensive to import, use `TYPE_CHECKING` guards with the real type.
- Use `from __future__ import annotations` for forward references where needed.
- Prefer `TYPE_CHECKING` imports for heavy dependencies used only in annotations.

## The CHAT Format Rule

All CHAT parsing and serialization must go through principled AST
manipulation in Rust. Python never touches CHAT text directly.

**Do not:**
- Use regex or string splitting to extract or modify CHAT content from Python.
- Process CHAT line-by-line in Python.
- Manipulate CHAT header metadata with ad-hoc text code.

**Instead:**
- From Python, shell out to the `batchalign3` CLI (`validate`, `to-json`,
  command-specific subcommands) and consume its structured output, or
  raise/catch `batchalign_core.CHATValidationException` at the parser
  boundary (the typed exception that the PyO3 layer surfaces).
- All CHAT AST manipulation lives in the Rust crates (`talkbank-parser`,
  `talkbank-model`, `talkbank-transform`, `batchalign`). When new
  AST-level behaviour is needed, add it on the Rust side and expose it
  through the CLI; do not invent a new Python-facing parsing surface.

CHAT has complex escaping, continuation lines, and encoding rules that
ad-hoc text manipulation will get wrong. The Rust AST handles all of
this correctly; the 2026-03-21 PyO3 slimdown deliberately retired the
older user-facing PyO3 parse / build / add-morphosyntax bindings in
favour of this CLI-and-typed-exception boundary.
