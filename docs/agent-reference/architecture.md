# Architecture reference

**Last modified:** 2026-09-09 20:37 EDT

Read the sections relevant to your task. [AGENTS.md](../../AGENTS.md)
is the canonical policy entry point and resolves workflow conflicts here.
Dated incidents, measurements, versions and paths are historical evidence;
verify current source and live artifacts before relying on them. Inline
repository paths are relative to the repository root unless stated otherwise.

## What this repo is now (read first)

`talkbank-tools` is the **batchalign3 workspace**: the Batchalign ML pipeline
(ASR, forced alignment, neural morphotag, utterance segmentation), its PyO3
bridge, Python package, dashboard, and experimental desktop shell. **It is no
longer a CHAT-format toolchain.** The CHAT core (grammar, spec, tree-sitter
parser, data model, validation, transform, CLI, LSP, CLAN) lives wholly in the
**chatter** repo (`TalkBank/chatter`, sibling clone at `../chatter`), which is
the single home for the CHAT format. This workspace **consumes** chatter's
crates.

History: chatter was extracted from talkbank-tools in 2026-05/06; the duplicate
CHAT core was then removed from talkbank-tools and batchalign repointed at
chatter on 2026-06-18.

### How the CHAT core is consumed

`[workspace.dependencies]` in `Cargo.toml` points `talkbank-model`,
`talkbank-parser`, `talkbank-parser-re2c`, `talkbank-parser-tests`, and
`talkbank-transform` at the published, public chatter via git deps pinned to a
release tag (see `Cargo.toml` for the current tag; e.g. `{ git = "https://github.com/TalkBank/chatter", tag = "v0.3.5" }`).
A plain checkout builds with no `../chatter` sibling. Adopt a newer chatter by
bumping the tag (then `cargo update`). **Do NOT re-introduce copies of those
crates here**; chatter owns them. New CHAT-format / grammar / spec / parser /
validation / CLAN work goes in chatter, not here.

To co-develop chatter+batchalign locally, add an UNCOMMITTED `[patch]` that
points the git deps at your local checkout (never commit it; committed builds
stay self-contained):

```toml
[patch."https://github.com/TalkBank/chatter"]
talkbank-model = { path = "../chatter/crates/talkbank-model" }
talkbank-parser = { path = "../chatter/crates/talkbank-parser" }
talkbank-parser-re2c = { path = "../chatter/crates/talkbank-parser-re2c" }
talkbank-parser-tests = { path = "../chatter/crates/talkbank-parser-tests" }
talkbank-transform = { path = "../chatter/crates/talkbank-transform" }
```

## Crates in this workspace

| Crate | Purpose |
|-------|---------|
| `batchalign` | The Batchalign pipeline: ASR, FA, morphotag, jobs/runner, store, dashboard API |
| `batchalign-transform` | Batchalign-specific CHAT transforms (`asr_postprocess`, `morphosyntax`, `utseg`, FA `decisions`, `compare`, `build_chat`, `dp_align`, ...) layered over chatter's generic `talkbank-transform`, which it re-exports via a facade (`pub use talkbank_transform::*`) |
| `batchalign-pyo3` | PyO3 bridge for the Python package |
| `batchalign-types`, `batchalign-whisper-pilot` (experimental) | Shared types |

Plus `apps/dashboard-desktop` (Tauri shell, experimental, excluded from CI
gates), `frontend/` (React dashboard), the `batchalign` / `batchalign_core`
Python packages, and `xtask` (build helpers).

## Crate boundary

The `batchalign-*` crates are the ML application; they **consume** chatter's
`talkbank-*` crates and never reimplement CHAT primitives. A CHAT primitive has
one home (chatter). Decision test for new code: if it fundamentally needs ML
models, audio/signal processing, network services, or fleet runtime, it belongs
here; otherwise it belongs in chatter.

## %mor / morphotag note

Batchalign emits Universal Dependencies (UD) `%mor` syntax (hyphen-separated
features, sentence-case tags), consumed/validated by chatter. Legacy CLAN-mor `&`
fusional markers are not produced. The canonical %mor/validation rules live in
chatter; this repo produces UD-tagged output and relies on chatter to validate it.
