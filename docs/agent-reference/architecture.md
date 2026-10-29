# Architecture reference

**Last modified:** 2026-10-29

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

`[workspace.dependencies]` in `Cargo.toml` pins `talkbank-model`,
`talkbank-parser`, `talkbank-parser-tests` and `talkbank-transform` to one
published Chatter release tag, resolved to an exact commit in `Cargo.lock`.
The current selection is v0.33.0. The manifest and lockfile own the identity;
application version strings do not identify the compiled dependency.
Adopt a newer reviewed Chatter commit by updating all pins and the lockfile
together, then verify the consuming targets. **Do NOT re-introduce copies of those
crates here**; chatter owns them. New CHAT-format / grammar / spec / parser /
validation / CLAN work goes in chatter, not here.

Co-development uses an immutable, buildable Chatter checkpoint, not an
uncommitted dependency tree or a committed local path patch. Local qualification
of an unpublished checkpoint is not clean-machine installation evidence.
Publish Chatter before claiming that a dependent candidate is publicly
reproducible; a published dependency builds without a sibling checkout.

The parser, model and transforms are linked into the BA3 binary and its
extension. Installing or deploying the standalone Chatter CLI does not change
those compiled libraries. A later Chatter release can ship independently;
BA3 adopts it only through a reviewed pin/lock update, compatibility checks,
rebuild and deployment. BA3 fixes that already work on the selected published
release do not wait for an unpublished Chatter candidate.

## Crates in this workspace

| Crate | Purpose |
|-------|---------|
| `batchalign` | The Batchalign pipeline: ASR, FA, morphotag, jobs/runner, store, dashboard API |
| `batchalign-transform` | Batchalign-specific CHAT transforms (`asr_postprocess`, `morphosyntax`, `utseg`, FA `decisions`, `compare`, `build_chat`, `dp_align`, ...) layered over chatter's generic `talkbank-transform`, which it re-exports via a facade (`pub use talkbank_transform::*`) |
| `batchalign-pyo3` | PyO3 bridge for the Python package |
| `batchalign-types` | Shared types |

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
