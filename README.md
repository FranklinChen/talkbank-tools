# talkbank-tools

**Status:** Current
**Last updated:** 2026-04-28 16:51 EDT

[![CI](https://github.com/TalkBank/talkbank-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/TalkBank/talkbank-tools/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

The CHAT toolchain from [TalkBank](https://talkbank.org/). Parses, validates, converts, and analyzes [CHAT](https://talkbank.org/0info/manuals/CHAT.html) transcription files — the standard format for annotated speech data across child language, aphasia, dementia, bilingualism, and conversation analysis research.

**`chatter`** is the CLI. The same engine powers a **VS Code extension** with live diagnostics and a **Rust API** for building your own tools.

This repo also now carries the imported **Batchalign** application stack: `crates/batchalign-*`, the `batchalign/` Python package, `pyo3/`, the dashboard frontend, and desktop wrapper sources. The unification is still in progress, but `talkbank-tools` is already the canonical home for the merged engineering surface.

**Windows, macOS, and Linux.** See [docs/PLATFORM-SUPPORT.md](docs/PLATFORM-SUPPORT.md) for platform support tiers.

**Release status:** The CLI (`chatter`) and Rust crates are the primary stable product. The VS Code extension is in preview. The desktop app is experimental. See [docs/RELEASE-CONTRACT.md](docs/RELEASE-CONTRACT.md) for details.

## Install

### Pre-built binaries (recommended)

Download the release archive for your platform from **[GitHub Releases](https://github.com/TalkBank/talkbank-tools/releases)**. Each archive contains two binaries: `chatter` (the CLI) and `talkbank-lsp` (the standalone language server for editor integrations):

| Platform | Archive |
|----------|---------|
| macOS (Apple Silicon) | `chatter-aarch64-apple-darwin.tar.gz` |
| macOS (Intel) | `chatter-x86_64-apple-darwin.tar.gz` |
| Linux (x86_64) | `chatter-x86_64-unknown-linux-gnu.tar.gz` |
| Windows (x86_64) | `chatter-x86_64-pc-windows-msvc.zip` |

Extract and add to your PATH, then verify: `chatter --version`

### Build from source

Requires [Rust](https://rustup.rs/) (stable):

```bash
cargo install --path crates/talkbank-cli   # installs chatter
cargo install --path crates/talkbank-lsp   # installs talkbank-lsp (optional, for editor integrations)
```

---

## Validate

Check any CHAT file — or an entire corpus — for structural, alignment, and semantic errors:

```bash
chatter validate transcript.cha
chatter validate corpus/              # recursive, parallel, cached
chatter validate corpus/ --format json --audit results.jsonl
```

### Rich diagnostics

Errors display source context with the exact location highlighted, powered by [miette](https://crates.io/crates/miette). Alignment errors include a columnar diff showing exactly which items match, which are missing (`⊖`), and which are extra (`⊕`):

```
  × error[E705]: Main tier has 5 alignable items, but %mor tier has 4 items

  │ Main tier                      %mor tier
  │ ────────────────────────────   ────────────────────────────
  │ I                              pro|I
  │ want                           v|want
  │ to                             inf|to
  │ go                             v|go
  │ home                           — ⊖

   ╭─[input:6:1]
 6 │ *CHI:   I want to go home .
   ·               ╰── here
 7 │ %mor:       pro|I v|want inf|to v|go
   ·                   ╰── %mor tier
   ╰────
  help: Each alignable word in main tier must have corresponding %mor item
```

Every error has a stable code (198 across 7 categories), a source span, and [documentation with fix guidance](book/src/user-guide/validation-errors.md). For `%wor` tier mismatches, the diff uses LCS-based fuzzy matching to align items by content rather than position.

| Range | Category |
|-------|----------|
| E1xx | UTF-8 and encoding |
| E2xx | File structure (`@Begin`, `@End`, headers) |
| E3xx | Main tier (speakers, terminators, content) |
| E4xx-E5xx | Headers and dependent tier structure |
| E6xx | Dependent tier validation |
| E7xx | Alignment (`%mor`, `%gra`, `%pho`, `%wor`) |
| W1xx-W6xx | Warnings |

### Interactive TUI

When run in a terminal, `chatter validate` opens an interactive two-pane browser — file list on the left, errors with source context on the right. Results stream live as files are processed:

```
┌─ Files (128 validated) ──────────┐┌─ Errors ─────────────────────────────────────────┐
│   Eng-UK/Thomas/020300a.cha  (0) ││ × error[E725]: %modsyl has 7 words but %mod has  │
│   Eng-UK/Thomas/020300b.cha  (0) ││   6: word counts must match                      │
│ ✗ French-EP/PF01_F.cha      (4) ││                                                   │
│ ✗ French-EP/PF03_EP.cha     (2) ││  183 │ %xmodsyl: ɛ̃:N t:Oʁ:Oy:Nk:E p:Ou:Nʁ:E ... │
│   French-EP/PF05_F.cha      (0) ││       ╰── here                                    │
│ ✗ French-EP/PF06_EP.cha     (3) ││                                                   │
│   Japanese/Aki/020212.cha    (0) ││ × error[E705]: Main tier has 5 items, but %mor    │
│ ✗ Japanese/Aki/020318.cha    (1) ││   has 4                                           │
│   Spanish/Irene/011006.cha   (0) ││                                                   │
│                                  ││  Main tier       %mor tier                        │
│                                  ││  ──────────────  ──────────────                   │
│                                  ││  I               pro|I                            │
│                                  ││  want            v|want                           │
│                                  ││  to              inf|to                           │
│                                  ││  go              v|go                             │
│                                  ││  home            — ⊖                              │
└──────────────────────────────────┘└───────────────────────────────────────────────────┘
  ↑/↓ navigate  Tab switch pane  r rerun  q quit
```

Supports `--theme dark|light` and custom themes via `~/.config/chatter/theme.toml`. Disable with `--tui-mode disable` for plain text output or piping.

```bash
chatter watch transcript.cha          # live re-validation on save
chatter lint transcript.cha           # auto-fix common issues
chatter lint transcript.cha --dry-run # preview fixes
```

---

## JSON Data Model

Every CHAT structure — headers, utterances, words, morphology, syntax, timing, annotations — is captured in a fully typed AST with lossless roundtrip fidelity. Backed by a published [JSON Schema](schema/).

```bash
chatter to-json transcript.cha -o transcript.json
chatter from-json transcript.json -o roundtripped.cha
chatter schema                        # print the JSON Schema
```

Words carry both the original transcript form and NLP-ready cleaned text:

```json
{
  "type": "word",
  "raw_text": "dog(s)",
  "cleaned_text": "dogs",
  "content": [{ "type": "text", "content": "dog" }],
  "category": null
}
```

This makes CHAT data accessible to Python, JavaScript, R, or any language that reads JSON — no CHAT parser required.

---

## VS Code Extension

> **Status: Preview** — Functional and actively developed, but not yet at 1.0.

A full-featured CHAT editor that replaces the legacy CLAN application. See [`vscode/`](vscode/) for installation, [full documentation](book/src/user-guide/vscode-extension.md) for details.

- **Live validation** — same 198 error codes as `chatter validate`, with quick fixes (Cmd+.)
- **Cross-tier alignment** — hover any word to see its `%mor`, `%gra`, `%pho` alignment; click to highlight linked items across tiers
- **Media playback** — play audio/video segments from timing bullets, continuous playback with cursor tracking, waveform view, loop, rewind, speed control
- **Transcription mode** — stamp timing bullets while audio plays (F4), auto-insert new speaker lines
- **Walker mode** — step through utterances one at a time with synchronized audio (Alt+Up/Down)
- **33 CLAN analysis commands** — run directly in VS Code with styled result panels, tables, charts, and CSV export
- **Assessment tools** — KidEval, Eval, and Eval-D panels with normative database comparison
- **Dependency graphs** — visualize `%gra` syntax trees (Cmd+Shift+G), rendered via Graphviz WASM
- **Navigation** — go to definition (F12), find all references, rename speaker (F2), tier-scoped search, folding, smart selection
- **Participant editor** — edit `@ID` headers in a table view
- **Coder mode** — annotate utterances with hierarchical coding schemes from `.cut` files
- **Special characters** — compose CA and CHAT symbols via Cmd+Shift+1/2
- **Built on a Language Server** ([`talkbank-lsp`](crates/talkbank-lsp/)) — works with any LSP-compatible editor

---

## Desktop App

> **Status: Experimental** — The desktop app is in early development and not yet part of the public release.

A native desktop application for validating CHAT files — designed for linguists and researchers who prefer a graphical interface over the terminal. Drag-and-drop files or folders, see errors with source context, and click through a collapsible file tree.

- **Drag-and-drop** or use file/folder picker buttons
- **Streaming progress** — results appear as files are validated
- **Source snippets** with caret underlines for every error
- **Open in CLAN** — jump to the error location in the CLAN editor
- **Export** — save results as JSON or text

Built with [Tauri v2](https://v2.tauri.app/) (Rust backend, React frontend). Uses the same validation engine as `chatter validate` — same errors, same caching, same parallel pipeline.

```bash
# Development
cd desktop && npm install
cargo tauri dev

# Build a distributable app
cargo tauri build
```

See [`desktop/`](desktop/) for the source.

---

## CLAN Analysis Commands

80 CLAN subcommands: 34 analysis commands, 21 transforms, 15 format converters, plus stubs and aliases for CLAN compatibility. All with consistent filtering and output options.

```bash
chatter clan freq corpus/             # word frequency + type-token ratio
chatter clan mlu corpus/              # mean length of utterance
chatter clan kwal corpus/ --include-word "want"  # keyword-in-context search
chatter clan vocd corpus/ --speaker CHI          # vocabulary diversity (D statistic)
chatter clan freq corpus/ --speaker CHI --gem Narrative --format csv
```

### Shared filtering

All commands support: `--speaker`, `--exclude-speaker`, `--gem`, `--include-word`, `--exclude-word`, `--range`, `--format text|json|csv|clan`. Filters evaluate cheapest-first (range → speakers → gems → words).

### Analysis commands

| Category | Commands |
|----------|----------|
| **Lexical** | freq, freqpos, vocd, corelex, maxwd, wdlen, wdsize, phonfreq |
| **Grammar** | mlu, mlt, sugar, ipsyn, complexity, dss, mortable, megrasp |
| **Clinical** | eval, kideval, flucalc, modrep |
| **Interaction** | chip, timedur, cooccur, dist |
| **Search** | kwal, combo |
| **Codes** | codes, chains, keymap, gemlist |
| **Accuracy** | trnfix, script, rely, uniq |
| **Morphology** | mor, post, postlist, posttrain, postmodrules |

### Transforms

File-modifying commands: `flo`, `lowcase`, `chstring`, `trim`, `tierorder`, `retrace`, `repeat`, `roles`, `dates`, `delim`, `fixbullets`, `compound`, `dataclean`, `gem`, `makemod`, `ort`, `indent`, `longtier`, `lines`, `quotes`, `combtier`, `postmortem`, `fixit`.

### Format converters

Bidirectional conversion between CHAT and external tools:

| Converter | Direction |
|-----------|-----------|
| ELAN (`.eaf`) | chat2elan, elan2chat |
| Praat TextGrid | chat2praat, praat2chat |
| SRT subtitles | chat2srt, srt2chat |
| Plain text | chat2text, text2chat |
| SALT, LENA, LIPP, PLAY, LAB, RTF | → CHAT |

---

## Language Server (LSP)

The [`talkbank-lsp`](crates/talkbank-lsp/) crate implements a full [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) server. While it powers the VS Code extension, it works with **any LSP-compatible editor** — Neovim, Emacs, Sublime Text, Helix, Zed, etc.

**Standard LSP features:** hover, completion, go-to-definition, find references, rename, document symbols, workspace symbols, code actions, formatting, folding, semantic tokens, inlay hints, selection ranges, linked editing, document links, code lens.

**CHAT-specific features** (via custom LSP commands):
- Cross-tier alignment hover — see `%mor`/`%gra`/`%pho` alignment for any word
- Alignment sidecar — structured JSON of per-utterance alignment data
- Dependency graph generation — Graphviz DOT from `%gra` tiers
- CLAN analysis execution — run any of 80 subcommands from the editor
- Tier-scoped search — search within specific dependent tiers
- Participant metadata — parse and format `@ID` headers
- Document filtering — extract utterances by speaker

**Performance:** Incremental parsing via tree-sitter. Single-utterance edits re-validate only the affected utterance (~100-1000x faster than full-file reparse). Splice detection handles insertions and deletions without rebuilding.

---

## Tree-Sitter Grammar

A complete [tree-sitter](https://tree-sitter.github.io/) grammar for the CHAT format, providing incremental parsing, syntax highlighting, and the foundation for all downstream tools.

- **372 grammar rules**, **380 named node types**, covering the full CHAT specification
- **66 header types** with structured subfields (e.g., `@ID` has 13 pipe-delimited components)
- **31 dependent tier types** (`%mor`, `%gra`, `%pho`, `%sin`, `%wor`, `%cod`, translations, and more)
- **16 utterance terminators**, overlap markers, pause types, linkers, and CA symbols
- **18 annotation types** (error markers, retraces, replacements, paralinguistic material)
- **160 test cases** organized by category (headers, main tiers, dependent tiers, words, errors)
- Follows the **"parse, don't validate"** design — the grammar accepts broad input; the Rust validator flags invalid constructs with specific error codes

The grammar lives in [`grammar/`](grammar/) and generates the C parser used by both the Rust crates and the LSP server.

---

## Rust API

The entire toolchain is available as a library. Parse CHAT into a fully typed AST, validate, inspect or transform, and serialize back — no CLI subprocess needed.

```rust,no_run
use talkbank_model::ParseValidateOptions;
use talkbank_transform::parse_and_validate;

// Parse + validate in one call
let options = ParseValidateOptions::default().with_validation();
let file = parse_and_validate(chat_text, options).unwrap();

// Typed access to the full CHAT structure
for utterance in file.utterances() {
    let speaker = utterance.speaker_code();
    let words = utterance.main_tier_text();
    // %mor, %gra, %pho, timing, annotations — all typed
}
```

The same repo now also contains the Batchalign stack — shared deterministic CHAT/text logic is being re-homed into canonical `talkbank-*` crates, while the Batchalign runtime/application layer lives under `crates/batchalign-*`, `batchalign/`, `frontend/`, and `pyo3/`.

| Crate | What it does |
|-------|--------------|
| [`talkbank-model`](crates/talkbank-model/) | Data model (AST), 198 validation rules, cross-tier alignment, content walker |
| [`talkbank-parser`](crates/talkbank-parser/) | Parser — tree-sitter CST to typed model |
| [`talkbank-transform`](crates/talkbank-transform/) | Pipelines: parse+validate, CHAT/JSON roundtrip, batch caching |
| [`talkbank-clan`](crates/talkbank-clan/) | 80 CLAN subcommands: 34 analysis, 21 transforms, 15 converters |
| [`talkbank-cli`](crates/talkbank-cli/) | `chatter` CLI binary with interactive TUI |
| [`talkbank-lsp`](crates/talkbank-lsp/) | Language Server with incremental parsing and 12 custom commands |
| [`talkbank-derive`](crates/talkbank-derive/) | Proc macros: SemanticEq, SpanShift, error_code_enum |
| [`batchalign-types`](crates/batchalign-types/) | Shared Batchalign API, job, and worker protocol types |
| [`batchalign-chat-ops`](crates/batchalign-chat-ops/) | Batchalign CHAT-aware ops, FA/speaker seams, compatibility layer during merge |
| [`batchalign-app`](crates/batchalign-app/) | Batchalign server/runtime orchestration and API surface |
| [`batchalign-cli`](crates/batchalign-cli/) | `batchalign3` CLI binary and operational commands |
| [`chatter-desktop`](desktop/) | Native desktop validation app (Tauri v2, React) |

See the [integration guide](book/src/integrating/library-usage.md) for API usage, the [JSON output reference](book/src/integrating/json-output.md), and the [diagnostic contract](book/src/integrating/diagnostic-contract.md).

---

## More Commands

```bash
chatter normalize transcript.cha      # rewrite to canonical format
chatter new-file --lang eng --speakers CHI,MOT
chatter show-alignment transcript.cha # debug %mor/%gra/%pho alignment
chatter cache stats                   # validation cache info
chatter cache clear                   # clear cached results
```

## Documentation

| | |
|---|---|
| **[User Guide](book/src/user-guide/installation.md)** | Installation, CLI reference, batch workflows, processing playbook |
| **[CHAT Format](book/src/chat-format/overview.md)** | Headers, utterances, dependent tiers, word syntax, symbols |
| **[Validation Errors](book/src/user-guide/validation-errors.md)** | All 198 error codes with examples and fixes |
| **[VS Code Extension](book/src/user-guide/vscode-extension.md)** | Setup, features, and configuration |
| **[Desktop App](book/src/user-guide/desktop-app.md)** | Drag-and-drop validation for non-terminal users |
| **[Batchalign Book](batchalign-book/src/introduction.md)** | Batchalign CLI, server, worker, and developer docs during the merge |
| **[Integration Guide](book/src/integrating/library-usage.md)** | Rust API, JSON output format, JSON Schema, diagnostic contract |
| **[CHAT Manual](https://talkbank.org/0info/manuals/CHAT.html)** | The official CHAT format specification (talkbank.org) |

## Building from Source

Requires Rust (stable, edition 2024). Node.js needed only for grammar/spec tooling.

```bash
make build          # build grammar + Rust workspace
make test           # run all tests (nextest + doctests + spec tools)
make verify         # pre-merge verification gates (G0-G10)
```

`make test-gen` is available when spec-driven generated artifacts need to be
refreshed. It is not the universal answer for parser-semantic changes; direct-
parser fragment and recovery behavior often needs direct tests.

## License

BSD-3-Clause. Copyright (c) 2026, Carnegie Mellon University. See [LICENSE](LICENSE).
