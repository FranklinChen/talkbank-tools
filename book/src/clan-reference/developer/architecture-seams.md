# Current Architecture Seams

**Status:** Current
This page documents the current internal seams that contributors should preserve when adding or restructuring CLAN-related functionality.

## CLI command registration

Top-level CLI argument wiring is no longer in one file.

- shared CLI args live in `crates/talkbank-cli/src/cli/args/core.rs`
- shared CLAN filters and common options live in `crates/talkbank-cli/src/cli/args/clan_common.rs`
- CLAN command variants live in `crates/talkbank-cli/src/cli/args/clan_commands.rs`

If you add a new CLAN command, register it in the appropriate split argument module instead of rebuilding a monolithic `args.rs`.

## CLAN dispatch

`run_clan` now lives in `crates/talkbank-cli/src/commands/clan/mod.rs` and dispatches into category files:

- `analysis.rs`
- `transforms.rs`
- `converters.rs`
- `compatibility.rs`

Keep family-specific logic in those modules. Shared file resolution, filtering, and output helpers belong in `helpers.rs` or another shared helper module, not copied into each family.

## Validation output

Parallel validation output now has a renderer seam:

- orchestration and stats live in `crates/talkbank-cli/src/commands/validate_parallel/runtime.rs`
- output shaping lives in `crates/talkbank-cli/src/commands/validate_parallel/renderer.rs`
- audit-specific behavior lives in `crates/talkbank-cli/src/commands/validate_parallel/audit.rs`

If you need a new output mode, add a renderer implementation instead of extending a large runtime `match`.

Audit-mode JSONL writing is also intentionally isolated. `crates/talkbank-cli/src/commands/validate/audit_reporter.rs` owns a dedicated writer thread and a cloneable reporting handle for workers, so future audit changes should preserve that explicit ownership boundary instead of reintroducing shared writer locks.

## Dashboard state ownership

`test-dashboard` now uses message passing rather than shared UI state. The worker sends `DashboardEvent` values and the UI reduces them into `AppState`.

That architecture is easier to test and reason about than `Arc<Mutex<AppState>>`. New dashboard features should generally be introduced as:

1. a new event variant
2. reducer logic in `app.rs`
3. worker-side emission in `runner.rs`

## Editor integration note

The VS Code extension and `talkbank-lsp` also now use a typed execute-command boundary, but the full developer note for that lives in `docs/developer-architecture.md` at the repo root because it spans more than the CLAN crate.
