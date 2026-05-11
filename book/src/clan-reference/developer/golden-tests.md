# Golden Tests

**Status:** Current
Golden tests in `tests/clan_golden.rs` and `tests/transform_golden.rs` compare `chatter clan` output against the legacy CLAN C binaries character-by-character.

## How they work

1. Run the legacy CLAN binary on a reference corpus fixture file
2. Run the equivalent `chatter clan` command on the same file
3. Snapshot both outputs with `insta` (suffixed `@clan` and `@rust`)
4. Any difference is a parity divergence that must be explained or fixed

The `@clan` snapshots capture CLAN's exact output and should never change (they represent the ground truth). The `@rust` snapshots capture our output and should match `@clan` as closely as possible.

## Requirements

Golden tests require CLAN binaries. The test runtime looks for them in:
1. `CLAN_BIN_DIR` env var (if set)
2. `../OSX-CLAN/src/unix/bin/` (workspace sibling)
3. `~/OSX-CLAN/src/unix/bin/` (legacy home path)

Tests are automatically skipped when binaries aren't found, making them CI-safe.

## Fixture files

Test fixtures come from the reference corpus at `corpus/reference/` (at the repo root). Common fixtures include:

- `tiers/mor-gra.cha` — morphology and grammar tiers
- `ca/overlaps.cha` — CA overlap markers
- Various language-specific files for encoding edge cases

## Current parity

**95% (113/118)** — 5 accepted divergences across 2 commands:

- **DELIM (4)**: CLAN writes an empty file when no changes are needed; we always write the full file
- **UNIQ (1)**: Unicode sort order for `U+230A` — C `strcoll()` vs Rust byte-order produces a single line swap with identical content and counts

## Adding a golden test

For an analysis command:

```rust
#[test]
fn golden_newcmd_fixture() {
    let file = corpus_file("path/to/fixture.cha");
    if clan_available() {
        let clan_output = run_clan("newcmd", &file, &[]);
        insta::assert_snapshot!("newcmd_fixture@clan", &clan_output);
    }
    let rust_output = run_rust_cmd("newcmd", &file, &[]);
    insta::assert_snapshot!("newcmd_fixture@rust", &rust_output);
}
```

For a transform command, see `tests/transform_golden.rs` for the pattern — transforms write to temp files rather than stdout.

## Diagnosing parity breaks

When a golden test fails:

1. Run `cargo insta review` to see the diff
2. Check if the `@clan` snapshot changed (it shouldn't — that means the CLAN binary or fixture changed)
3. If only `@rust` changed, determine whether it's a bug fix or regression
4. If the divergence is intentional and justified, accept it and document in [Per-Command Divergences](../divergences/per-command.md)
