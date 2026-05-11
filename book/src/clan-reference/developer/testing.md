# Testing Strategy

**Status:** Current
## Unit tests

Each command module includes `#[cfg(test)]` unit tests that verify counting logic, edge cases, and output formatting against known inputs.

```bash
cargo nextest run -p talkbank-clan          # All tests (preferred)
cargo test -p talkbank-clan                 # Alternative
cargo test --doc -p talkbank-clan           # Doctests only (nextest can't run these)
```

## Snapshot tests

Snapshot tests use `insta` to capture command output and detect regressions. Snapshots live in `tests/snapshots/`.

```bash
cargo insta review -p talkbank-clan         # Review pending snapshot changes
cargo insta accept --all                    # Accept all pending changes
```

When you change a command's output format, the snapshot test will fail. Review the diff to confirm the change is intentional, then accept.

## Golden tests

Golden tests compare our output against legacy CLAN C binaries. See [Golden Tests](golden-tests.md) for details.

## Running specific tests

```bash
# By name pattern
cargo nextest run -p talkbank-clan -E 'test(freq)'

# Only golden tests
cargo nextest run -p talkbank-clan -E 'test(golden)'

# Only indent tests
cargo nextest run -p talkbank-clan -E 'test(indent)'
```

## Test fixtures

Test fixtures are real CHAT files from the reference corpus at `corpus/reference/` (at the repo root). Never create ad hoc `.cha` test files — use existing corpus files or ask for new ones to be added to the reference corpus.

## Current test counts

376 tests across unit, snapshot, golden, and integration test suites.
