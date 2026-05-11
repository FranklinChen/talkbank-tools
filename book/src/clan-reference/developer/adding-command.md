# Adding a Command

**Status:** Current
## Steps

1. **Create the module** — `src/commands/<name>.rs` with four types:
   - `Config` — command configuration (from CLI flags)
   - `State` — mutable accumulator (implements `Default`)
   - `Result` — output data (implements `Serialize`, `Debug`, `CommandOutput`)
   - `Command` — unit struct implementing `AnalysisCommand`

2. **Register** — add `pub mod <name>;` to `src/commands/mod.rs`

3. **Wire CLI** — add a subcommand variant to `ClanCommands` in
   `crates/talkbank-cli/src/cli/args/clan_commands.rs`

4. **Wire dispatch** — add the match arm in the appropriate family module under
   `crates/talkbank-cli/src/commands/clan/` and keep `run_clan()` in
   `crates/talkbank-cli/src/commands/clan/mod.rs` as the thin top-level dispatcher

5. **Add golden test** — add a test case in the relevant file under
   `crates/talkbank-clan/tests/clan_golden/`

## Skeleton

```rust,ignore
//! # NAME — Brief description
//!
//! What the command does and when to use it.

use crate::framework::prelude::*;

/// Command configuration.
pub struct Config {
    // fields from CLI flags
}

/// Per-file accumulator.
#[derive(Default)]
pub struct State {
    // mutable state built up during processing
}

/// Analysis result.
#[derive(Debug, Serialize)]
pub struct Result {
    // typed output fields
}

impl CommandOutput for Result {
    fn render_text(&self) -> String { todo!() }
    fn render_clan(&self) -> String { todo!() }
}

/// The command.
pub struct Command;

impl AnalysisCommand for Command {
    type Config = Config;
    type State = State;
    type Result = Result;

    fn process_utterance(config: &Config, state: &mut State, utterance: &Utterance) {
        // accumulate per-utterance data
    }

    fn end_file(config: &Config, state: State) -> Self::Result {
        // compute final result from accumulated state
        todo!()
    }
}
```

## Conventions

- Use `countable_words()` for word iteration — don't roll your own filter
- Use `NormalizedWord` for frequency maps
- Handle missing `%mor` gracefully (skip morpheme counting, don't panic)
- Keep the module under 400 lines; split into submodules well before it becomes another 800+ line file
