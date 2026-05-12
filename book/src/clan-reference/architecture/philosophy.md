# Design Philosophy

**Status:** Current
**Last updated:** 2026-05-12 08:37 EDT

## Semantic AST processing

The original CLAN processes CHAT files as text, using string-prefix checks like `starts_with('&')` or `== "xxx"` to identify word types. The Rust reimplementation works on a parsed AST where every word has typed fields (`Word.category`, `Word.untranscribed()`) and where replacements live as a distinct `UtteranceContent::ReplacedWord` variant wrapping the original `Word`. This eliminates entire classes of bugs where string patterns match unintended content.

## Two-layer architecture

```text
framework/     Shared infrastructure (runner, filters, output, word utilities)
commands/      Individual command implementations
transforms/    File-modifying commands
converters/    Format conversion commands
```

The framework handles file discovery, CHAT parsing, speaker/tier/word/gem/range filtering, and output formatting. Commands implement only their analysis logic.

## Typed results

Every command defines a result struct implementing `CommandOutput`. This struct is the single source of truth — `render_text()`, `render_clan()`, `render_csv()`, and JSON output (via the `Serialize` bound and `to_json_value()` / `render(OutputFormat::Json)`) all derive from the same data. No ad-hoc string building that could drift between formats.

## Stateless commands

Commands hold only configuration (`Config`). All mutable state lives in a separate `State` type that is built up utterance-by-utterance via `process_utterance()`, optionally settled per file in `end_file()`, and consumed at the end of the run by `finalize(state) -> Output`. This makes the data flow explicit and testable.

## Parity by default

Output from `render_clan()` must match legacy CLAN output exactly, warts and all. Improvements go in `render_text()`. Golden tests compare against actual CLAN binary output to enforce this.
