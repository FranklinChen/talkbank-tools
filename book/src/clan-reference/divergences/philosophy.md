# Why We Diverge

**Status:** Current
## The parity goal

The primary goal is **exact output parity** with legacy CLAN wherever possible. We match CLAN's output format, counting rules, and even its quirks (like reverse speaker ordering from C linked-list prepend patterns). Every implemented command has golden tests that compare our output character-by-character against the original CLAN C binary.

Current parity: **95% (113/118 golden snapshots match exactly)**. The 5 accepted divergences are in DELIM (4 cosmetic) and UNIQ (1 Unicode sort order difference).

## When we accept divergences

We diverge from CLAN only when one of these conditions holds:

1. **CLAN has a bug** that produces incorrect results — we fix the bug and document it. Examples: INDENT's infinite loop on certain overlap patterns, FREQ double-counting words in retrace groups.

2. **The difference is cosmetic** and doesn't affect analysis results — different whitespace, column alignment, or header formatting that no downstream tool depends on.

3. **Matching would require unreasonable complexity** for zero user benefit — e.g., replicating CLAN's 80-character line wrapping in output, or matching exact floating-point rounding from C's `printf("%.3f")`.

4. **We deliberately improve** on CLAN's behavior — structured output formats (JSON, CSV), recursive directory traversal, rich error diagnostics with file locations. These are extensions, not replacements of existing behavior.

## What we never change

- **Counting semantics**: If CLAN counts a word, we count it. If CLAN uses population SD (`/n`), we use population SD. Counting rules are discovered empirically through golden test comparison and documented in [Framework-Level Divergences](framework.md).

- **Speaker ordering**: CLAN outputs speakers in reverse encounter order due to its C linked-list prepend pattern. We replicate this, even though it's a data structure artifact, because users and scripts depend on it.

- **Algorithm behavior**: For stochastic commands like VOCD, we use the same random sampling algorithm (D_optimum curve fitting on 100 random samples at each token count from 35-50).

## The AST advantage

The most fundamental architectural difference: CLAN processes CHAT files as flat text using string pattern matching and character scanning. The Rust reimplementation parses files into a typed AST (Abstract Syntax Tree) and operates on structured data.

This means:
- Word classification uses typed fields (`word.category`, `word.untranscribed()`) instead of string prefix checks (`starts_with('&')`, `== "xxx"`)
- Annotations are structured objects with typed fields, not regex-extracted substrings
- Transform commands modify AST nodes, then serialize — guaranteeing structural validity
- Filters operate on typed speaker codes, tier names, and word categories

This eliminates entire classes of bugs where CLAN's string patterns match unintended content (e.g., a word starting with `&` in a speaker name being misclassified as a special form). The tradeoff is occasional whitespace or formatting differences in edge cases, which we track and document.

## Documentation policy

Every divergence — intentional or discovered — is documented in three places:

1. **Module doc comments**: Each command's `//!` header includes a "Differences from CLAN" section
2. **The book**: [Per-Command Divergences](per-command.md) provides a centralized reference
3. **Golden test snapshots**: The `@clan` vs `@rust` snapshot pairs make divergences visible in code review
