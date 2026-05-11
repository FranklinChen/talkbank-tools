# Go to Definition

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Press `F12` or `Cmd+Click` (macOS) / `Ctrl+Click` (Windows/Linux) on a speaker code or dependent tier item to jump to its definition or aligned source. The extension knows the alignment between main tier words, `%mor` items, and `%gra` relations, so it can navigate across tiers in a single keystroke.

## Speaker Definition

Clicking on a speaker code (e.g., `*CHI` on a main tier line, or `CHI` inside a `%gra` or `%mor` tier) jumps to its declaration in the `@Participants` header. This is the canonical location where the speaker's full name and role are defined.

## %mor to Main Tier

Clicking on a `%mor` item jumps to the aligned word on the main tier. For example, clicking `n|cookie` on the `%mor` tier jumps the cursor to the word `cookie` on the `*CHI:` line above.

The alignment is computed from the parsed model's positional correspondence between main tier words and morphological items, not from string position guessing.

> **(SCREENSHOT: Go-to-definition from %mor tier to main tier)**
> *Capture this: Cmd+Click on a `%mor` item like `n|cookie`. The cursor should jump to the corresponding word `cookie` on the main tier line above.*

## %gra to Main Tier

Clicking on a `%gra` relation jumps through the `%mor` tier to the aligned main tier word. The `%gra` tier indexes into `%mor` positions (e.g., `3|2|OBJ` means "word 3 depends on word 2"), so the extension follows the chain: `%gra` item -> `%mor` position -> main tier word.

This two-hop navigation is resolved transparently -- you click once and land on the main tier word.

## How It Works

The extension's go-to-definition handler inspects the cursor position to determine what kind of element you clicked:

1. **Speaker code** -- resolves to the `@Participants` header line via the speaker registry.
2. **`%mor` item** -- resolves to the main tier word at the same alignment index.
3. **`%gra` item** -- parses the source index from the `source|head|relation` triple, maps it through `%mor` alignment, and resolves to the main tier word.

All resolution uses the typed `ChatFile` AST produced by `talkbank-parser`, never raw text offsets.

## Related Chapters

- [Cross-Tier Alignment](alignment.md) -- hover tooltips showing the full alignment chain
- [Dependency Graphs](dependency-graphs.md) -- visual representation of `%gra` relations
- [Document Symbols](symbols.md) -- F2 rename and find-all-references for speaker codes
- [Syntax Highlighting](../editing/syntax-highlighting.md) -- visual distinction between tiers
