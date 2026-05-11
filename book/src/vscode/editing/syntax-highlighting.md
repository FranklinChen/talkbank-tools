# Syntax Highlighting

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

The extension provides two layers of syntax highlighting for CHAT files: a TextMate grammar for immediate fallback coloring, and LSP semantic tokens for context-aware precision. Together they make CHAT transcripts readable at a glance, with distinct colors for headers, speaker codes, annotations, morphology, and other structural elements.

## Two Highlighting Layers

### TextMate Grammar (Fallback)

The TextMate grammar (`syntaxes/chat.tmLanguage.json`) provides basic token coloring that activates instantly when you open a `.cha` file, before the language server has finished starting. It uses pattern-matching rules to identify CHAT constructs by their textual form.

### LSP Semantic Tokens (Context-Aware)

Once the language server is running, it provides semantic tokens that override the TextMate grammar with more precise coloring based on the parsed AST. For example, the semantic layer can distinguish specific POS tags in morphology, identify error-coded tokens, and differentiate between header types that look similar as raw text.

The language server advertises 11 semantic token types:

| Index | Token Type | Used For |
|-------|------------|----------|
| 0 | `keyword` | Headers (`@Begin`, `@End`, `@UTF8`), tier prefixes |
| 1 | `variable` | Speaker codes (`*CHI`, `*MOT`, `*INV`) |
| 2 | `string` | Quoted strings, word content |
| 3 | `comment` | Comment lines (`%com:`) |
| 4 | `type` | Type annotations, complex structures |
| 5 | `operator` | Postcodes, morphological separators (`\|`, `-`, `+`, `&`) |
| 6 | `number` | Timing values, indices |
| 7 | `function` | Dependent tier prefixes, special markers |
| 8 | `tag` | Tier labels, annotation markers |
| 9 | `punctuation` | Terminators (`.` `?` `!`), special punctuation |
| 10 | `error` | Syntax errors, malformed tokens |

Range-based semantic tokens are supported, computing tokens only for the visible range. This reduces work for large files.

## Color Categories

The following table summarizes how CHAT elements are colored in the editor. Exact colors depend on your VS Code color theme, but the semantic categories are consistent:

| Element | Examples | Color Category |
|---------|----------|----------------|
| Required headers | `@UTF8`, `@Begin`, `@End` | Blue / bold keyword |
| Metadata headers | `@Participants`, `@ID`, `@Languages` | Blue / keyword |
| General headers | `@Date`, `@Location`, custom headers | Blue / keyword |
| Speaker codes | `*CHI:`, `*MOT:`, `*INV:` | Bold / variable color (distinct from headers) |
| Dependent tier prefixes | `%mor:`, `%gra:`, `%pho:` | Italic / function color |
| Utterance terminators | `.` `?` `!` `+/.` `+...` | Red / punctuation |
| Annotations | `[= text]`, `[: alt]`, `[*]`, `[+ post]` | Green / string-operator |
| Scoped groups | `<word word>` | Bracket highlighting |
| Retrace markers | `[/]`, `[//]`, `[//?]` | Operator color |
| Actions and events | `&=laughs`, `&Claps` | Cyan / tag |
| Pauses | `(0.5)`, `(.)` | Gray / number |
| Omissions | `0word` | Special styling |
| Morphology | `n\|cookie`, `v\|go-PROG` | POS / stem / affix coloring |
| Grammar relations | `1\|2\|DET`, `3\|0\|ROOT` | Index / relation coloring |
| Comments | `%com:` lines | Comment color (typically gray-green) |

## On-Type Formatting

When you type `:` on a `*SPEAKER:` or `%tier:` line, a tab character is automatically inserted after the colon. This matches the CHAT format convention where a tab always follows the colon on speaker and tier lines. No manual tab insertion is needed.

This behavior is provided by the language server's `textDocument/onTypeFormatting` handler and works in real time as you type.

## Document Formatting

Format the entire document to canonical CHAT style with **Shift+Alt+F** (or your configured format shortcut), or via the Command Palette (**Format Document**).

The formatter normalizes:

- Whitespace and indentation
- Header ordering
- Speaker code formatting
- Tier indentation (tabs after colons)
- Line endings

If the document is already in canonical form, no changes are applied. You can also enable **Format on Save** via VS Code's `editor.formatOnSave` setting to automatically normalize the file every time you save.

The formatting operation is backed by the language server's `textDocument/formatting` handler, which serializes the document through the canonical CHAT serializer. This ensures the output matches the authoritative CHAT format exactly.

## Related Chapters

- [Real-Time Validation](validation.md) -- error squiggles work alongside highlighting
- [Code Completion & Snippets](completion.md) -- completions appear in the context of highlighted elements
- [Quick Fixes](quick-fixes.md) -- fix errors that are highlighted as diagnostics
- [Cross-Tier Alignment](../navigation/alignment.md) -- clicking highlighted words shows cross-tier connections
