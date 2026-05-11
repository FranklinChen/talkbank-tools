# Document Symbols

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

The extension provides standard VS Code symbol navigation features -- document outline, workspace search, rename, find references, linked editing, and code lens -- all aware of CHAT structure. These make it easy to navigate large transcript files and perform safe, file-wide speaker operations.

## Document Symbols (Outline)

Press `Cmd+Shift+O` (macOS) / `Ctrl+Shift+O` (Windows/Linux) to open the document symbol picker. The outline shows:

- **`@` headers** -- `@Begin`, `@Participants`, `@ID`, `@Languages`, `@Media`, and all other headers
- **Speaker lines** -- every `*SPK:` main tier line, grouped under the speaker code

This is the same data that populates the **Outline** view in the Explorer sidebar. Use it to jump directly to any header or speaker utterance.

> **(SCREENSHOT: Document outline showing headers and speaker lines in the Outline sidebar view)**
> *Capture this: open a `.cha` file with multiple speakers and several headers. Show the Outline view in the Explorer sidebar with the hierarchical symbol tree.*

## Workspace Symbols

Use `Cmd+T` (macOS) / `Ctrl+T` (Windows/Linux) to search across all open CHAT files for headers and speaker lines. Type a speaker code like `CHI` or a header name like `@Media` to jump to any matching location across the entire workspace, without switching files manually.

This is especially useful when working with a corpus of related transcripts and you need to find a specific speaker or header across multiple files.

## Rename Speaker (F2)

Place your cursor on a speaker code -- either on a `*CHI:` main tier line or in the `@Participants` header -- and press `F2`. Type the new code and all occurrences are updated atomically:

- The speaker entry in `@Participants`
- The corresponding `@ID` header line
- All main tier lines using that speaker code (`*CHI:` becomes `*NEW:`)

This is a standard LSP rename operation. It also works via right-click **Rename Symbol** or the Command Palette (**TalkBank: Rename Symbol**).

> **(SCREENSHOT: Rename dialog on a speaker code showing the preview of all affected locations)**
> *Capture this: place cursor on `*CHI` and press F2. Type a new code like `TAR` and show the rename preview with all locations that will change.*

## Find All References (Shift+F12)

Place your cursor on a speaker code and press `Shift+F12` (or right-click **Find All References**). The References panel shows every location where that speaker appears:

- `@Participants` declaration
- `@ID` header line
- All main tier lines for that speaker

This helps you quickly see the full extent of a speaker's contributions and verify that all references are consistent.

## Linked Editing

When you edit a speaker code, all other occurrences of that speaker in the document are highlighted for simultaneous editing. As you type, every matching instance updates in real time.

This is different from F2 Rename: linked editing is live, in-place, character-by-character editing of all matching speaker codes. F2 Rename is a one-shot atomic operation with a preview dialog.

Enable linked editing via VS Code's `editor.linkedEditing` setting:

```json
{
  "editor.linkedEditing": true
}
```

## Code Lens

Above the `@Participants` header, a code lens annotation shows the utterance count for each speaker (e.g., `CHI: 42 utterances | MOT: 38 utterances`). This provides an at-a-glance summary of speaker activity without running an analysis command.

The counts update automatically as you edit the file.

## Related Chapters

- [Go to Definition](go-to-definition.md) -- jump from a speaker reference to its `@Participants` declaration
- [Speaker Filtering](speaker-filtering.md) -- view only selected speakers' utterances
- [Cross-Tier Alignment](alignment.md) -- hover and highlighting for tier-level navigation
- [Keyboard Shortcuts](../configuration/keyboard-shortcuts.md) -- customize symbol navigation bindings
