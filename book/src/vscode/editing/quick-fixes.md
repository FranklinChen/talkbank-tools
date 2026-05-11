# Quick Fixes

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

When the language server detects certain errors, it can offer automatic corrections. These appear as a lightbulb icon in the left gutter, or you can invoke them directly with **Cmd+.** (macOS) / **Ctrl+.** (Windows/Linux). Quick fixes are the fastest way to resolve common CHAT formatting and structural problems without manual editing.

## How Quick Fixes Work

1. The language server continuously validates the document as you type (see [Real-Time Validation](validation.md)).
2. For errors that have a well-defined automatic correction, a code action is registered.
3. When your cursor is on a diagnostic that has a quick fix, a lightbulb icon appears in the gutter.
4. Click the lightbulb or press **Cmd+.** to see the available fixes.
5. Select a fix to apply it instantly.

Some errors offer multiple fix options (for example, E301 offers three different terminators). Others apply a single unambiguous correction.

> **(SCREENSHOT: Quick fix lightbulb showing options for E308 (undeclared speaker))**
> *Capture this: open a .cha file with a speaker code on a main tier that is not listed in @Participants. Place cursor on the error squiggle, then press Cmd+. to show the lightbulb menu with the "Add 'INV' to @Participants" option.*

## Available Quick Fixes

The extension currently provides automatic fixes for 21 error codes:

| Error Code | Description | Quick Fix Action |
|------------|-------------|------------------|
| **E241** | Illegal untranscribed marker (`xx`) | Replace `xx` with `xxx` (the correct CHAT untranscribed speech marker) |
| **E242** | Incomplete word | Append the trailing-off marker (`+...`) to the word |
| **E244** | Consecutive stress markers | Remove the duplicate separator |
| **E258** | Consecutive commas | Clean up consecutive comma punctuation |
| **E259** | Form error | Correct the malformed expression |
| **E301** | Missing terminator | Insert a terminator: offers `.` (declarative), `?` (question), or `!` (exclamation) |
| **E305** | Missing period | Insert the missing period |
| **E306** | Empty utterance | Delete the empty utterance line (shown with fade-out styling) |
| **E308** | Undeclared speaker | Add the speaker code to the `@Participants` header |
| **E312** | Bare group (unclosed bracket) | Insert the missing closing bracket |
| **E313** | Bare annotation (unclosed paren) | Insert the missing closing parenthesis |
| **E322** | Empty colon | Delete the empty colon line (shown with fade-out styling) |
| **E323** | Missing colon after speaker | Insert the missing colon and tab after the speaker code |
| **E362** | Timestamp swap | Swap the reversed start and end timestamps in the timing bullet |
| **E501** | Missing `@Begin` | Insert `@Begin` after the `@UTF8` header line |
| **E502** | Missing `@End` | Insert `@End` at the end of the file |
| **E503** | Missing `@UTF8` | Insert `@UTF8` at the very start of the file |
| **E504** | Missing `@Languages` | Insert an empty `@Languages:` header template |
| **E506** | Missing `@Participants` | Insert an empty `@Participants:` header template |
| **E507** | Participant format error | Correct the participant entry format |
| **E604** | Unknown/orphaned tier | Remove or correct the orphaned dependent tier |

## Diagnostic Tags

Some quick-fixable errors use VS Code's fade-out styling to visually indicate that content can be safely removed. Empty utterances (E306) and empty colons (E322) appear with dimmed text in the editor, making them easy to spot as candidates for deletion.

## Tips

- **Batch fixing**: When a file has many instances of the same error (for example, multiple undeclared speakers), you can fix them one at a time with Cmd+. or use the Problems panel (**Cmd+Shift+M**) to navigate between them.
- **Undo**: Every quick fix is a normal text edit, so **Cmd+Z** reverts it if the fix was not what you intended.
- **Format after fixing**: Some fixes (like inserting a missing `@Begin`) may leave whitespace that is not perfectly canonical. Use [Document Formatting](syntax-highlighting.md#document-formatting) (**Shift+Alt+F**) to normalize the file after applying fixes.

## Related Chapters

- [Real-Time Validation](validation.md) -- how diagnostics are produced
- [Syntax Highlighting](syntax-highlighting.md) -- visual indicators for CHAT elements
- [Code Completion & Snippets](completion.md) -- proactive insertion of correct CHAT structures
