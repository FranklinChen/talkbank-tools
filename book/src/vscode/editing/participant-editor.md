# Participant Editor

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

CHAT files encode participant metadata in `@ID` headers as pipe-delimited strings with 10 fields. Editing these by hand is tedious and error-prone -- it is easy to miscount pipes, leave a field blank, or use an invalid format. The Participant Editor provides a visual table interface for viewing and editing all participant information at once.

## Opening the Editor

There are two ways to open the Participant Editor:

- **Right-click** in a `.cha` file and select **Edit Participants...** from the context menu
- **Command Palette** (Cmd+Shift+P / Ctrl+Shift+P) and search for **TalkBank: Edit Participants**

A webview panel opens beside the editor showing all `@ID` lines in a tabular form.

> **(SCREENSHOT: Participant editor panel with filled fields)**
> *Capture this: open a .cha file that has at least two @ID headers (e.g., one for CHI and one for MOT). Open the participant editor via right-click. The panel should show a table with rows for each participant and columns for all 10 fields. Fill in some values (age, sex, role) so the table is not empty.*

## Fields

The table displays one row per participant with 10 editable columns, corresponding to the 10 pipe-delimited fields of the `@ID` header:

| Column | Description | Example Values |
|--------|-------------|----------------|
| **Language** | Three-letter language code | `eng`, `fra`, `zho` |
| **Corpus** | Corpus name | `MacWhinney`, `Brown`, `Sachs` |
| **Speaker Code** | Three-letter speaker code (matches `@Participants`) | `CHI`, `MOT`, `FAT`, `INV` |
| **Age** | Age in years;months.days format | `2;6.`, `3;0.15`, `4;` |
| **Sex** | Biological sex | `male`, `female` |
| **Group** | Experimental or demographic group | `typical`, `SLI`, `control` |
| **SES** | Socioeconomic status | `MC`, `WC`, `UC` |
| **Role** | Speaker's role | `Target_Child`, `Mother`, `Father`, `Investigator` |
| **Education** | Education level | `high_school`, `college`, `graduate` |
| **Custom** | Free-form field for project-specific data | Any text |

## Editing and Saving

1. Click any cell in the table to edit its value.
2. Modify as many fields as needed across any number of participants.
3. Click **Save Changes** to write the updated `@ID` lines back to the document.

When you save, the editor generates canonical `@ID` header lines and replaces the existing ones in the `.cha` file. The formatting is handled entirely by the language server -- the TypeScript UI layer sends the edited data to the LSP via `talkbank/formatIdLine`, and the server produces correctly formatted pipe-delimited output.

## Architecture

The Participant Editor is a thin UI layer over two LSP commands:

- **`talkbank/getParticipants`** -- parses all `@ID` headers in the current document and returns the structured field data.
- **`talkbank/formatIdLine`** -- takes edited field data and serializes it back to a canonical `@ID` header string.

No CHAT parsing happens in TypeScript. This ensures that the editor always produces correctly formatted output that matches the language server's understanding of the document.

## Tips

- **Adding a new participant**: Add the speaker to `@Participants` first (either manually or via the [E308 quick fix](quick-fixes.md)), then open the Participant Editor to fill in the `@ID` fields.
- **Speaker code renaming**: If you need to change a speaker code, use the [Rename](../navigation/go-to-definition.md) feature (F2) instead, which updates the code across `@Participants`, `@ID`, and all main tier lines in one operation.
- **Age format**: CHAT ages follow the format `years;months.days` -- for example, `2;6.` means 2 years and 6 months (days omitted). The editor does not enforce this format, so take care to use it correctly.

## Related Chapters

- [Code Completion & Snippets](completion.md) -- the `@ID` snippet provides a quick template for new ID headers
- [Quick Fixes](quick-fixes.md) -- E308 adds undeclared speakers to `@Participants`
- [Real-Time Validation](validation.md) -- validation checks participant consistency
