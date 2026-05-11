# Speaker Filtering

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

In multi-party conversations, it is often useful to focus on a single speaker's contributions. The speaker filtering feature creates a read-only filtered view of the transcript showing only the utterances from selected speakers.

## How to Use

1. Open a `.cha` file.
2. Invoke the command using any of these methods:
   - **Context menu:** right-click in the editor, then select **Filter by Speaker...**
   - **Command Palette:** `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows/Linux), then type **TalkBank: Filter by Speaker**
3. A multi-select picker appears listing all speakers defined in the file's `@Participants` header.
4. Select one or more speakers and confirm.
5. A filtered read-only document opens in a side-by-side panel.

## What the Filtered View Shows

The filtered document includes:

- **All file headers** -- `@UTF8`, `@Begin`, `@Participants`, `@ID`, `@Languages`, `@Media`, and all other `@` headers are preserved so the document remains structurally valid.
- **Selected speakers' utterance blocks** -- each main tier line for the selected speakers, along with all their dependent tiers (`%mor`, `%gra`, `%pho`, etc.).
- **No other speakers** -- utterance blocks from non-selected speakers are omitted entirely.

The filtered view is a **virtual document** -- it does not modify the original file on disk. The language server still processes it, so you get full validation, highlighting, and hover information even in the filtered view.

## Use Cases

- **Focus on child speech:** filter to `CHI` only to review a child's productions without the surrounding adult turns.
- **Examine interviewer questions:** filter to `INV` to see just the interviewer's prompts.
- **Compare two speakers:** select both `CHI` and `MOT` to see the mother-child dyad without other participants.
- **Quick utterance review:** combine with the [Document Symbols](symbols.md) outline to jump between the filtered speaker's utterances.

## Related Chapters

- [Document Symbols](symbols.md) -- rename, find references, and code lens for speaker codes
- [Scoped Find](scoped-find.md) -- search within specific tiers or speakers without creating a filtered view
- [Cross-Tier Alignment](alignment.md) -- hover and highlighting work in the filtered view
- [CLAN Analysis Commands](../analysis/command-reference.md) -- many CLAN commands accept `--speaker` flags for per-speaker analysis
