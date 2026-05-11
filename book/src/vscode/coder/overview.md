# Coder Mode Overview

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Coder Mode is a structured annotation workflow for coding CHAT transcripts with a predefined coding scheme. It is the VS Code equivalent of CLAN's coder mode (implemented in `ced_codes.cpp`), adapted to work with VS Code's QuickPick interface and the LSP-backed document model.

## What Is Coding?

In language research, "coding" means annotating each utterance in a transcript with one or more labels from a controlled vocabulary. These labels (codes) capture information that is not directly present in the transcript text -- for example, the pragmatic function of an utterance (request, demand, acknowledgment), the type of error a speaker made, or the conversational strategy being employed.

Codes are stored on dependent tiers, one per utterance:

```text
*CHI:	I want that one .
%cod:	$PRA:request
*MOT:	which one ?
%cod:	$PRA:question
```

The coding scheme is defined in a `.cut` file -- a tab-indented hierarchy of valid codes. Different research projects use different coding schemes depending on what they are studying.

## Why Use Coder Mode?

Without Coder Mode, coding a transcript is tedious and error-prone: the coder must remember the valid codes, type them correctly on dependent tiers, and manually navigate to the next uncoded utterance. Coder Mode automates all of this:

- **Code selection from a picker** -- no memorization or typos. The hierarchical QuickPick shows all valid codes from the `.cut` file.
- **Automatic navigation** -- after inserting a code, the cursor advances to the next uncoded utterance. No manual searching.
- **Progress tracking** -- the picker shows how many utterances remain uncoded.
- **Consistent results** -- every code comes from the controlled vocabulary, ensuring inter-rater reliability.

## CLAN Equivalent

In CLAN, coder mode is activated from the Commands Window and uses `ced_codes.cpp` to drive a custom modal editor. The coder navigates with keyboard shortcuts (`ESC-e` to toggle modes, `ESC-c` to move between code levels, `CTRL-T` to advance speakers) and selects codes from a hierarchical menu.

The VS Code implementation replaces this with standard VS Code UI patterns:

| CLAN | VS Code |
|------|---------|
| `ESC-e` toggle editing mode | Command Palette: Start/Stop Coder Mode |
| `ESC-c` code level navigation | Hierarchical QuickPick with indentation |
| `CTRL-T` next speaker | `Cmd+Enter` next uncoded utterance |
| Custom modal menu | VS Code QuickPick with fuzzy search |

## Available Standard Code Files

TalkBank distributes several standard `.cut` code files in the `clan-info/lib/coder/` directory:

| File | Target Tier | Purpose |
|------|-------------|---------|
| `codes-basic.cut` | `%spa` | Basic speech act coding (positive/negative, questions/responses) for mother-child interaction |
| `codes1.cut` | `%spa` | Extended speech act coding with question types (NV/VE), answers, comments, acknowledgments |
| `codeserr.cut` | `%err` | Error coding for lexical errors (incomplete, uncertain, accented, added, lost, etc.) |
| `codeshar.cut` | `%spa` | Conversational strategy coding (NIA categories: acknowledgment, adaptation, clarification, etc.) |

These files serve as examples and starting points. Most research projects create custom `.cut` files tailored to their specific coding scheme. See [Codes Files](codes-files.md) for the file format and how to create your own.

## Related Chapters

- [Codes Files (.cut)](codes-files.md) -- file format, structure, and how to create custom code files
- [Coding Workflow](workflow.md) -- step-by-step usage of Coder Mode in VS Code
