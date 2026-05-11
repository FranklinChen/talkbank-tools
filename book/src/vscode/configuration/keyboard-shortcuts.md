# Keyboard Shortcuts

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

This chapter lists all keyboard shortcuts provided by the TalkBank CHAT extension, grouped by feature area. All shortcuts are active only when a `.cha` file is open in the editor (`editorLangId == chat`), unless otherwise noted.

## Media Playback

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Play Media at Cursor | `Cmd+Shift+Enter` | `Ctrl+Shift+Enter` | CHAT file open |
| Play Media Continuously | `Cmd+Shift+/` | `Ctrl+Shift+/` | CHAT file open |
| Rewind Media (2 s) | `F8` | `F8` | CHAT file open |
| Toggle Segment Loop | `Shift+F5` | `Shift+F5` | CHAT file open |
| Show Waveform View | `Cmd+Shift+W` | `Ctrl+Shift+W` | CHAT file open |

## Walker Mode

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Walker: Next Utterance | `Alt+Down` | `Alt+Down` | CHAT file open |
| Walker: Previous Utterance | `Alt+Up` | `Alt+Up` | CHAT file open |

## Transcription Mode

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Stamp Timestamp Bullet | `F4` | `F4` | Transcription mode active |

The `F4` key is only active when transcription mode has been started via **TalkBank: Start Transcription Mode** (the `talkbank.transcriptionActive` context is set).

## Coder Mode

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Coder: Next Utterance | `Cmd+Enter` | `Ctrl+Enter` | Coder mode active |
| Coder: Insert Code | `Cmd+Shift+C` | `Ctrl+Shift+C` | Coder mode active |

These keys are only active when coder mode has been started via **TalkBank: Start Coder Mode** (the `talkbank.coderActive` context is set).

## Review Mode

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Review: Next Flagged | `Alt+]` | `Alt+]` | Review mode active |
| Review: Previous Flagged | `Alt+[` | `Alt+[` | Review mode active |
| Review: Rate Good | `1` | `1` | Review mode active, not editing text |
| Review: Rate Early | `2` | `2` | Review mode active, not editing text |
| Review: Rate Late | `3` | `3` | Review mode active, not editing text |
| Review: Rate Wrong | `4` | `4` | Review mode active, not editing text |
| Review: Skip | `5` | `5` | Review mode active, not editing text |

The number keys (`1`-`5`) for rating are active only when the editor does **not** have text focus (`!editorTextFocus`). This prevents accidental ratings while typing corrections. Click outside the text area (e.g., on the editor margin or the media panel) to enable the rating keys.

## Navigation and Analysis

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Show Dependency Graph | `Cmd+Shift+G` | `Ctrl+Shift+G` | CHAT file open |

## Special Character Input

| Action | macOS | Windows/Linux | When Active |
|--------|-------|---------------|-------------|
| Insert CA Special Character | `Cmd+Shift+1` | `Ctrl+Shift+1` | Editor focused, CHAT file |
| Insert CHAT Special Character | `Cmd+Shift+2` | `Ctrl+Shift+2` | Editor focused, CHAT file |

## Standard VS Code Shortcuts (CHAT-Aware)

These are standard VS Code shortcuts that gain CHAT-specific behavior through the extension's LSP:

| Action | macOS | Windows/Linux | Description |
|--------|-------|---------------|-------------|
| Quick Fix | `Cmd+.` | `Ctrl+.` | Show available code actions (fixes for 21 error codes) |
| Rename | `F2` | `F2` | Rename speaker code across the entire file |
| Find All References | `Shift+F12` | `Shift+F12` | Find all occurrences of a speaker code |
| Go to Definition | `F12` or `Cmd+Click` | `F12` or `Ctrl+Click` | Jump to speaker declaration or aligned tier |
| Document Symbols | `Cmd+Shift+O` | `Ctrl+Shift+O` | Navigate utterances by speaker in the outline |
| Workspace Symbols | `Cmd+T` | `Ctrl+T` | Search across all open CHAT files |
| Format Document | `Shift+Alt+F` | `Shift+Alt+F` | Reformat to canonical CHAT form |
| Open Problems | `Cmd+Shift+M` | `Ctrl+Shift+M` | Show all diagnostics in the Problems panel |

## Customizing Keyboard Shortcuts

To change any keybinding:

1. Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Select **Preferences: Open Keyboard Shortcuts**
3. Search for "talkbank" to see all extension keybindings
4. Click the pencil icon next to any shortcut to assign a new key combination

### Transcription Keybinding Configuration

The extension provides a dedicated command for configuring transcription keys:

1. Open the Command Palette
2. Select **TalkBank: Configure Transcription Keybindings**
3. The keybindings editor opens, pre-filtered to show only TalkBank transcription commands (stamp, rewind, loop)

This is the recommended way to remap keys for foot pedals or other input devices.

### Keybindings in keybindings.json

For manual configuration, add entries to your `keybindings.json`:

```json
[
  {
    "key": "f9",
    "command": "talkbank.stampBullet",
    "when": "editorLangId == chat && talkbank.transcriptionActive"
  },
  {
    "key": "f10",
    "command": "talkbank.rewindMedia",
    "when": "editorLangId == chat"
  }
]
```

The `when` clause ensures the keybinding is only active in the appropriate context.

## Related Chapters

- [Settings Reference](settings.md) -- extension settings
- [Transcription Workflow](../workflows/transcription.md) -- transcription mode details
- [Coding Workflow](../coder/workflow.md) -- coder mode details
- [Post-Alignment Review](../workflows/post-alignment-review.md) -- review mode details
