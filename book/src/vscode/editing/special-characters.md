# Special Characters

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

CHAT transcription uses many Unicode symbols that are not available on a standard keyboard -- overlap markers, prosody arrows, IPA-adjacent characters, and specialized punctuation. The extension provides a compose-key system for inserting these characters without memorizing Unicode codepoints or relying on system input methods.

## How Compose Mode Works

1. Press the compose shortcut to enter a mode:
   - **Cmd+Shift+1** (macOS) / **Ctrl+Shift+1** (Windows/Linux) for **CA mode** (Conversation Analysis symbols)
   - **Cmd+Shift+2** (macOS) / **Ctrl+Shift+2** (Windows/Linux) for **CHAT mode** (CHAT-specific symbols)
2. A status bar indicator appears showing the active compose mode (e.g., "CA Char..." or "CHAT Char...").
3. Press the trigger key corresponding to the symbol you want to insert.
4. The Unicode symbol is inserted at the cursor position, and compose mode deactivates.
5. Press **Escape** at any time to cancel without inserting anything.

> **(SCREENSHOT: Status bar showing compose mode active)**
> *Capture this: press Cmd+Shift+1 in a .cha file. The status bar at the bottom of the VS Code window should show a "CA Char..." indicator. Capture the status bar region.*

This compose-key system is ported from the CLAN macOS application's `CharToSpChar()` function, so researchers familiar with CLAN's character input will find the same key mappings.

## CA Mode Characters (Cmd+Shift+1)

Conversation Analysis notation uses specialized Unicode symbols for prosody, overlap, voice quality, and other phonetic features. After pressing Cmd+Shift+1, press one of the following keys:

| Key | Symbol | Name | Meaning |
|-----|--------|------|---------|
| `1` | ⇗ | North-east arrow | Rise to high |
| `2` | ↗ | Up-right arrow | Rise to mid |
| `3` | → | Rightward arrow | Level pitch |
| `4` | ↘ | Down-right arrow | Fall to mid |
| `5` | ⇘ | South-east arrow | Fall to low |
| `[` | ⌈ | Left ceiling | Overlap start (first speaker) |
| `]` | ⌉ | Right ceiling | Overlap end (first speaker) |
| `{` | ⌊ | Left floor | Overlap start (second speaker) |
| `}` | ⌋ | Right floor | Overlap end (second speaker) |
| `.` | ∙ | Bullet operator | Inhalation |
| `=` | ≈ | Almost equal | Latching (no gap between turns) |
| `0` | ° | Degree sign | Softer / quieter speech |
| `)` | ◉ | Fisheye | Louder speech |
| `w` | ∬ | Double integral | Whisper |
| `s` | ∮ | Contour integral | Singing |
| `b` | ♋ | Cancer sign | Breathy voice |
| `*` | ⁎ | Low asterisk | Creaky voice |
| `/` | ⁇ | Double question mark | Unsure / uncertain transcription |

### Overlap Markers

The four overlap bracket symbols (⌈ ⌉ ⌊ ⌋) mark the boundaries of overlapping speech between two speakers. In CA transcription:

- Speaker A's overlap is marked with ceiling brackets: ⌈overlapping words⌉
- Speaker B's overlap is marked with floor brackets: ⌊overlapping words⌋

### Prosody Contours

The five numbered keys (1-5) map to pitch contour arrows arranged from highest to lowest, matching standard CA prosody notation conventions.

## CHAT Mode Characters (Cmd+Shift+2)

CHAT format uses additional special characters for phonological notation, grouping, and morphological analysis. After pressing Cmd+Shift+2, press one of the following keys:

| Key | Symbol | Name | Meaning |
|-----|--------|------|---------|
| `q` | ʔ | Glottal stop | IPA glottal stop |
| `Q` | ʕ | Pharyngeal fricative | Hebrew glottal (pharyngeal) |
| `:` | ː | Triangular colon | Long vowel (IPA length mark) |
| `H` | ʰ | Superscript h | Aspiration marker |
| `<` | ‹ | Single left angle quote | Group start |
| `>` | › | Single right angle quote | Group end |
| `{` | 〔 | Left tortoise shell bracket | Sign group start |
| `}` | 〕 | Right tortoise shell bracket | Sign group end |
| `1` | ˈ | Modifier letter vertical line | Primary stress |
| `2` | ˌ | Modifier letter low vertical line | Secondary stress |
| `m` | ... | Horizontal ellipsis | Missing word (used on %pho tier) |
| `/` | ↫ | Leftward arrow with loop | Left arrow with circle |
| `=` | ≠ | Not equal | Crossed equal sign |

## Tips

- **One character at a time**: Compose mode deactivates after each character insertion. To insert multiple special characters in sequence, press the compose shortcut again before each one.
- **Mode confusion**: If you are unsure which mode you are in, check the status bar indicator. Press Escape to cancel and start over.
- **System input methods**: On macOS, you can also use the built-in Character Viewer (Ctrl+Cmd+Space) as an alternative for occasional use. The compose-key system is faster for frequent transcription work.

## Related Chapters

- [Syntax Highlighting](syntax-highlighting.md) -- how special characters are colored in the editor
- [Code Completion & Snippets](completion.md) -- bracket annotations that contain some of these symbols
- [Quick Fixes](quick-fixes.md) -- automatic corrections for common CHAT errors
