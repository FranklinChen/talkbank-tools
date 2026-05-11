# Walker Mode

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Step through utterances one segment at a time, playing each segment's audio
automatically. Walker mode is the primary tool for reviewing transcript alignment
--- verifying that each utterance's timing bullet matches the actual speech in the
audio.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt+Down` | Step to the **next** utterance |
| `Alt+Up` | Step to the **previous** utterance |

Each step moves the editor cursor to the corresponding utterance line. If media is
available and auto-play is enabled, the segment's audio plays immediately.

> **[VIDEO: 30s demo of walker stepping through utterances)**
> *Capture this: open a .cha file with media and the waveform panel. Press Alt+Down
> repeatedly, showing the cursor advancing through utterance lines while each
> segment's audio plays and the waveform highlights the active segment.*

## How It Works

1. The walker maintains a pointer to the current utterance index in the document
2. Pressing `Alt+Down` advances the pointer and moves the editor cursor to the
   next main tier line (a line starting with `*SPK:`)
3. If `talkbank.walker.autoPlay` is enabled, the utterance's bullet segment plays
   automatically
4. The [waveform panel](waveform.md) (if open) highlights the active segment
5. After playback completes, the walker waits for the next keypress (or pauses
   for `pauseSeconds` in continuous mode)

## Configuration

All walker settings live under the `talkbank.walker` namespace:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `talkbank.walker.autoPlay` | boolean | `true` | Automatically play the media segment when stepping to a new utterance |
| `talkbank.walker.loopCount` | integer | `1` | Number of times to loop each segment before allowing advance. `0` = loop indefinitely until manually advanced. |
| `talkbank.walker.pauseSeconds` | number | `0` | Seconds to pause between segments during continuous playback |
| `talkbank.walker.walkLength` | integer | `0` | Number of utterances to play continuously. `0` = all remaining utterances from the current position. |

### Example Configuration

To set up the walker for careful alignment review --- playing each segment twice
with a 1-second pause between utterances:

```json
{
  "talkbank.walker.autoPlay": true,
  "talkbank.walker.loopCount": 2,
  "talkbank.walker.pauseSeconds": 1
}
```

To step through utterances without audio (text-only review):

```json
{
  "talkbank.walker.autoPlay": false
}
```

## Document Change Reset

The walker resets its position when the active document changes. Switching to a
different `.cha` file or a non-CHAT document clears the walker state. When you
return to the original file, stepping resumes from wherever the cursor is positioned.

## Tips

- **Combine with the waveform panel** (`Cmd+Shift+W`) for visual feedback --- you
  can see the highlighted segment on the waveform as you step through utterances.

- **Use with Review Mode** for systematic post-alignment review. The
  [Post-Alignment Review](../workflows/post-alignment-review.md) workflow describes
  the full process of verifying that timing bullets match the actual speech.

- **Adjust loop count for difficult segments.** When reviewing a transcript with
  fast or unclear speech, set `loopCount` to 2 or 3 so each segment replays
  automatically before advancing.

- **Use `walkLength` for focused review.** If you only need to review the next 10
  utterances, set `walkLength` to 10. The walker stops after that many steps
  rather than continuing to the end of the file.

## See Also

- [Media Playback](playback.md) --- playback controls, speed, rewind, and loop
- [Waveform Visualization](waveform.md) --- visual waveform with segment highlighting
- [Transcription Mode](transcription.md) --- create new transcripts (walker is for reviewing existing ones)
- [Post-Alignment Review](../workflows/post-alignment-review.md) --- full review workflow
- [Settings Reference](../configuration/settings.md) --- all extension settings
