# Transcription Mode

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Create new CHAT transcripts from audio by typing utterance text while the audio
plays, then stamping timing bullets at utterance boundaries. Transcription mode
turns VS Code into a transcription workstation --- the CLAN equivalent of
Transcribe mode.

## Prerequisites

The `.cha` file must contain an `@Media:` header pointing to an audio or video
file. The media file must be findable by the extension's resolver (see
[Media Resolution](resolution.md)). A minimal starting file looks like:

```text
@UTF8
@Begin
@Languages: eng
@Participants: CHI Target_Child
@Media: interview, audio
*CHI:
@End
```

## Starting Transcription Mode

1. Open the `.cha` file
2. Open the Command Palette (`Cmd+Shift+P`)
3. Search for **TalkBank: Start Transcription Mode**
4. Audio begins playing from the start of the file

The waveform panel opens automatically to provide visual reference.

## Workflow

The core transcription loop is:

1. **Listen** to the audio as it plays
2. **Type** the speaker's utterance text on the current `*SPK:` line
3. Press **`F4`** to stamp a timing bullet and advance

When you press `F4`:

- A timing bullet is inserted at the end of the current utterance line, marking the
  time span from the previous stamp (or file start) to the current playback position:
  `go home .`
- The cursor automatically advances to a new `*SPK:` line, ready for the next
  utterance
- The speaker code on the new line uses the configured default speaker

4. **Repeat** until the recording is complete
5. Open the Command Palette and search for **TalkBank: Stop Transcription Mode**

> **[VIDEO: 60s demo of transcription workflow with F4 stamping)**
> *Capture this: start transcription mode on a short audio file. Type 3-4 utterances,
> pressing F4 after each one. Show the timing bullets being inserted and new speaker
> lines being created. Include the waveform panel in the recording.*

## Key Commands During Transcription

| Shortcut | Command | Description |
|----------|---------|-------------|
| `F4` | Stamp timing bullet | Insert bullet with current time range, advance to next line |
| `F8` | Rewind | Jump back by `rewindSeconds` to re-listen |
| `F5` | Loop | Toggle looping of the current playback region |

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `talkbank.transcription.defaultSpeaker` | string | `CHI` | Speaker code inserted on new utterance lines (e.g., `CHI`, `MOT`, `INV`) |
| `talkbank.transcription.rewindSeconds` | number | `2` | Seconds to rewind when pressing F8. Minimum: 0.5 |

### Example Configuration

For an investigator transcribing an adult interview:

```json
{
  "talkbank.transcription.defaultSpeaker": "INV",
  "talkbank.transcription.rewindSeconds": 3
}
```

## Foot-Pedal and Custom Keybindings

USB foot pedals are commonly used in transcription to control playback without
taking hands off the keyboard. To remap transcription keys for a foot pedal:

1. Open the Command Palette (`Cmd+Shift+P`)
2. Search for **TalkBank: Configure Transcription Keybindings**
3. The VS Code keybindings editor opens, pre-filtered to TalkBank transcription
   commands
4. Click the pencil icon next to any command to assign a new keybinding

This is the recommended way to map foot-pedal buttons to `F4` (stamp), `F8`
(rewind), and `F5` (loop). Most USB foot pedals present as standard keyboard
devices and can be mapped to any VS Code keybinding.

See [Keyboard Shortcuts](../configuration/keyboard-shortcuts.md) for more on
customizing keybindings.

## Tips

- **Use the waveform panel** for visual reference. The waveform shows where you
  are in the recording and helps you identify utterance boundaries by looking at
  silence gaps between speech regions.

- **Rewind generously.** The default 2-second rewind (`F8`) lets you re-hear the
  end of an utterance. For fast or overlapping speech, increase `rewindSeconds`
  to 3 or 4.

- **Slow down playback** using the speed slider on the media panel. Dropping to
  0.5x or 0.75x gives you more time to type without pausing.

- **Edit timing after transcription.** Transcription mode creates an initial
  pass. Use [Walker Mode](walker.md) afterward to review each segment and adjust
  timing bullets as needed.

## See Also

- [Media Playback](playback.md) --- playback controls, speed, and rewind
- [Waveform Visualization](waveform.md) --- visual waveform for boundary identification
- [Walker Mode](walker.md) --- review and adjust alignment after transcription
- [Media Resolution](resolution.md) --- how the extension finds media files
- [Transcription Workflow](../workflows/transcription.md) --- end-to-end transcription process
- [Keyboard Shortcuts](../configuration/keyboard-shortcuts.md) --- customize keybindings
