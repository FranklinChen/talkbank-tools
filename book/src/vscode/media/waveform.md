# Waveform Visualization

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Visualize the audio waveform of the linked media file with timed utterance overlays.
The waveform panel uses the Web Audio API to decode and render audio amplitude data,
with colored segments marking each speaker's bullet ranges.

## Opening the Waveform

**Shortcut:** `Cmd+Shift+W` (macOS) / `Ctrl+Shift+W` (Windows/Linux)

1. Open a `.cha` file with an `@Media:` header
2. Press `Cmd+Shift+W`
3. A waveform panel opens showing the audio amplitude as a continuous waveform trace

The waveform panel is a singleton --- pressing the shortcut again when the panel is
already open brings it to focus rather than creating a duplicate.

> **(SCREENSHOT: Waveform panel with colored bullet segments)**
> *Capture this: open a multi-speaker .cha file with media, press Cmd+Shift+W, and
> screenshot the waveform panel showing colored segment overlays for different
> speakers. Ensure at least two speaker colors are visible.*

## Colored Segment Overlays

Each bullet segment is drawn as a colored overlay on the waveform. Speakers are
assigned distinct colors, so you can visually distinguish which parts of the audio
belong to which participant. Overlapping bullets from different speakers appear as
stacked colored bars (see [Playback --- Overlapping Bullets](playback.md#overlapping-bullets-cross-speaker-overlap)).

## Interaction

### Click to Seek

Click anywhere on the waveform to seek to that time position. When you click:

- The media panel (if open) seeks to the clicked time
- The editor cursor moves to the utterance line nearest to that time
- If playback is active, it continues from the new position

This makes it easy to jump to a specific point in the audio by visual inspection
of the waveform shape.

### Coordination with Media Panel

The waveform and media panels are fully coordinated:

- Playing a segment highlights it in the waveform
- Seeking in the waveform updates the media panel position
- The currently-playing segment is visually distinguished (brighter overlay or
  outline) so you can track playback progress

## Zoom

The waveform panel provides several zoom controls:

| Control | Action |
|---------|--------|
| **Zoom in/out buttons** | Toolbar buttons for incremental zoom steps |
| **Zoom slider** | Continuous zoom from 100% to 2000% |
| **Fit button** | Reset zoom to 100% (fit entire waveform in view) |
| **Mouse wheel** | Scroll wheel zooms in/out, centered on the pointer position |

### Pointer-Centered Zoom

Mouse wheel zooming is centered on the pointer position. This means the point under
your cursor stays fixed as you zoom in, allowing precise navigation to a specific
region of the audio. This is particularly useful when examining a short segment
within a long recording.

## Scroll

When zoomed in beyond the panel width, the waveform scrolls horizontally:

- **Manual scroll:** Use the horizontal scrollbar or shift+scroll wheel
- **Auto-scroll during playback:** When playback is active, the view automatically
  scrolls to keep the current segment visible. The scroll follows the playback
  position so you never lose sight of what is currently playing.

## Typical Workflow

1. Open a `.cha` file and press `Cmd+Shift+W` to open the waveform
2. Press `Cmd+Shift+Enter` to open the media panel
3. Click on the waveform to jump to a region of interest
4. Zoom in with the scroll wheel for fine-grained inspection
5. Use [Walker Mode](walker.md) (`Alt+Down` / `Alt+Up`) to step through
   utterances while watching the waveform highlight each segment

## See Also

- [Media Playback](playback.md) --- audio/video playback controls and behavior
- [Walker Mode](walker.md) --- step through utterances with synchronized waveform
- [Transcription Mode](transcription.md) --- use the waveform for visual reference while transcribing
- [Media Resolution](resolution.md) --- how the extension finds the audio file
