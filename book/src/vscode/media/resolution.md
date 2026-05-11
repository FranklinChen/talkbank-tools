# Media Resolution

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

This chapter explains how the extension locates media files referenced by CHAT
transcripts. Understanding the resolution process helps when media playback fails
or when organizing corpus files on disk.

## The @Media Header

Every `.cha` file that uses media features must include an `@Media:` header
specifying the base filename and the media type:

```text
@Media: interview, audio
@Media: session03, video
```

The header provides two pieces of information:

1. **Base filename** --- the name without extension (e.g., `interview`)
2. **Media type** --- either `audio` or `video`, which determines the file
   extensions the resolver tries

For `audio`, the resolver looks for: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`

For `video`, the resolver looks for: `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`

## Resolution Search Order

When you trigger playback or open the waveform, the extension's media resolver
(`mediaResolver.ts`) searches for the media file in the following order:

1. **Same directory** as the `.cha` file
2. **`media/` subdirectory** --- a sibling `media/` folder next to the `.cha` file
3. **Configured media roots** --- additional search paths from extension settings

At each location, the resolver tries every applicable file extension for the
declared media type. The first match wins.

### Example

Given a file at `/corpus/English/Brown/Adam/adam01.cha` with header
`@Media: adam01, audio`, the resolver searches:

```text
/corpus/English/Brown/Adam/adam01.mp3
/corpus/English/Brown/Adam/adam01.wav
/corpus/English/Brown/Adam/adam01.m4a
  ...
/corpus/English/Brown/Adam/media/adam01.mp3
/corpus/English/Brown/Adam/media/adam01.wav
/corpus/English/Brown/Adam/media/adam01.m4a
  ...
```

## Document Links

`@Media` file references are rendered as **clickable links** in the editor. If the
resolver finds the media file, clicking the link opens it directly in the system's
default media player (or in VS Code's built-in preview for supported formats).

This provides a quick way to verify that the media file exists and is accessible
without starting playback mode.

## Picture Display

The extension can display elicitation pictures (Cookie Theft, picture descriptions,
etc.) alongside the transcript --- the CLAN equivalent of PictController.

### Opening a Picture

1. Open a `.cha` file
2. Right-click in the editor and select **TalkBank: Media** then **Show Elicitation
   Picture**, or use the Command Palette

### How Pictures Are Found

The extension searches for associated images using three strategies, in order:

1. **`%pic:` tier references** --- if the document contains dependent tier lines like
   `%pic: "image002.jpg"`, those filenames are used directly

2. **Same-name image files** --- image files matching the `.cha` file's base name
   (e.g., `adam01.cha` matches `adam01.jpg`, `adam01.png`)

3. **Directory contents** --- any image files in the same directory as the `.cha` file

### Multi-Image Picker

If multiple candidate images are found, a picker dialog appears listing all matches.
Select the desired image to display it.

### Display

The selected picture opens in a side panel and scales to fit the available space.
You can arrange the picture panel alongside the transcript and waveform for a
complete annotation workspace.

## Troubleshooting

When media playback or the waveform panel fails to open, check the following:

### 1. Verify the @Media Header

The `.cha` file must contain a valid `@Media:` line. Check for:

- Missing header entirely --- add `@Media: filename, audio` (or `video`)
- Typos in the filename --- the base name must match the actual media file
- Wrong media type --- `audio` vs `video` must match the actual file format

### 2. Check That the File Exists

Verify that the media file is present in one of the resolver's search locations
(same directory or `media/` subdirectory). The filename must match the `@Media`
base name exactly (case-sensitive on macOS and Linux).

### 3. Check the File Extension

The media file must have a recognized extension for its type. An `.mp3` file
declared as `video` in the `@Media` header will not be found --- the resolver only
tries video extensions (`.mp4`, `.mov`, etc.) for video-typed headers.

### 4. Check the Output Panel

Open the **Output** panel in VS Code (`Cmd+Shift+U`) and select the **TalkBank**
channel. The extension logs media resolution attempts and failures here, showing
exactly which paths were searched and why resolution failed.

### 5. Large or Unsupported Formats

VS Code's embedded browser engine (Chromium) has limits on media format support.
If the file exists but playback fails, the format may not be supported. Common
working formats are MP3, WAV, and MP4 (H.264). Uncommon codecs (e.g., FLAC in
some configurations, AVI with legacy codecs) may not play.

## See Also

- [Media Playback](playback.md) --- playback controls and behavior
- [Waveform Visualization](waveform.md) --- visual audio waveform
- [Media Troubleshooting](../troubleshooting/media.md) --- extended troubleshooting guide
