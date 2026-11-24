# File Name Matching

**Status:** Current
**Last updated:** 2026-11-17 21:32 EST

batchalign3 finds a transcript's recording by name. Commands that read audio
for an existing transcript (`align`, speaker identification, remote staging)
take the transcript's name, look in each place a recording may be, and use
`<name>.<extension>`. This page is about getting that name right. The
places searched, in order, are on
[Server Mode](../user-guide/server-mode.md#how-the-server-finds-a-recording)
and in [Media Conversion](media-conversion.md#media-resolution).

## Which names must agree

For a transcript `session01.cha`:

- its `@Media` header names `session01`;
- the recording is `session01.mp3`, `session01.mp4`, `session01.wav` or
  another supported format, with the extension in lower case;
- in a mapped or inferred media directory, each directory on the way is
  spelled as it is in the corpus.

All of these must match character for character. The transcript-versus-`@Media`
comparison belongs to CHAT validation: chatter reports a different name as
E531, a name in another Unicode form as W109, and a name that differs only in
letter case as W110 (chatter book: *File Names and @Media*). batchalign3
checks the recording side.

## Why the names must agree

This is TalkBank policy, settled long before batchalign3, and CLAN CHECK
enforces it as error 157 ("Media file name has to match datafile name"). A
transcript and its recording share one name, apart from the extension, for
three reasons:

- **The collections are organized by name.** The media hierarchy mirrors the
  transcript hierarchy, so a recording is found from its transcript's place
  and name alone, without a lookup table that could fall out of step.
- **Tools write and read the name.** CLAN's `medialine` writes the `@Media`
  line from the transcript's file name, CLAN plays a recording by the
  `@Media` name, and batchalign3 searches for the transcript's own name.
  These agree only because the two names are equal.
- **Collections are checked in both directions.** Every transcript must have
  its recording and every recording its transcript; pairing them by name
  is what makes that check possible.

The one exception is a recording hosted elsewhere: `@Media` may then hold a
full `http://` or `https://` address.

The practical consequence is that a rename is never of one file. Rename the
transcript and its recording together, and change the `@Media` name to
match.

## How batchalign3 matches

Every lookup lists the directory and compares names as strings. It never asks
the filesystem whether a constructed path exists, because filesystems answer
that question differently:

| Difference | macOS (default) | Linux |
|---|---|---|
| Letter case: `Session01.WAV` vs `session01.wav` | same file | different files |
| Unicode form: composed vs decomposed `ü` | same file | different files |

Asking the filesystem would find a recording on a Mac that a Linux host cannot
find, so the same transcript would align on one machine and fail on another.
Comparing strings gives the same answer everywhere, the one a Linux host gives.

A file whose name differs from the wanted one only in letter case or Unicode
form is a **near miss**. It is reported, never used:

- **Alignment and speaker identification** keep searching, exactly as a Linux
  host would. If a later place has the exact name, that file is used. If
  nothing is found, the "Cannot find audio file" error lists every place
  searched and then the near miss, for example:

  ```text
  found '/media/corpus/Session01.WAV', whose name differs from
  '/media/corpus/session01.wav' only in letter case or Unicode form; media
  names must match exactly (macOS's filesystem forgives both differences, a
  Linux one forgives neither), so rename the file
  ```

  A directory that cannot be listed is reported the same way, as an unknown
  rather than a miss.
- **Remote staging** warns with both spellings and stages no media for that
  transcript.
- **The media listing** (`/media/list`) lists nothing for a subdirectory
  spelled differently, and logs a warning.

An exact name under any supported extension beats a near miss under another:
`session01.mp3` is used even when `Session01.wav` is also present.

## Recordings named on the command line

A recording you name explicitly, as in `transcribe recording.WAV`, is not
looked up by name, so its extension may be spelled either way. Its extension
never reaches CHAT. The transcript made from it takes its `@Media` name from
the file's stem, so publish the recording under that stem with a lower-case
extension.

## Fixing a near miss

Rename the recording to the spelling the message gives. If the transcript's
own name is the odd one out, validate it with chatter first: E531, W109 or W110
says which of the transcript and `@Media` names to change. On macOS,
`mv old new` performs a rename that changes only letter case or Unicode form,
and stores the new spelling. Finder is not suitable for a Unicode-form repair.
