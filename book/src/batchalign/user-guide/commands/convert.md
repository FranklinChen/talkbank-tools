# convert

**Status:** Current
**Last updated:** 2026-10-06 15:09 EDT

Export recordings as WAV or MP3 through the managed runner, without ML inference.
This is separate from the automatic mono/resampled audio prepared for models.

```bash
batchalign3 convert recordings/ -o exported/ --format wav
batchalign3 convert recording.wav -o exported/ --format mp3
```

`--format` is required. Each source produces `<stem>.converted.wav` or
`<stem>.converted.mp3`, preserving nested paths. Without `-o`, the new recording
is placed beside its source. Sources and unrelated files are not rewritten or
copied. `--in-place` is refused. Existing destinations, symlinks and colliding
source stems are refused rather than overwritten or silently skipped.

WAV uses PCM16 at the source sample rate and channel count. MP3 uses VBR quality
2 and retains compatible sample rates and mono/stereo channels; incompatible
layouts require WAV rather than implicit resampling/downmixing. Multiple audio
streams are refused. Video, subtitles, container metadata and original compressed
bytes are not preserved. PCM16 is not lossless preservation of higher-bit-depth
or floating-point samples; MP3 is lossy.

The command uses managed scheduling, memory admission, file progress and
cancellation. It does not select a language model or start an inference worker.
Both the user output and the job's staged download copy must publish before the
file reports success. A later destination conflict can leave an internal staged
artifact, not a successful result or an overwritten recording.

See [media conversion](../../reference/media-conversion.md) for the checked
producer and binary-delivery contracts.
