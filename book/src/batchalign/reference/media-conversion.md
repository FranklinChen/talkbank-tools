# Media Conversion

**Status:** Current
**Last updated:** 2026-09-10 01:43 EDT

## Overview

Batchalign commands that process audio (`align`, `transcribe`, `opensmile`,
`avqi`, `benchmark`) must resolve a media file for each input. Depending on the
command, Rust then either prepares typed PCM artifacts for worker-protocol V2
execution or passes through a normalized media path to a provider-specific
engine. Container formats that downstream audio libraries cannot read,
primarily **MP4**: must first be converted to WAV via ffmpeg.

This conversion is automatic, cached, and transparent to the user.

## Formats

| Extension | Can `soundfile` read? | Conversion needed? |
|-----------|:---------------------:|:------------------:|
| `.wav`    | Yes | No  |
| `.mp3`    | Yes | No  |
| `.flac`   | Yes | No  |
| `.ogg`    | Yes | No  |
| `.mp4`    | **No** | **Yes** |
| `.m4a`    | **No** | **Yes** |
| `.webm`   | **No** | **Yes** |
| `.wma`    | **No** | **Yes** |

The canonical list of forced-conversion extensions is defined in
`crates/batchalign/src/ensure_wav.rs::FORCED_CONVERSION`.

## Align Pipeline End-to-End

The `align` command has the most complex media handling. Here is the
complete pipeline, from CLI invocation to output CHAT, showing where
media resolution and conversion fit in.

```text
batchalign3 [--server http://<your-server>:8001] align input/ output/ --lang eng
  │
  ├─ CLI: discover .cha files in input/ (sorted largest-first)
  ├─ CLI: detect dispatch mode
  │     paths_mode / execution-host local: audio sits alongside .cha files
  │     content mode: .cha text POSTed, server resolves media from its own view
  │
  ├─ Server: POST /jobs/submit → create job (Queued → Running)
  │
  │  ┌──── For each .cha file (sequential, each has its own audio) ────┐
  │  │                                                                   │
  │  │  1. PARSE                                                         │
  │  │     parse_lenient() → ChatFile AST                                │
  │  │     pre-validate (MainTierValid)                                  │
  │  │                                                                   │
  │  │  2. MEDIA RESOLUTION                                              │
  │  │     paths_mode:                                                   │
  │  │       look alongside .cha for matching stem with known extensions │
  │  │     content mode / shared-fs remap:                               │
  │  │       trust server-visible local paths only                       │
  │  │       source_dir when the server shares that filesystem           │
  │  │       or local media_mappings / explicit --media-dir             │
  │  │                                                                   │
  │  │  3. MEDIA CONVERSION (ensure_wav)                ◄── THIS STEP    │
  │  │     .wav/.mp3/.flac/.ogg → pass through unchanged                │
  │  │     .mp4/.m4a/.webm/.wma → ffmpeg convert to WAV, cache result   │
  │  │       fingerprint: full-file BLAKE3 + versioned recipe    │
  │  │       cache dir: platform data_dir/batchalign3/media_cache/      │
  │  │       file lock: per-fingerprint .lock prevents concurrent ffmpeg │
  │  │       output: 16kHz mono PCM_S16LE WAV                           │
  │  │                                                                   │
  │  │  4. AUDIO IDENTITY                                                │
  │  │     compute_audio_identity(path, mtime, size)                     │
  │  │     used as cache key component for FA results                    │
  │  │                                                                   │
  │  │  5. AUDIO DURATION PROBE (optional)                               │
  │  │     ffprobe → total_audio_ms                                      │
  │  │     used for proportional estimation of untimed utterances        │
  │  │                                                                   │
  │  │  6. GROUP UTTERANCES                                              │
  │  │     split into ~20s time windows (Whisper) or ~15s (Wave2Vec)    │
  │  │                                                                   │
  │  │  7. CACHE LOOKUP                                                  │
  │  │     BLAKE3(words + audio_identity + time_window + engine)         │
  │  │     hits → skip worker IPC                                        │
  │  │                                                                   │
  │  │  8. FA INFERENCE (cache misses only)                              │
  │  │     checkout worker from pool                                     │
  │  │     execute_v2(task="fa", prepared_audio + prepared_text)         │
  │  │     Python reads prepared artifacts → model inference             │
  │  │     returns raw timings                                           │
  │  │                                                                   │
  │  │  9. DP ALIGNMENT                                                  │
  │  │     Hirschberg align model tokens → transcript words              │
  │  │     convert chunk-relative → file-absolute milliseconds           │
  │  │                                                                   │
  │  │  10. POST-PROCESSING                                              │
  │  │      inject timings → chain word ends → update bullets            │
  │  │      generate %wor tier → monotonicity check (E362)               │
  │  │      same-speaker overlap enforcement (E704)                      │
  │  │                                                                   │
  │  │  11. SERIALIZE                                                    │
  │  │      validate → to_chat_string() → write output .cha             │
  │  │                                                                   │
  │  └───────────────────────────────────────────────────────────────────┘
  │
  └─ CLI: poll /jobs/{id}/results → write output files
```

## ensure_wav: Conversion Cache

**Module:** `crates/batchalign/src/ensure_wav.rs`

Implements content-fingerprinted WAV conversion with file-locking and atomic writes.

### Algorithm

1. **Check extension**: if `.wav`/`.mp3`/`.flac`/`.ogg`, return unchanged.
2. **Check ffmpeg**: if not on PATH, return a clear error with install hint.
3. **Fingerprint**: stream every source byte through BLAKE3 and retain its full
   digest in the `whole-pcm16-mono16k-strict-v3` recipe namespace. This reads
   the complete file with bounded memory; same-size edits in the middle of a
   recording cannot reuse the old key. Segment keys additionally bind the
   requested time window in their own strict-recipe namespace.
4. **Cache lookup**: check the media cache directory for
   `{fingerprint}.wav`. If it exists, return immediately (cache hit).
5. **Lock**: acquire exclusive `fs2` file lock on `{fingerprint}.wav.lock`
   to prevent concurrent ffmpeg invocations for the same source file. This
   is important for parallel FA processing where multiple groups reference
   the same audio.
6. **Re-check**: another task may have completed conversion while we waited.
7. **Convert**: `ffmpeg -y -nostdin -v error -i source -acodec pcm_s16le -ar 16000 -ac 1 tmp.wav`.
   The locked slot becomes a produced slot only after its own temporary path
   passes conversion admission (see Error Handling).
8. **Atomic rename**: publish the produced slot while retaining its lock;
   release the lock only after the rename succeeds.

### ffmpeg Arguments

| Flag | Purpose |
|------|---------|
| `-y` | Overwrite output without asking |
| `-nostdin` | Prevent an unattended conversion from consuming terminal input |
| `-v error` | Emit error diagnostics, which trigger the damage check |
| `-i source` | Input file (mp4, m4a, etc.) |
| `-acodec pcm_s16le` | 16-bit signed PCM (what soundfile reads natively) |
| `-ar 16000` | 16 kHz sample rate (FA/ASR model input rate) |
| `-ac 1` | Mono (models expect single channel) |

### Cache Management

```bash
# Default cache location
ls ~/Library/Application\\ Support/batchalign3/media_cache/

# Or relocate it for isolated runs
export BATCHALIGN_MEDIA_CACHE_DIR=/tmp/ba-media-cache

# Inspect or clear both analysis + media caches
batchalign3 cache stats
batchalign3 cache clear --yes
```

### Where ensure_wav Is Called

`ensure_wav` is called in four dispatch paths, always **after** media
resolution and **before** the audio path is passed to Python workers:

| Dispatch Path | File | Purpose |
|---------------|------|---------|
| FA (align) | `runner/dispatch/fa_pipeline.rs` | Before audio identity + FA inference |
| Transcribe | `runner/dispatch/transcribe_pipeline.rs` | Before ASR inference |
| Benchmark | `runner/dispatch/benchmark_pipeline.rs:process_one_benchmark_file` | Before Rust benchmark orchestration dispatches ASR |
| Media analysis | `runner/dispatch/media_analysis_v2.rs` | Before openSMILE/AVQI prepared-audio execution |

### Error Handling

Whole-file and segment conversions report decoding errors at error level
and decode through them. A decode that reported nothing is clean. A decode
that reported errors is admitted only when it lost no audio: the source's
declared length for the requested span (probed with `ffprobe`, clipped to
the source's end for a window) and the decoded length must agree within
`DAMAGE_SHORTFALL_TOLERANCE_MS` (100 ms). Such a decode is admitted as
`DecodeIntegrity::ConcealedDamage`, carrying ffmpeg's diagnostics and both
lengths, and logged as a warning. A shortfall beyond the tolerance
(`DamagedAudioLost`) means ffmpeg dropped audio, which would shift every
later timing, so it is refused; so is a damaged decode whose lengths cannot
be measured (`DamageUnmeasured`). A refused conversion's partial output is
removed; it cannot be published.

Until 2026-09-24 conversions ran with `-xerror` and refused any diagnostic,
which refused recordings whose damaged AAC packets are concealed without
losing a sample. A clean decode's bytes are identical with or without
`-xerror`, so the `strict-v3` cache namespace is unchanged and its entries
stay valid. A concealed-damage conversion is warned about when it is made;
a later cache hit of it does not repeat the warning. This does not certify
preexisting cache bytes or media formats passed through without conversion.

If conversion fails, the file is marked with a clear error:

```text
Media conversion failed for ACWT01a.cha: ffmpeg not found in PATH.
Hint: install ffmpeg (https://ffmpeg.org/download.html) or convert
your input audio to .wav beforehand.
```

or:

```text
Media conversion failed for example.cha: ffmpeg conversion failed
for /path/to/media/example.mp4: [stderr]
```

The job continues processing remaining files, one conversion failure
does not abort the entire job.

## Media Resolution

Before conversion can happen, the server must find the audio file.
Resolution depends on the dispatch mode:

### paths_mode / execution-host local

Audio files sit alongside the `.cha` files in the input directory. The
server looks for a file with the same stem and a known media extension:

```text
input/ACWT01a.cha  →  input/ACWT01a.mp4  (or .wav, .mp3, etc.)
```

### `--server` on this machine (loopback)

For every command, an explicit loopback `--server` submits filesystem paths
via `paths_mode`:

- `source_paths`: absolute input paths the server must be able to read
- `output_paths`: absolute output paths the server must be able to write

Media is resolved from the execution host's own view, as above.

### `--server` on another host (content mode)

Only transcript-input commands reach a remote server, and the CLI posts the
transcript text; recording-input commands are refused before anything is
sent. The server resolves media from server-visible places only, in the
order given on [Server Mode](../user-guide/server-mode.md#how-the-server-finds-a-recording);
a client's private directory layout is never dereferenced, so a recording
that exists only on the client cannot be used.

## MP4 Media on Network Volumes

Total: **16,739 MP4 files** across all volumes.

| Volume | MP4 | MP3 | WAV |
|--------|----:|----:|----:|
| CHILDES | 7,988 | 20,924 | 11,042 |
| aphasia | 2,973 | 3,140 | 601 |
| ca | 1,801 | 4,696 | 4,139 |
| phon | 1,437 | 9,312 | 9,018 |
| fluency | 1,217 | 1,124 | 58 |
| class | 438 | 26 | 19 |
| tbi | 262 | 145 | 149 |
| rhd | 198 | 42 | 51 |
| asd | 101 | 47 | 37 |
| slabank | 83 | 5,478 | 3,649 |
| open | 82 | 0 | 0 |
| homebank | 65 | 2,320 | 22,455 |
| psychosis | 36 | 979 | 479 |
| samtale | 20 | 73 | 72 |
| dementia | 15 | 6,117 | 2,456 |
| psyling | 13 | 0 | 0 |
| biling | 0 | 315 | 228 |
| motor | 0 | 0 | 0 |

## Benchmarking Considerations

- **First run on MP4 files**: includes WAV conversion time (~seconds per
  file depending on duration)
- **Subsequent runs**: WAV is cached, no conversion overhead
- **For fair benchmarks**: either use `--override-media-cache` or ensure both
  old/new runs have the same cache state (warm or cold)
- **For %wor-only fixes**: conversion cache is irrelevant since the audio
  doesn't change. FA cache keys include audio identity, so same audio =
  same cached alignment.
- **Re-alignment scenario**: if re-aligning files that already had
  alignment, both the FA cache and the media conversion cache will be
  warm. Use `--override-media-cache` for cold-start numbers.

## Dependencies

- **ffmpeg** must be on PATH for mp4/m4a/webm/wma conversion. Without it,
  those formats fail with a clear error. WAV/MP3/FLAC/OGG work without
  ffmpeg.
- **ffprobe** (bundled with ffmpeg) is used for audio duration probing in
  the FA pipeline. Optional, if unavailable, proportional estimation
  uses a fallback.
- **blake3** crate for content fingerprinting.
- **fs2** crate for cross-platform file locking.
