# Network and Transfer Costs

**Status:** Current
**Last updated:** 2026-09-22 14:00 EDT

Batchalign moves bytes over a network in four distinct ways, and every
command and engine combination decides which of them it uses. None of them
is announced on the command line. If you work over a slow or metered link,
or your recordings must not leave a particular machine, read this page
before running anything. Every statement here was checked against the
source of this release; sizes are approximate.

## The four ways bytes move

1. **Reading the recording on the execution host.** A recording on local
   disk costs nothing to read. A recording on a network mount (NFS, SMB, a
   cloud drive) crosses that mount's link on every read, and the read looks
   exactly like a local one: same path, same command, no warning.
   **Where Batchalign runs decides where the media travels.**
2. **Client to server, with `--server`.** Only transcript text crosses this
   connection, never a recording. See [Server Mode](server-mode.md).
3. **Execution host to a cloud provider.** With a cloud engine selected, the
   audio (or, for translation, the text) is uploaded to that provider from
   whichever machine runs Batchalign. Moving the run to another host does
   not change this; changing the engine does.
4. **Model downloads.** Local engines fetch their models once, on first use,
   and a few components make small refresh requests on later runs. See
   [Model Downloads and Caching](model-downloads.md).

## Which commands read a recording at all

| Command | Reads a recording? | How it finds it | How much it reads |
| --- | --- | --- | --- |
| `morphotag`, `utseg`, `translate`, `coref`, `compare` | **No.** These never open media, even when the server has media mappings configured. | n/a | Nothing |
| `align`, `speaker-identify` | Yes, resolved from the transcript. | See "How a recording is located" below | See "What `align` reads" below |
| `transcribe`, `transcribe_s`, `benchmark` | Yes, the recording IS the input. | The path you pass | The whole file |
| `opensmile`, `avqi`, `diarize` | Yes, the recording IS the input. | The path you pass | The whole file, fully decoded |

### How a recording is located

For `align` and `speaker-identify` the execution host searches a fixed list
of places, set out in [How the server finds a
recording](server-mode.md#how-the-server-finds-a-recording). The step to
know about is the media mapping: if any component of the input path equals
a mapping key, for example a corpus repository name such as
`childes-eng-na-data`, that mapping's root is searched automatically, with
no flag and nothing printed. A mapping whose root is a network mount
therefore turns `batchalign3 align` on a corpus transcript into a remote
read of the recording, silently.

### What `align` reads

- Containers that need conversion (`.mp4`, `.m4a`, `.webm`, `.wma`) are
  decoded in full by ffmpeg, once, and the 16 kHz mono WAV is cached
  locally.
- `.wav`, `.mp3`, `.flac` and `.ogg` are used in place, so later reads go
  to the original path.
- Forced alignment extracts one audio segment per alignment group with
  ffmpeg, from the source path.
- For those container formats the conversion cache is keyed on a hash of
  every byte of the source, so a second run still reads the whole source; it
  skips only the conversion. Other formats are identified by path, size and
  modification time, once per file.
- With the default `--utr-engine rev`, any transcript that has untimed
  utterances causes the **whole original file** to be uploaded to Rev.AI,
  a video included. See the engine table below.

## The mounted-media trap

A media mapping or media root that points at a network mount makes every
read of a recording a transfer over that mount's link. On a fast local
network this is the intended way to work: mount the media server once and
run against it. Over a slow, metered or intercontinental link the same
configuration is the most expensive thing Batchalign can do, and it gives
no sign of doing it.

The rule: **run Batchalign on a host that is on the same network as the
media it will read**, and copy only transcripts across the slow link.
Concretely:

- To align or re-transcribe recordings that live on a media server, log in
  to a machine on that server's network and run there; or, for the
  transcript commands, use `--server` against a server on that network,
  which ships only the transcript ([Server Mode](server-mode.md)).
- To work with recordings that are on your own machine, keep them beside
  the transcripts and run locally. Do not mount a remote media root just to
  have it available; an input path that happens to name a corpus repository
  will select it.
- Never configure a media mapping whose root crosses the slow link.

## What each engine sends off the machine

Uploads happen from the execution host, whatever machine that is. "Source
size" means the file as it is on disk for `.wav`, `.mp3`, `.flac` and
`.ogg`; for containers that were converted it means the converted 16 kHz
mono 16-bit WAV, which is about 115 MB per hour of audio.

| Engine | Used by | What leaves the machine | Sent to | Cached? |
| --- | --- | --- | --- | --- |
| `rev` (Rev.AI), the default ASR | `transcribe`, `transcribe_s`, `benchmark` | The whole file as one upload: the source as it is on disk, or the converted WAV for the container formats. Up to three attempts, each re-sending it. `--lang auto` adds a second full upload for language identification. | Rev.AI | Yes, by content and settings: a rerun with a cache hit uploads nothing. |
| `rev` as `--utr-engine` (the default) | `align` | The whole **original** file, whenever the transcript has untimed utterances. Never windows. | Rev.AI | Yes, same cache. |
| `tencent` (ASR) | `transcribe` | The whole file, to your own Tencent COS bucket, deleted after the job. | Tencent Cloud | No: every rerun uploads again. |
| `tencent` as `--utr-engine` | `align` | Windows around the untimed utterances (16 kHz mono WAV segments) when fewer than half are untimed and the audio exceeds 60 s; otherwise the whole converted file. | Tencent Cloud | Per window. |
| `aliyun` (ASR) | `transcribe` | The whole recording as uncompressed PCM, streamed. | Aliyun NLS | No. |
| `pyannote-ai`, the default when `--diarization enabled` | `transcribe`, `diarize` | A mono 16-bit WAV of the whole recording. | pyannoteAI | Yes, by content and settings. |
| `google`, `tencent`, `aliyun` (translation) | `translate` | Text only, one request per utterance. | The provider | No. |
| `whisper` as `--utr-engine` | `align` | **Nothing.** Local inference over windows around the untimed utterances. | n/a | Per window |
| Every engine not named above (`--help` lists them per task) | various | **Nothing.** Local inference, after a one-time model download. | n/a | Model cache |

Two small requests happen regardless of engine: the CLI checks PyPI for a
newer release once a day (disable with `BATCHALIGN_NO_UPDATE_CHECK=1`), and
every language-processing worker refreshes Stanza's model catalog, about
1 MB, when it starts.

## Model downloads

Local engines download their models on first use and read them from a local
cache afterwards. The sizes range from a few hundred megabytes to over five
gigabytes per model, and a multilingual workflow can cache tens of
gigabytes. The full table, the cache locations, which components still make
a small network request when cached, and how to download ahead of time on a
good connection are on
[Model Downloads and Caching](model-downloads.md).

## Working over a slow or metered link

- Keep recordings beside their transcripts on the machine that runs
  Batchalign, and run locally.
- For recordings that live on a media server, run on that server's network
  and bring back only the transcripts. For `align`, `morphotag` and the
  other transcript commands, `--server` does this for you.
- Prefer local engines: `whisper_rs` or `whisper` for most languages,
  `funaudio`, `paraformer` or `qwen` for Chinese varieties, `--speaker-engine
  pyannote` or `nemo` for diarization, `nllb` for translation. Nothing then
  leaves the machine after the first model download.
- With Rev.AI, pass the language explicitly rather than `--lang auto`, which
  uploads the recording twice.
- Download models on a good connection first: `batchalign3 setup
  --prefetch-whisper-rs` covers the native Whisper model, and the manual
  recipe on the model-downloads page covers the rest.
- Do not point `media_roots` or `media_mappings` at a mount that crosses the
  slow link.
