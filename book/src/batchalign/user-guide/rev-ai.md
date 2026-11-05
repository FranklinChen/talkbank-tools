# Rev.AI Integration

**Status:** Current
**Last updated:** 2026-11-05 19:37 EST

Rev.AI is the default ASR engine for `batchalign3 transcribe`, and the default
UTR engine for `batchalign3 align`.

In server mode, those Rev.AI paths are now Rust-owned end to end: the server
submits or polls Rev.AI jobs directly and keeps Python reserved for engines
that genuinely require Python-hosted model libraries.

## Configure a Rev.AI key

Interactive setup:

```bash
batchalign3 setup
```

Non-interactive setup:

```bash
batchalign3 setup --non-interactive --engine rev --rev-key <YOUR_REV_AI_KEY>
```

This writes the key and default engine selection to `~/.batchalign.ini`.

## Use Rev.AI explicitly

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine rev --lang eng
batchalign3 align corpus/ -o aligned/ --utr-engine rev
```

## Speaker labels, utterance segmentation, and `--diarize`

- Rev.AI already returns first-pass speaker labels. BA3 applies those labels
  by default, so plain Rev transcription already produces multi-speaker output.
- BA3 still performs its own utterance segmentation after ASR; speaker
  attribution and utterance boundary detection are separate steps.
- If you pass `--diarize` (or `--diarization enabled`), BA3 runs the separate
  speaker stage even on top of Rev output. Dedicated labels replace Rev's
  speaker projection, are applied to timed ASR words, and split chunks before
  utterance segmentation. pyannoteAI Precision-2 is the default;
  `--speaker-engine pyannote` selects the local alternative.

## Provider-visible audio format matters

Rev.AI can return different words, timings, confidence values, and speaker
boundaries for perceptually equivalent encodings of one recording. In a
controlled 94-clip test, submitting original MP3 bytes versus decoded PCM16
WAV changed the lexical response in 81 cases and measured speaker/monologue
boundaries in 54; matched word starts and ends shifted by a median 40 ms. This
proves sensitivity, not that WAV is inherently more accurate. Individual clips
showed plausible gains and losses, so a default-format change needs blinded
quality adjudication against the audio.

BA3's durable Rev cache therefore keys the exact provider-visible media bytes
and their upload presentation, not an assumption that two encodings “sound the
same.” Re-encoding produces a different evidence entry and can incur a new Rev
call. Renaming or copying byte-identical media reuses the existing entry,
whatever the file is called: the upload's declared type and file name come from
the container ffprobe reads in the bytes, never from the extension, so a WAV
renamed `.mp3` is still uploaded as a WAV and keys the same.

Current production preparation preserves source bytes. It presents them with a
stable digest-derived metadata label and a multipart file name
`provider-media.<ext>` and media type that are both the probed container's:
`mp3` and `audio/mpeg`, `wav` and `audio/wav`, `flac` and `audio/flac`, `ogg`
and `audio/ogg`, `aac` and `audio/aac`, `mp4` and `audio/mp4`, under the
preparation recipe `source_bytes_declared_container_v1`. MPEG audio keeps the
historical recipe `source_bytes_legacy_audio_mpeg_v1`, because its declaration
was already true, so an MPEG recording at a `.mp3` path has the request key it
always had. Before 2026-11-03 every upload was declared `audio/mpeg` and named
after the source extension, which worked only because Rev.AI sniffs content; a
recording cached under that presentation (a WAV, FLAC, Ogg, AAC or MP4, or an
MPEG file under another name) is a different evidence identity today, and its
first run under the new key finds the old entry by consulting the key it had
then, stores the evidence under the new key and uses it, with no second
fetch; see the transcribe page's note on the evidence key. A file in a container the probe
has not measured (AIFF, say), or one BA3 does not upload as is (Matroska and
ASF, which media conversion turns into WAV before any engine sees them), is
refused by name rather than declared as something it is not. Alternative PCM,
FLAC or padding recipes must still be revisioned and evaluated as separate
evidence identities before becoming a default.

## Retain a reproducible Rev evidence record

Add `--debug-dir PATH` to a Rev transcription or Rev-backed `align` run to
write fail-closed `*_rev_evidence.json` sidecars. They join the source and
provider-media BLAKE3 digests, preparation recipe, exact multipart
filename/MIME/metadata, language, speaker count, request-policy and model
revisions, raw cache key, cache outcome, exact-versus-legacy transcript
fidelity, and local projection revision. The sidecar contains no credential or
machine-local source path. If requested
evidence cannot be serialized or durably written, the file fails instead of
silently completing without the research record.

A transcribe file normally has one record. Align may have one full-file record
or several stable cache-keyed records when UTR analyzes partial audio windows.

## Use a local model instead

If you do not want cloud ASR, use a local Whisper model:

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine whisper --lang eng
```

For Rust-native whisper.cpp, run in-process with no Python worker:

```bash
batchalign3 transcribe recordings/ -o transcripts/ --asr-engine whisper_rs --lang eng
```

## When an upload fails

A failed Rev.AI upload is reported as a **provider** failure, never as a
validation failure, and the distinction is the one that decides what you should
do next:

- `error_category: provider_transient`: the connection dropped, the request
  body could not be delivered, or Rev.AI answered 5xx. Nothing about your
  request was wrong; the same files are worth submitting again.
- `error_category: provider_terminal`: Rev.AI refused the request (4xx) or
  failed the job. Repeating it will not help; read the message.

The message carries the provider error together with its full cause chain, and
for an exhausted upload it lists what happened on **each** of the three
attempts rather than only the last. A transport fault whose top line reads
`request or response body error for url (...)` will name the underlying socket
condition after the colon.

Until 2026-09-03 every Rev.AI failure was flattened into
`error_category: validation`, which reads as "you sent bad input" and hid the
fact that a batch was worth retrying.

## When a job takes too long, or a status poll fails

After the upload, BA3 polls the job's status until Rev.AI reports it
transcribed or failed. That wait is bounded, and a failure of a single poll
does not end it:

- **Deadline.** A transcription is polled for at most two hours (a
  language-identification job, under `--lang auto`, for twenty minutes). The
  bound is BA3's limit on how long one worker is held by one job, not a claim
  about Rev.AI's turnaround.
- **Transient poll failures.** A dropped connection, a 5xx or a 429 on a
  status poll is absorbed: the wait continues after the usual interval. Five
  such failures in one wait exhaust the budget and end it.

Either way the wait ends, the job is **not abandoned**. It stays submitted and
paid for at Rev.AI, the file is reported as a `provider_transient` failure whose
message names the Rev.AI job id and the reason (`still in progress after 7200s,
past the 7200s polling deadline`, or the list of poll failures), and the job is
recorded in the evidence cache against the request that submitted it. The next
run of the same request, which includes the control plane's own retry of the
file, finds that record and **resumes the wait for the recorded job instead of
uploading again**; when its transcript arrives it is committed under the usual
key and the record is removed. Rev.AI's own verdict that a resumed job failed,
or that it no longer exists, removes the record too, so the run after that
submits afresh.

Before 2026-11-03 the wait had no bound, so a job Rev.AI reported in progress
forever held the file forever, and one transient failure of a status poll was
reported as the job's failure, so the retry re-submitted the media and paid
twice for one recording.

An unfinished language-identification job is not recorded: the transcription
falls back to `auto` and reads the detected language from the completed job,
as it does when identification fails.

## Privacy note

Using Rev.AI sends audio to an external service. Enabling the default
pyannoteAI speaker engine sends it to a second external service. If your
workflow has data-use or IRB constraints, review both accounts and your local
policy before sending production data.
