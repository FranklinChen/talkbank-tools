# Rev.AI Integration

**Status:** Current
**Last updated:** 2026-09-07 07:04 EDT

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
  `--speaker-engine pyannote` and `nemo` select local alternatives.

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
call. Renaming or copying byte-identical media with the same extension reuses
the existing entry; changing the extension deliberately does not, because the
multipart filename is provider-visible.

Current production preparation preserves source bytes. It presents them with a
stable digest-derived metadata label, a normalized filename that retains only
the source extension, and BA3's historical `audio/mpeg` multipart type. This
describes the current request exactly; it is not a claim that the historical
MIME choice is ideal. Alternative PCM, FLAC, MIME, or padding recipes must be
revisioned and evaluated as separate evidence identities before becoming a
default.

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

## Privacy note

Using Rev.AI sends audio to an external service. Enabling the default
pyannoteAI speaker engine sends it to a second external service. If your
workflow has data-use or IRB constraints, review both accounts and your local
policy before sending production data.
