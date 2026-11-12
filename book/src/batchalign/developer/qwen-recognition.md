# Qwen recognition: states, language profiles, and timing

Qwen recognition separates configuration, loaded inference capability, and
validated alignment. Extend these boundaries rather than adding readiness flags
or changing the request language to satisfy a model's language vocabulary.
For CLI options and supported command scope, see
[Language Code Resolution](../reference/language-code-resolution.md#taiwanese-transcription-nan).

## Ownership and loading

The implementation still lives under `inference/languages/cantonese/` for
historical reasons; that directory name does not restrict Qwen to Cantonese.
Paths below are relative to the repository root.

| Owner | Responsibility |
| --- | --- |
| `crates/batchalign/src/transcribe/segmentation.rs` (`qwen_recognition`) | Decide each language's recognition policy (decoding window, punctuation fate) from the one table of recognizer-boundary languages. |
| `batchalign/inference/qwen_asr_profile.py` | Resolve the recognition and companion alignment labels (`QWEN_RECOGNITION_LABELS`), admit optional context, and translate the received policy into recognizer terms. |
| `batchalign/worker/_model_loading/asr.py` (`qwen_recognition`) | Parse the spawn policy at the worker boundary; refuse a Qwen worker spawned without it. |
| `batchalign/inference/languages/cantonese/_qwen_common.py` | Validate configuration, load both models, decode bounded chunks and convert aligned times to recording coordinates. |
| `batchalign/inference/languages/cantonese/_qwen_asr.py` | Resolve worker options, enforce the loading timeout and publish the ready recognizer. |
| `batchalign/inference/qwen_forced_alignment.py` | Shared aligner adapter and window-relative alignment types; standalone FA language admission. |
| `batchalign/inference/qwen_punctuation.py` | Validate lexical correspondence and retain punctuation with the original timed units. |
| `crates/batchalign/src/transcribe/segmentation.rs` (`TranscribeSegmentation`) | Admit a transcription route with either a text-model capability or ASR boundaries, from the same table. |

```mermaid
flowchart LR
    O["Worker options + pinned model paths"] --> C["QwenRecognizerConfig"]
    C -->|"load(): both models succeed"| R["ReadyQwenRecognizer"]
    C -->|"validation, load or timeout failure"| E["Error: no new ready state published"]
    R --> P["Worker publishes recognizer"]
    P --> I["Inference"]
```

`QwenRecognizerConfig` is frozen and validates the native checkpoint spelling
and language profile in construction. It has a `load()` operation, but no
`transcribe()` operation or partially loaded model field. The worker supplies
resolved snapshot paths separately from model identifiers: loading uses the
pinned paths while checkpoint validation retains the requested identity.

`load()` returns `ReadyQwenRecognizer`, which owns that configuration and a
`LoadedQwen` containing the processor, recognition model and forced aligner.
Inference methods belong to this ready state. Do not restore a lazy `_get_model`
method, mutable language aliases, or a `warm()` call whose ordering every caller
must remember. Frozen Python dataclasses prevent field reassignment; they do
not make the underlying model objects deeply immutable or prevent deliberate
manual construction. Production readiness comes from the loader transition.

The worker brackets `config.load()` with its timeout and restores the previous
signal handler in `finally`. It assigns the module's `_recognizer` only after
loading completes. The optional worker slot means “not published yet”; it is
not a recognizer with optional internal models. Tests may construct a ready
recognizer with fake loaded components to exercise inference boundaries.

## Keep three language decisions separate

For `nan`, the request and CHAT language remain `nan`. The profile selects
`Chinese` for Qwen recognition and its Chinese timestamp companion. Those
model labels do not change the recording's language and do not qualify a
separate Taiwanese forced aligner.

The decoding window and the fate of recognized punctuation are decided in
Rust, once, by `qwen_recognition` in `transcribe/segmentation.rs`. It reads the
same table of recognizer-boundary languages that `TranscribeSegmentation` reads
to choose the ASR-boundaries route, so a route that relies on recognizer
punctuation always gets a recognizer told to restore it. The worker pool sends
the decision to every Qwen worker at spawn as `QwenRecognitionV2` JSON under
`QWEN_RECOGNITION_KEY` (`qwen_recognition`). The Python loader refuses to
load without it and never decides it from the language code.

Today Taiwanese gets 30-second windows and restored punctuation
(`RecognitionPunctuationV2.RESTORE`); every other language gets the chunker's
default window and discarded punctuation, because a segmentation model
supplies their boundaries. The recognizer reads the profile with an
exhaustive `match`. Adding a recognizer-boundary language is one entry in the
Rust table, from which the route, the policy and the worker follow, plus a
Qwen recognition label in `QWEN_RECOGNITION_LABELS`
(`batchalign/inference/qwen_forced_alignment.py`) if the language is new to
Qwen. That label table is separate from the aligner's `QWEN_LANG_LABELS`, which
also decides standalone forced-alignment admission.

```mermaid
flowchart LR
    T["RECOGNIZER_BOUNDARY_LANGUAGES<br/>(Rust, one table)"] --> S["TranscribeSegmentation::resolve<br/>AsrBoundaries or Model"]
    T --> Q["qwen_recognition(language)<br/>QwenRecognitionV2"]
    Q -->|"spawn key qwen_recognition"| W["Python loader<br/>refuses if absent"]
    W --> P["QwenAsrProfile<br/>window + punctuation"]
```

`qwen_context` is optional text admitted into the
configuration before loading; absent or empty context supplies no prompt.
The worker passes a nonempty context through the processor's `prompt` argument.
Context can be emitted as invented speech, so dialect instructions are not
injected automatically. Keep explicit overrides in the submitted job options
so comparisons can identify the actual context used.

Rust keeps transcription routing separate from standalone utterance
segmentation. `TranscribeSegmentation` wraps a private `SegmentationRoute`:

- `Model(UtsegRoute)` carries a resolved text-model route.
- `AsrBoundaries { language }` preserves recognition and speaker boundaries
  without claiming a text model.

The wrapper exposes `model() -> Option<&UtsegRoute>`. Model-consuming code
must narrow to that capability before calling model operations. A boolean
such as “uses a model” followed by unrelated fallback getters would lose this
relationship. The `nan` route takes ASR boundaries and skips the corresponding
text-model inference. Standalone `UtsegRoute::resolve` remains unchanged.
Engine-specific request validation is also independent: admitting native
transcription boundaries does not grant another ASR engine language support.

Adding a recognition profile therefore does not automatically add standalone
`align`, `utseg`, or morphology support. Review each command's admission
boundary separately; see [Adding Language Support](adding-language-support.md).

## Pair punctuation with alignment evidence

Recognition and forced alignment have different output units. The aligner can
return character-sized units and omit punctuation. The ASR adapter handles
that difference through this sequence:

```text
PairedWindowAlignment (model units and reported times, arity checked)
  -> WindowAlignment + AudioWindow (bounded observed spans or unavailable timing)
  -> SpokenWindowWords (spoken units for ASR)
  -> PunctuatedAlignment (profile RESTORE only; validated lexical correspondence)
  -> RecordingChunkSpeech (place the bounded speech at its recording offset)
  -> RecordingStart / RecordingPrefix (admit the next disjoint window)
  -> CompletedRecording (publish immutable recording chunks)
```

`PairedWindowAlignment` snapshots both sequences and admits their arity.
Both ASR and standalone FA must then construct `WindowAlignment` with the
actual audio duration. Admission never fails the file on model timing: every
unit gets exactly one `WindowTiming` state, and every state but
`ObservedWindowSpan` is explicit absence that keeps its reason.

```mermaid
flowchart TD
    decoded["decoded unit"] --> read{"read_decoded_unit:<br/>both times present and numeric?"}
    read -->|no| missing["RejectedWindowTiming<br/>(MissingUnitTime)"]
    read -->|yes| unit["WindowAlignedWord"]
    unit --> judge{"_judge_span: own span finite,<br/>nonnegative, forward, inside window?"}
    judge -->|no| rejected["RejectedWindowTiming<br/>(that unit's defect)"]
    judge -->|"yes, zero width"| unavailable["UnavailableWindowTiming"]
    judge -->|"yes, positive"| observed["ObservedWindowSpan"]
    observed --> order{"_first_disorder: spoken observed units'<br/>starts and ends nondecreasing?"}
    order -->|yes| kept["kept: window keeps its timing"]
    order -->|"no: first backwards pair"| disordered["DisorderedWindowTiming<br/>(every observed span of the window,<br/>each keeping the same pair)"]
```

`read_decoded_unit` is the boundary parse, and it is total: a decoded unit with
a missing or non-numeric time is an `UntimedReportedUnit`, kept in its
position (the fold attributes units to words by position) and admitted as
`RejectedWindowTiming` with a `MissingUnitTime` defect. Nothing invents a zero
or NaN in its place. A unit whose own span is non-finite, negative, reversed or
past the window's end is `RejectedWindowTiming`: only that unit is untimed. A pair of spoken
units whose start or end runs backwards is a defect of the window's order,
not of either unit, so no span in that window can be trusted:
`_first_disorder` finds the first such pair and every observed span in the
window becomes `DisorderedWindowTiming`, keeping the pair. Units already
rejected keep their own, more specific defect. Other windows are unaffected.
Ordinary overlap is permitted when both boundaries advance. Zero-width units
retain their fold positions as unavailable evidence and cannot supply observed
word endpoints. The FA fold leaves a word untimed if any constituent is not
observed, or if its interval disappears when rounded to milliseconds. No
neighboring interval supplies the missing times.

Recording placement adds the offset once, through `_place`, the one edge from
window time to recording time. At a large enough offset a short positive span
cannot stay positive in float64; `_place` then returns `CollapsedPlacement`, an
untimed state keeping a `CollapsedSpan` defect, and `_align_chunk` logs it
through `log_placement_defects`, instead of failing the chunk or inventing a
span. `_stays_positive` is the one placement rule, shared by `_place` and the
`ObservedRecordingSpan` guard. An immutable recording prefix accepts
only a subsequent disjoint window; containment plus ordered windows prevents
cross-chunk time reversal. Publication consumes that prefix, so the ASR result
cannot publish a bare, unchecked list of chunk timestamps. Unavailable words
retain their text with absent timestamps on the ASR wire.

A window with any untimed defect is logged once at WARNING by
`_log_untimed_defects`: how many units are untimed, each defect with the
original reported times (for a backwards pair, both units and whether the start
or end moved), and the unchanged report as `RejectedAlignmentDiagnostic` JSON,
including the native classifier trace. `_align_chunk` passes its
`RecordingWindow`, so the log names the recording window while the times
stay window-relative. The reason does not yet cross the FA worker wire, which
carries only "no timing" for a word; the file's untimed-words diagnostic counts
these words. `QwenTimingInvalid` remains only as the guard on the span
constructors, so a direct caller cannot build an observed span from a span
admission rejected. Do not sort or clamp these times to manufacture an
apparently valid alignment.

`restore_punctuation(text, spoken)` compares the complete lexical sequence of
the recognition text with the concatenated alignment units. Letters, numbers,
Unicode combining marks and the ASCII apostrophe count as lexical characters.
This preserves tone marks as evidence: dropping a combining mark is a mismatch,
not punctuation normalization. `_judge_correspondence` is the one rule: it
returns a `_Correspondence` proof (each unit's punctuation) or a
`PunctuationRefusal`. There is no fuzzy match. `PunctuatedAlignment` is built
only from that proof, so nothing judges twice, and `restore_punctuation` never
raises; it returns the alignment or the refusal, and `_with_punctuation`
decides what each refusal honestly keeps:

| Refusal | Meaning | The window keeps |
| --- | --- | --- |
| `NonlexicalUnit` | a unit with no lexical character; the units still spell the recognized text | its timed spoken units, without restored punctuation |
| `LexicalDrift` | the aligner's units spell different letters, digits or marks | the recognized text, untimed (`UntimedRecognition`), never the aligner's spelling |

Both are logged with the recording window. Neither fails the file, and the
transcript text is never replaced by the aligner's spelling.

On success, `PunctuatedAlignment` holds one owned tuple of `AlignedWindowWord`
values. It appends supported sentence/clause punctuation to the preceding unit
and preserves the same immutable timing evidence. Quotes and arbitrary symbols
are not restored. Do not return a separate spelling list and zip it onto times:
that permits changed text and timing evidence to diverge after validation.

Window-to-recording conversion belongs in `_align_chunk`, where the audio
chunk offset is available. The shared forced aligner must continue returning
window-relative times; adding an offset there would affect standalone FA too.
Likewise, the ASR-specific spoken-unit filtering does not belong in the shared
aligner, whose standalone fold needs every unit's position.

## Replay compatibility

The ready standalone FA host reports a `QwenFaCacheIdentity`: model ID plus
`QwenFaAdmissionRevision.TIMING_DEFECTS_UNTIMED_V3`. The wire name ends in
`@timing-defects-untimed-v3`. Both raw and derived FA cache envelopes are
admitted against this name. Under this revision no timing defect fails the
file: a unit whose own span is unusable or whose time is missing is
`RejectedWindowTiming`, and a window whose units run backwards against each
other is `DisorderedWindowTiming`, as described above. Neither earlier revision
replays under this one:

- `@unit-defects-untimed-v2` failed the file for a backwards pair or a missing
  time.
- `@bounded-unit-evidence-v1` also failed it for one unusable unit.
- Earlier model-only namespaces lack the unit evidence needed to prove these
  rules retroactively.

Existing cache bytes are retained. This changes replay compatibility without
renaming the selected model.

The shared identity fixture is consumed by the Python loader test and the
Rust raw/derived replay test, including acceptance of current evidence. A
revision is an admission compatibility statement, not a model-quality score.

These types establish correspondence and coordinate ownership, not acoustic
accuracy. A lexically consistent alignment can still contain zero-duration or
misplaced intervals. Restored punctuation also creates nonlexical slots in
later speaker projection, so aggregate inferred-speaker counts are not a
lexical speaker-error rate.

## Extension and verification

For a change to this route, inspect the profile, worker publication, ready
inference methods, Rust request admission and segmentation consumers together.
Keep the following regression boundaries:

| Boundary | Existing coverage |
| --- | --- |
| Default/explicit context, failed load, publication of a ready recognizer, frozen configuration, restore versus discard punctuation | `batchalign/tests/pipelines/asr/test_qwen_taiwanese.py` |
| Exact punctuation pairing, lexical drift, combining marks, nonzero chunk offset | `batchalign/tests/pipelines/asr/test_qwen_taiwanese.py` |
| Checkpoint spelling and explicit model choice | `batchalign/tests/pipelines/asr/test_qwen_checkpoint_ids.py` |
| Bounded decoding and classification before blank-text handling | `batchalign/tests/pipelines/asr/test_qwen_transcribe_chunk_bounds.py`, `test_qwen_run_model_ordering.py` in the same directory |
| Loader timeout | `test_load_qwen_asr_times_out_on_hang` in `batchalign/tests/pipelines/asr/test_asr_model_loading.py` |
| Shared alignment adapter contracts | `batchalign/tests/pipelines/fa/test_qwen_forced_alignment.py` |
| Real-model word timestamps (golden test; requires weights) | `batchalign/tests/pipelines/asr/test_qwen_asr_forced_aligner.py` |
| Native boundaries without standalone utseg support; route and Qwen policy agree per language | Tests in `crates/batchalign/src/transcribe/segmentation.rs` |
| Every Qwen worker spawns with its language's policy | `a_qwen_worker_spawns_with_its_language_recognition_policy` in `crates/batchalign/src/worker/pool/execute_v2.rs` |
| Policy wire shape on both sides; loader refusal without a policy | `crates/batchalign-types/src/worker_v2/qwen_recognition.rs`, `test_qwen_recognition_policy` in `batchalign/tests/test_ipc_type_conformance.py`, `test_qwen_taiwanese.py` |
| Qwen transcription admission without granting Rev.AI support | `transcribe_taiwanese_qwen_preserves_its_language` in `crates/batchalign/src/types/request.rs` |

A focused profile/worker regression command is:

```bash
uv run pytest batchalign/tests/pipelines/asr/test_qwen_taiwanese.py -q
```

Select the other affected boundary tests when changing their implementation;
this one command is not full engine qualification. Test doubles establish
state transitions and failures without loading weights. Real-model evaluation
must separately retain the input identity, runtime build, both model revisions,
submitted options, raw ASR evidence and resulting CHAT. Change one relevant
setting at a time when attributing a quality difference. CHAT validation,
interval diagnostics, lexical accuracy and acoustic review answer different
questions; record each result at its actual scope.
