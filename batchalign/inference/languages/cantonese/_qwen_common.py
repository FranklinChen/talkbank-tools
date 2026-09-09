"""Qwen3-ASR helpers for the built-in Cantonese engine.

Drives ``transformers``' native ``qwen3_asr`` module directly. Mirrors the
``_funaudio_common.FunAudioRecognizer`` shape so the worker bootstrap can swap
engines uniformly. The actual model load is deferred until first inference
call, which keeps worker startup bounded when no Qwen jobs are dispatched.

Engine selection rationale: Qwen3-ASR is an open-weight Cantonese-capable ASR
model from Alibaba. It is wired here as one engine option among several.

**Why this no longer uses the ``qwen-asr`` package.** That package declared
``transformers==4.57.6``, an exact pin, and it was the single dependency
holding the whole project on the last 4.x release, which in turn blocked the
calibrated AudioFlamingo3 model in ``machine_ear``. Upstream ``transformers``
implements these same checkpoints natively as of 5.13.0, so the pin is
avoidable rather than negotiable. The swap was gated on a CER differential over
all 18 fixtures of the Cantonese benchmark, not on the test suite, because the
failure mode of an engine change is a quality regression that imports cleanly
and passes everything. That differential put the native path ahead on 15 of 18
fixtures and ahead on every CHILDES fixture. Two things it also turned up are
handled in ``_qwen_chunking``: the package's real value-add was long-audio
chunking, which we now own, and greedy decoding can run away into repetition,
which we now detect.

Device default is ``cpu``. Apple Silicon hosts have no CUDA, and MPS is not
usable for this model here: long audio is SIGKILLed with no exception and no
crash report, and float16 on MPS segfaults at load. Hosts with CUDA can opt in
via ``engine_overrides["qwen_device"]="cuda"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.qwen_forced_alignment import (
    QWEN_FORCED_ALIGNER_MODEL_ID,
    ForcedAligner,
    QwenForcedAligner,
    QwenLanguageLabel,
    SpokenWindowWords,
    resolve_qwen_language,
)

from ._asr_types import AsrElement, AsrGenerationPayload, AsrMonologue, TimedWord
from ._qwen_chunking import (
    SAMPLE_RATE,
    AudioChunk,
    Bounded,
    BoundReason,
    ChunkDecodeBudget,
    ChunkDecodeOutcome,
    Completed,
    DecodeBudget,
    RunawayOutput,
    detect_runaway,
    split_audio_into_chunks,
)

L = logging.getLogger("batchalign.hk.qwen")


# The language-label map, the aligner checkpoint id and the aligner call all
# moved to `batchalign.inference.qwen_forced_alignment` when the aligner became
# a first-class FA engine in its own right. They are imported above rather than
# restated here: the ASR engine and the `qwen3_fa` engine drive the same
# aligner, and two copies of that call is how two coordinate conversions come
# to disagree.


def _check_native_checkpoint(model_id: str) -> None:
    """Refuse a pre-migration checkpoint id with an instruction, not a stack trace.

    The native ``transformers`` module targets the ``-hf`` repositories; the
    bare ones are the sidecar's. Both exist on the Hub, so an operator who
    kept an old ``--engine-overrides '{"qwen_model": ...}'`` invocation would
    otherwise get an obscure failure deep inside a loader. Deliberately NOT
    rewritten to the ``-hf`` form on their behalf: silently substituting a
    different model than the one asked for is how a run ends up not being the
    run somebody thinks it was.
    """
    if model_id.startswith("Qwen/Qwen3-ASR") and not model_id.endswith("-hf"):
        raise ValueError(
            f"Qwen3-ASR checkpoint {model_id!r} is the pre-2026-08 sidecar "
            f"spelling. The engine now uses transformers' native qwen3_asr "
            f"module, which targets the '-hf' repositories: pass "
            f"{model_id + '-hf'!r} instead."
        )


@dataclass
class QwenAsrSegment:
    """One chunk of transcript with its word-level timestamps.

    ``text`` is the chunk's full string. ``word_timestamps`` is a list of
    ``(start_s, end_s, word_text)`` tuples in RECORDING-relative seconds,
    already converted out of the aligner's window-relative space. It is empty
    when the aligner returned nothing for the chunk.
    """

    text: str
    word_timestamps: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass(frozen=True)
class QwenTranscription:
    """Everything one recording produced, including what it did NOT produce.

    A named seam rather than a bare pair because `refused` is not a detail of
    the segments: it is audio that reached the transcript as nothing at all,
    and a caller that never looks at it cannot tell that stretch from silence.
    Keeping it beside the segments is what stops the refusal from living only
    in a log line.
    """

    segments: list[QwenAsrSegment]
    refused: list[RunawayOutput]

    @property
    def refused_seconds(self) -> float:
        return sum(r.audio_seconds for r in self.refused)


@dataclass(frozen=True)
class LoadedQwen:
    """Both models, loaded. Its existence is the proof the aligner is wired.

    The engine needs an ASR model AND a forced aligner, because word timings
    are what `%wor` is built from. Previously that pairing was a convention:
    the loader passed an aligner argument and a regression test watched to see
    that it kept doing so, after a production run deadlocked for 32 minutes
    when it did not.

    A test is the wrong instrument for that. This type has one constructor,
    which loads both or raises, so "transcribe with no aligner" has no
    signature to travel through and the test that watched for it is gone.

    The aligner is now the shared `ForcedAligner` rather than a loose
    processor/model pair of `Any`s, so the pairing is a type as well as a
    constructor: the standalone `qwen3_fa` engine loads exactly the same thing.
    """

    processor: Any
    model: Any
    aligner: ForcedAligner

    @classmethod
    def load(
        cls, model_id: str, aligner_id: str, device: str, dtype: Any
    ) -> LoadedQwen:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForMultimodalLM,
            AutoProcessor,
        )

        return cls(
            processor=AutoProcessor.from_pretrained(model_id),
            model=AutoModelForMultimodalLM.from_pretrained(
                model_id, device_map=device, dtype=dtype
            ),
            aligner=QwenForcedAligner.load(
                aligner_id=aligner_id, device=device, dtype=dtype
            ),
        )


class _RecordingMaxTimeCriteria:
    """``transformers.generation.MaxTimeCriteria`` that remembers firing.

    ``StoppingCriteriaList`` ORs together every criterion's per-row bool
    tensor and hands the caller only the combined result, so nothing in that
    protocol tells a caller afterward WHICH criterion fired. Recording it
    here is what lets ``QwenRecognizer._transcribe_chunk`` classify a
    ``Bounded`` decode from the criterion's OWN fact, rather than
    re-deriving "did the deadline fire" from a wall-clock measurement taken
    outside ``generate`` that could, in principle, disagree with what the
    criterion itself decided (a slow return from ``generate`` after the
    criterion already stopped it, for instance).

    Subclassing (not wrapping) ``MaxTimeCriteria`` keeps the calibrated
    timeout logic itself in ``transformers`` rather than reimplemented here;
    this class adds exactly the one bit ``transformers`` does not expose.
    """

    def __init__(self, max_time: float, initial_timestamp: float | None = None) -> None:
        from transformers.generation import (  # type: ignore[import-not-found]
            MaxTimeCriteria,
        )

        # Composition, not inheritance: `MaxTimeCriteria` lives behind a
        # lazy `transformers` import (see the module docstring on why torch
        # itself is imported lazily throughout this file), so a class body
        # cannot subclass it without importing `transformers` at module load
        # time. `self._inner` is that instance; every other member of this
        # class exists only to observe it.
        self._inner = MaxTimeCriteria(
            max_time=max_time, initial_timestamp=initial_timestamp
        )
        self.fired = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        is_done = self._inner(input_ids, scores, **kwargs)
        if bool(is_done.any()):
            self.fired = True
        return is_done


class QwenRecognizer:
    """Wrapper around Qwen3-ASR model invocation.

    The transcribe API returns a typed ``AsrGenerationPayload`` +
    timed-words list, matching the contract the BA3 worker protocol
    expects. Projection from per-utterance results to speaker-tagged
    monologues is done locally in Python (single-speaker for now;
    Qwen3-ASR doesn't emit speaker diarization).

    For multi-speaker audio, the downstream BA3 pipeline applies
    diarization separately: Qwen3-ASR's role here is the
    speech-to-character transcription only. This matches how
    BA3 currently uses other diarization-free ASR engines.
    """

    def __init__(
        self,
        lang: LanguageCode = "yue",
        model_id: str = "Qwen/Qwen3-ASR-1.7B-hf",
        device: str = "cpu",
    ) -> None:
        self.lang = lang
        self.model_id = model_id
        self.device = device
        self._model: LoadedQwen | None = None
        # Both resolved at construction so an unsupported language or a
        # stale checkpoint id surfaces at worker startup rather than on the
        # first inference request, minutes into a job.
        _check_native_checkpoint(model_id)
        self._qwen_language: QwenLanguageLabel = resolve_qwen_language(lang)

    def warm(self) -> None:
        """Force the lazy model load. Call at worker bootstrap so the
        first inference request sees a warm cache, not a ~3.4 GB
        download + load delay."""
        self._get_model()

    def _get_model(self) -> LoadedQwen:
        if self._model is not None:
            return self._model

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Qwen3-ASR engine dependency 'torch' is missing from this "
                "environment. Reinstall batchalign3."
            ) from exc

        if self.device == "cuda":
            dtype = torch.bfloat16
        elif self.device == "mps":
            # User opted into MPS via override; warn rather than raise because
            # the user is sovereign on device selection. Measured on this
            # engine 2026-08-20: MPS is SIGKILLed with no exception on any
            # file needing more than three chunks, and float16 on MPS
            # segfaults during load.
            L.warning(
                "Qwen3-ASR MPS device requested; measured 2026-08-20 to be "
                "unusable for this model (long audio is SIGKILLed with no "
                "exception; float16 segfaults at load). Prefer cpu or cuda."
            )
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._model = LoadedQwen.load(
            model_id=self.model_id,
            aligner_id=QWEN_FORCED_ALIGNER_MODEL_ID,
            device=self.device,
            dtype=dtype,
        )
        L.info(
            "Qwen3-ASR loaded: model=%s, aligner=%s, lang=%s (%s), device=%s, dtype=%s",
            self.model_id,
            QWEN_FORCED_ALIGNER_MODEL_ID,
            self.lang,
            self._qwen_language,
            self.device,
            dtype,
        )
        return self._model

    def _transcribe_chunk(
        self, loaded: LoadedQwen, chunk: AudioChunk, budget: ChunkDecodeBudget
    ) -> ChunkDecodeOutcome:
        """ASR one chunk against an already-built budget.

        Returns the typed decode outcome, window-scoped. ``budget`` is built
        by the caller (``_run_model``) from the request's ``DecodeBudget``,
        drawn down chunk by chunk, so file-level accounting is the caller's
        job, not this method's.

        Classified by a ``match`` over two independently-observed facts:
        whether ``_RecordingMaxTimeCriteria`` itself recorded firing (its
        OWN fact, not a wall-clock measurement re-derived out here that
        could disagree with it), and whether the returned token count
        reached ``budget.max_new_tokens``. The deadline takes priority when
        both are true: a decode that reached the cap AFTER the deadline had
        already fired is, first, a deadline overrun.
        """
        import torch  # type: ignore[import-not-found]
        from transformers.generation import (  # type: ignore[import-not-found]
            StoppingCriteriaList,
        )

        deadline_criterion = _RecordingMaxTimeCriteria(max_time=budget.deadline_seconds)
        stopping_criteria = StoppingCriteriaList([deadline_criterion])

        inputs = loaded.processor.apply_transcription_request(
            audio=chunk.samples,
            # The model accepts full language names as well as ISO codes, so
            # the existing label mapping carries over unchanged. We pass an
            # explicit label rather than letting it auto-detect: the caller
            # has already established the session language via ``@Languages``,
            # and auto-detect on short or low-energy audio mis-classifies.
            language=self._qwen_language,
        ).to(loaded.model.device, loaded.model.dtype)

        with torch.inference_mode():
            output_ids = loaded.model.generate(
                **inputs,
                max_new_tokens=budget.max_new_tokens,
                stopping_criteria=stopping_criteria,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = loaded.processor.decode(generated, return_format="parsed")[0]
        text = str(parsed.get("transcription", "") or "")

        hit_token_cap = generated.shape[1] >= budget.max_new_tokens
        match (deadline_criterion.fired, hit_token_cap):
            case (True, _):
                return Bounded(reason=BoundReason.DEADLINE, text=text)
            case (False, True):
                return Bounded(reason=BoundReason.TOKEN_CAP, text=text)
            case (False, False):
                return Completed(text=text)

    def _align_chunk(
        self, loaded: LoadedQwen, chunk: AudioChunk, text: str
    ) -> list[tuple[float, float, str]]:
        """Word timings for one chunk, converted into recording-relative time.

        The aligner call itself is shared with the standalone ``qwen3_fa``
        engine and lives in ``inference.qwen_forced_alignment``. What is NOT
        shared, and stays here, is the conversion: the aligner answers in
        window time (`WindowAlignedWord`), and only this caller holds the
        chunk offset that turns it into recording time. That conversion is the
        one thing in this file that is silently wrong when omitted, so it has
        exactly one site.

        The `SpokenWindowWords` step is the second thing this caller owns: it
        reads aligner units as WORDS, so a unit with no text is not one of its
        answers. The FA fold must keep every unit in position, which is why the
        filter cannot live in `align`.
        """
        alignment = loaded.aligner.align(chunk.samples, text, self._qwen_language)
        return [
            (chunk.to_file(word.start_s), chunk.to_file(word.end_s), word.text)
            for word in SpokenWindowWords.of_units(alignment.words).words
        ]

    def _run_model(
        self, source_path: str, decode_budget_seconds: float | None = None
    ) -> QwenTranscription:
        """Transcribe and align the whole recording, chunk by chunk.

        ``decode_budget_seconds``, when given, is the request's own
        file-level wall-clock decode budget (the Rust side will populate
        this from the same audio duration once the request carries a
        ``decode_budget_seconds`` field). Until then it defaults to
        ``None``, and this method derives one itself, from THIS file's own
        total audio seconds via ``DecodeBudget.for_total_seconds``, using
        the one named realtime factor the whole scheme is built on. Either
        way, exactly one ``DecodeBudget`` is built for the request and every
        chunk's deadline is drawn down from it (see ``ChunkDecodeBudget.
        for_chunk``), so no chunk computes its own deadline in isolation.
        """
        import librosa  # type: ignore[import-not-found]

        loaded = self._get_model()
        # A second audio-decode path in this repo, which already has one
        # (`inference.audio.load_audio`, soundfile-backed, used by the Aliyun
        # engine). Deliberate for now: `load_audio` does not resample, so
        # matching this single call means hand-composing load, mono-mix and a
        # torchaudio resampler, which is MORE code and a DIFFERENT resampler
        # than the one the engine's CER was measured under. Unify when
        # somebody re-measures, not before.
        wav, _sr = librosa.load(source_path, sr=SAMPLE_RATE, mono=True)
        chunks = split_audio_into_chunks(wav)

        remaining_budget = (
            DecodeBudget(seconds=decode_budget_seconds)
            if decode_budget_seconds is not None
            else DecodeBudget.for_total_seconds(sum(chunk.seconds for chunk in chunks))
        )

        segments: list[QwenAsrSegment] = []
        refused: list[RunawayOutput] = []
        for chunk in chunks:
            chunk_budget = ChunkDecodeBudget.for_chunk(chunk, remaining_budget)
            outcome = self._transcribe_chunk(loaded, chunk, chunk_budget)
            remaining_budget = remaining_budget.remaining_after(
                chunk_budget.deadline_seconds
            )

            # Classify BEFORE the blank-text check: a `Bounded` outcome is a
            # runaway regardless of how much text it produced, including
            # none at all (a chunk whose deadline fired before it emitted
            # anything). Checking blank-text first would silently swallow
            # exactly that case into "no speech" and never record which
            # bound fired.
            runaway = detect_runaway(outcome, chunk)
            if runaway is not None:
                L.warning("Qwen3-ASR discarding runaway chunk: %s", runaway.describe())
                refused.append(runaway)
                continue

            # Only reachable for a `Completed` outcome now: `detect_runaway`
            # already returned non-None for every `Bounded` one above.
            text = outcome.text
            if not text.strip():
                continue

            segments.append(
                QwenAsrSegment(
                    text=text,
                    word_timestamps=self._align_chunk(loaded, chunk, text),
                )
            )
        return QwenTranscription(segments=segments, refused=refused)

    def transcribe(
        self, source_path: str, decode_budget_seconds: float | None = None
    ) -> tuple[AsrGenerationPayload, list[TimedWord]]:
        """Run Qwen3-ASR on the audio file and return the shared
        ``(monologues_payload, timed_words)`` tuple.

        ``decode_budget_seconds`` forwards to ``_run_model``: see its
        docstring. ``None`` (the default, and the only value reachable
        today) means derive the budget from this file's own audio.

        Unlike the FunASR path, this projection is done entirely in
        Python (no Rust ``batchalign_core`` call). The Qwen3-ASR
        output structure is simple enough that a per-engine Rust
        projection isn't warranted yet; if the multi-speaker /
        diarization story changes in a future Qwen release we can
        consolidate into a shared Rust helper alongside the FunASR
        and Tencent projections.
        """
        result = self._run_model(
            source_path, decode_budget_seconds=decode_budget_seconds
        )
        if result.refused:
            # Surfaced here as well as at the discard site, because by this
            # point it is a property of the whole transcript: this many
            # seconds of audio reached the output as nothing at all.
            L.warning(
                "Qwen3-ASR produced no transcript for %d chunk(s), %.1fs of audio: %s",
                len(result.refused),
                result.refused_seconds,
                "; ".join(r.describe() for r in result.refused),
            )
        segments = result.segments

        elements: list[AsrElement] = []
        timed_words: list[TimedWord] = []
        for seg in segments:
            if seg.word_timestamps:
                for start_s, end_s, word in seg.word_timestamps:
                    # Pre-filter empty/whitespace tokens so downstream
                    # never sees them: saves allocation and matches
                    # the FunASR + Tencent shape.
                    if not word.strip():
                        continue
                    elements.append(
                        AsrElement(
                            type="text",
                            ts=start_s,
                            end_ts=end_s,
                            value=word,
                        )
                    )
                    timed_words.append(
                        TimedWord(
                            word=word,
                            start_ms=int(start_s * 1000),
                            end_ms=int(end_s * 1000),
                        )
                    )
            elif seg.text.strip():
                # Fall back to the whole-segment text when the model
                # didn't return per-word timestamps. CJK output is
                # tokenized per-character downstream; we leave that
                # to BA3's standard Cantonese tokenizer rather than
                # splitting here (avoid double-tokenization).
                elements.append(
                    AsrElement(
                        type="text",
                        ts=None,
                        end_ts=None,
                        value=seg.text,
                    )
                )

        monologues: list[AsrMonologue] = [AsrMonologue(speaker=0, elements=elements)]
        payload: AsrGenerationPayload = AsrGenerationPayload(monologues=monologues)
        return payload, timed_words
