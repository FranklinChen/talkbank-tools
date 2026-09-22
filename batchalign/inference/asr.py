"""ASR inference: audio -> raw tokens with timestamps.

Pure inference: returns raw engine-shaped ASR payloads for Rust post-processing.
No CHAT assembly, no number expansion, no retokenization.
"""

from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np
import pycountry
from pydantic import BaseModel, Field

from batchalign.inference._domain_types import (
    AudioPath,
    ConfidenceScore,
    LanguageCode,
    RevAiJobId,
    SampleRate,
    SpeakerId,
    TimestampSeconds,
    WhisperCpuPrecision,
)
from batchalign.worker._types_v2 import ProviderDiarizationV2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from batchalign.inference.types import WhisperASRHandle
    from batchalign.worker._types_v2 import WhisperChunkResultPayloadV2

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AsrBatchItem(BaseModel):
    """A single ASR inference request."""

    audio_path: AudioPath
    lang: LanguageCode = "eng"
    # Whether this provider must separate speakers, and into how many, carried
    # verbatim from the request's ``ProviderMediaInputV2.diarization``. NO
    # default: the two states answer a question only the caller can answer, and
    # a default answers it for everyone who forgets to.
    #
    # This was ``num_speakers: NumSpeakers = 1`` until 2026-09-16, and the same
    # boundary then carried two spellings of "do not separate": the Rust bridge
    # sent 0, while this default said 1, which under the ruling that a
    # diarization count is either automatic or at least two means "separate this
    # recording into one speaker", the contradiction submission refuses. The
    # default was unreachable in production only because Rust always sets the
    # key, which is a property of one caller rather than of the type.
    diarization: ProviderDiarizationV2
    rev_job_id: RevAiJobId | None = None
    # The request's own wall-clock decode budget, forwarded verbatim from
    # the Rust worker-protocol V2 request's ``decode_budget_seconds``
    # (`AsrRequestV2` in `crates/batchalign-types/src/worker_v2/requests.rs`).
    # ``None`` means Rust could not derive one (see that field's doc) and
    # this engine must derive its own from the file it is given, exactly
    # the pre-existing fallback path. Only the native Qwen3-ASR engine
    # reads this today (`_qwen_common.QwenRecognizer._run_model`); other
    # engines carry it unused.
    decode_budget_seconds: float | None = None


class AsrElement(BaseModel):
    """One raw ASR element in a speaker monologue payload."""

    value: str
    ts: TimestampSeconds | None = None
    end_ts: TimestampSeconds | None = None
    type: str = "text"
    confidence: ConfidenceScore | None = None


class AttributedSpeaker(BaseModel):
    """The provider named a speaker; this is its own label for them."""

    kind: Literal["attributed"] = "attributed"
    label: SpeakerId = Field(min_length=1)


class UndiarizedSpeaker(BaseModel):
    """The provider separates no speakers and named none.

    A claim about the ENGINE, not a guess about the recording. Aliyun, FunASR,
    Whisper and Qwen all report this; before it existed they reported speaker
    ``0``, which no downstream reader could tell from a provider's real first
    speaker.
    """

    kind: Literal["undiarized"] = "undiarized"


SpeakerAttribution: TypeAlias = Annotated[
    AttributedSpeaker | UndiarizedSpeaker,
    Field(discriminator="kind"),
]
"""Who a provider adapter attributed one monologue to."""


class AsrMonologue(BaseModel):
    """One speaker-attributed ASR span returned by a provider adapter.

    ``speaker`` has NO default. It used to default to ``0``, so an adapter that
    simply did not know invented a first speaker by omission.
    """

    speaker: SpeakerAttribution
    elements: list[AsrElement] = Field(default_factory=list)


class MonologueAsrResponse(BaseModel):
    """Tagged raw ASR payload built from speaker monologues."""

    kind: Literal["monologues"] = "monologues"
    lang: LanguageCode = "eng"
    monologues: list[AsrMonologue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Whisper pipeline output boundary models
# ---------------------------------------------------------------------------


class WhisperChunk(BaseModel):
    """One chunk from HuggingFace Whisper pipeline output."""

    text: str = ""
    timestamp: tuple[float | None, float | None] | list[float | None] = (None, None)


class WhisperChunksAsrResponse(BaseModel):
    """Tagged raw ASR payload for the local Whisper pipeline output."""

    kind: Literal["whisper_chunks"] = "whisper_chunks"
    lang: LanguageCode = "eng"
    text: str = ""
    chunks: list[WhisperChunk] = Field(default_factory=list)


def iso3_to_language_name(iso3: LanguageCode) -> str:
    """Convert ISO-639-3 language code to a Whisper language name.

    Raises ``ValueError`` if the code is not recognized by pycountry.
    Previously this silently fell back to ``"english"``, which caused
    wrong-language transcription with no warning, a regression from
    batchalign2 which used the same pycountry lookup but in a context
    where unrecognized codes would surface earlier.
    """
    special: dict[str, str] = {"yue": "Cantonese", "cmn": "chinese", "auto": "auto"}
    if iso3 in special:
        return special[iso3]
    lang_obj = pycountry.languages.get(alpha_3=iso3)
    if lang_obj is not None:
        return str(lang_obj.name).lower()
    raise ValueError(
        f"Unrecognized ISO 639-3 language code '{iso3}', pycountry has no "
        f"entry for this code. Whisper cannot determine the target language. "
        f"Check that the --lang value is a valid ISO 639-3 code."
    )


# ---------------------------------------------------------------------------
# Whisper ASR load/infer (replaces WhisperASRModel class)
# ---------------------------------------------------------------------------


def _cpu_torch_dtype(precision: WhisperCpuPrecision) -> Any:
    """The torch dtype for a CPU precision; exhaustive so a new member decides."""
    import torch

    match precision:
        case WhisperCpuPrecision.FLOAT32:
            return torch.float32
        case WhisperCpuPrecision.FLOAT16:
            return torch.float16
        case _:
            typing.assert_never(precision)


def load_whisper_asr(
    model: str = "openai/whisper-large-v3",
    base: str = "openai/whisper-large-v3",
    language: str = "english",
    target_sample_rate: SampleRate = 16000,
    *,
    device_policy=None,
    cpu_precision: WhisperCpuPrecision = WhisperCpuPrecision.FLOAT32,
) -> WhisperASRHandle:
    """Load a Whisper ASR pipeline. Returns a typed handle.

    ``cpu_precision`` applies only when the resolved device is a CPU; CUDA
    always loads float16.
    """
    import torch
    from transformers import (
        GenerationConfig,
        WhisperProcessor,
        WhisperTokenizer,
        pipeline,
    )

    from batchalign.device import resolve_inference_device
    from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
    from batchalign.inference.types import WhisperASRHandle

    device = resolve_inference_device(device_policy)

    # No download probe here. The worker resolves the pinned snapshot before
    # calling this function and passes the resolved DIRECTORY as both ``model``
    # and ``base``, so probing them asked whether a local path was a cached
    # repository: it never is, the notification fired on every load and named a
    # directory, and ``model != base`` could no longer be true, so the second
    # probe was dead. The announcement now lives in
    # ``worker._model_loading.asr._load_hub_member``, the call that actually
    # downloads and the only one that knows the revision it is fetching.

    config = GenerationConfig.from_pretrained(base)
    config.no_repeat_ngram_size = 4
    config.use_cache = True

    if language == "Cantonese":
        config.no_timestamps_token_id = 50363
        config.alignment_heads = [
            [5, 3],
            [5, 9],
            [8, 0],
            [8, 4],
            [8, 8],
            [9, 0],
            [9, 7],
            [9, 9],
            [10, 5],
        ]

    asr_dtype = (
        torch.float16 if device.type == "cuda" else _cpu_torch_dtype(cpu_precision)
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=WhisperTokenizer.from_pretrained(base),
        chunk_length_s=25,
        stride_length_s=3,
        device=device,
        torch_dtype=asr_dtype,
        return_timestamps=True,
    )
    bind_whisper_token_timestamp_extractor(pipe.model)
    pipe.model.eval()
    WhisperProcessor.from_pretrained(base)

    return WhisperASRHandle(
        pipe=pipe,
        config=config,
        lang=language,
        sample_rate=target_sample_rate,
    )


def _infer_whisper(
    model: WhisperASRHandle,
    item: AsrBatchItem,
) -> WhisperChunksAsrResponse:
    """Run local Whisper inference and return the pipeline's chunk payload.

    Calls the HuggingFace pipeline with the source path directly and extracts
    the raw chunk payload. That keeps the Python adapter closer to a pure model
    host: it does not decode or resample audio itself before inference.

    No turn assembly, no speaker interleaving, no punctuation splitting. Rust
    still owns all shared postprocessing after deserializing this tagged
    payload.
    """
    gen_kwargs = model.gen_kwargs(model.lang)

    raw = model(item.audio_path, batch_size=1, generate_kwargs=gen_kwargs)

    # Parse pipeline output at the boundary into typed model
    output = WhisperChunksAsrResponse.model_validate(
        {
            "kind": "whisper_chunks",
            "lang": item.lang,
            "text": raw.get("text", "") if isinstance(raw, dict) else "",
            "chunks": raw.get("chunks", []) if isinstance(raw, dict) else [],
        }
    )

    return output


def infer_whisper_prepared_audio(
    model: WhisperASRHandle,
    audio: np.ndarray,
    lang: LanguageCode,
) -> WhisperChunkResultPayloadV2:
    """Run local Whisper on Rust-prepared mono audio.

    This is the local-model V2 boundary for ASR. Rust owns media decoding and
    audio preparation, while Python just feeds the prepared waveform into the
    HuggingFace runtime and returns the raw chunk payload.
    """
    from batchalign.worker._types_v2 import (
        WhisperChunkResultPayloadV2,
        WhisperChunkSpanV2,
    )

    gen_kwargs = model.gen_kwargs(iso3_to_language_name(lang))
    raw = model(
        {
            "raw": np.asarray(audio, dtype=np.float32),
            "sampling_rate": model.sample_rate,
        },
        batch_size=1,
        generate_kwargs=gen_kwargs,
    )

    chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
    # Overlap at a seam and an inverted chunk travel as emitted: the Rust
    # consumer projects every producer's spans onto non-decreasing boundaries
    # in one place (`worker::chunk_spans`), so nothing is guessed here.
    with_span: list[WhisperChunkSpanV2] = []
    dropped_without_span = 0
    for chunk in chunks:
        # A missing bound is an ABSENCE, not a time. Substituting 0.0 here used
        # to satisfy `WhisperChunkSpanV2`'s non-optional fields by inventing a
        # value: a chunk missing only its start then claimed to run from the
        # beginning of the audio to its real end, and passed every downstream
        # check because both numbers were real and correctly ordered. It also
        # reached the Rust boundary as `Some(0.0)` rather than `None`, so the
        # filter there could not see it either. Note 0.0 is a legal start, so
        # `or 0.0` could not distinguish the two cases even in principle.
        timestamp = chunk.get("timestamp") or [None, None]
        raw_start, raw_end = timestamp[0], timestamp[1]
        if raw_start is None or raw_end is None:
            dropped_without_span += 1
            continue
        with_span.append(
            WhisperChunkSpanV2(
                text=str(chunk.get("text", "")),
                start_s=float(raw_start),
                end_s=float(raw_end),
            )
        )
    if dropped_without_span:
        logger.warning(
            "Dropped %d Whisper chunk(s) with no usable timestamp span "
            "(kept %d); these carry no placeable timing",
            dropped_without_span,
            len(with_span),
        )
    # The handle is the honest witness: it is the object the loader resolved a
    # snapshot into, so it knows which commit actually came off the hub. A
    # handle that never got one cannot have its output attributed, and guessing
    # the configured checkpoint here would record a model that may not be the
    # one on disk, which is the exact substitution this workstream removes.
    identity = model.model_identity
    if identity is None:
        raise RuntimeError(
            "this Whisper handle was never told which checkpoint it loaded, so "
            "its results cannot name their model. The worker ASR loader sets "
            "`model_identity` as soon as it resolves the snapshot; a handle "
            "built outside that path must do the same before inference."
        )
    return WhisperChunkResultPayloadV2(
        lang=lang,
        text=raw.get("text", "") if isinstance(raw, dict) else "",
        chunks=with_span,
        model=identity,
    )
