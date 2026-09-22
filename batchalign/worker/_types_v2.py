"""Typed schema models for worker protocol V2.

These models mirror `crates/batchalign-types/src/worker_v2/` (re-exported via
`crates/batchalign/src/types/worker_v2.rs`). The ``*_v2`` namespace is
intentional: Rust and Python still ship the frozen V1 worker surface
(``worker`` / ``_types.py``), while ``worker_v2`` is the live typed execute
contract, checked against the JSON Schema layer (``ipc-schema/worker_v2``).

- define the canonical V2 protocol shape in Python
- validate canonical fixture files shared with Rust
- prevent drift while V1 and V2 coexist

The design goal is to make Python a thin model host. Request and response
models therefore describe model-ready inputs plus prepared-artifact references,
not CLI commands or document-processing workflows.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, FiniteFloat, StringConstraints, model_validator

from batchalign.inference._domain_types import (
    LanguageCode,
    NumSpeakers,
    SpeakerId,
    TranslationBackend,
)
from batchalign.worker._types import ReportedEngineName, WorkerJSONValue

WorkerRequestIdV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Stable identifier for one V2 protocol request/response pair."""

WorkerArtifactIdV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Stable identifier for one prepared worker artifact."""

WorkerArtifactPathV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Filesystem path to a prepared worker artifact."""

ProtocolVersionV2: TypeAlias = Annotated[int, Field(ge=2)]
"""Worker protocol major version."""

FiniteNonNegativeFloat: TypeAlias = Annotated[FiniteFloat, Field(ge=0)]
"""Finite floating-point value constrained to be non-negative."""


class WorkerKindV2(str, Enum):
    """Worker role selected during the V2 handshake."""

    INFER = "infer"


class InferenceTaskV2(str, Enum):
    """High-level V2 task family."""

    MORPHOSYNTAX = "morphosyntax"
    UTSEG = "utseg"
    TRANSLATE = "translate"
    COREF = "coref"
    ASR = "asr"
    FORCED_ALIGNMENT = "forced_alignment"
    SPEAKER = "speaker"
    SPEAKER_EMBEDDING = "speaker_embedding"
    OPENSMILE = "opensmile"
    AVQI = "avqi"


class AsrBackendV2(str, Enum):
    """ASR backend selected by Rust."""

    LOCAL_WHISPER = "local_whisper"
    # HuggingFace Whisper fine-tune selected by model_id. Same worker-side
    # runtime shape as LOCAL_WHISPER, both host a ``WhisperASRHandle``,
    # but a distinct backend variant so the control-plane pool key and the
    # worker's engine dispatch select the fine-tune loader at bootstrap.
    WHISPER_HUB = "whisper_hub"
    HK_TENCENT = "hk_tencent"
    HK_ALIYUN = "hk_aliyun"
    HK_FUNAUDIO = "hk_funaudio"
    # Qwen3-ASR Cantonese provider (local model via qwen-asr package).
    # Mirrors `crates/batchalign-types/src/worker_v2/requests.rs::AsrBackendV2::HkQwen`.
    HK_QWEN = "hk_qwen"
    REVAI = "revai"


class FaBackendV2(str, Enum):
    """Forced-alignment backend selected by Rust."""

    WHISPER = "whisper"
    WAVE2VEC = "wave2vec"
    WAV2VEC_CANTO = "wav2vec_canto"
    # Mirrors `crates/batchalign-types/src/worker_v2/requests.rs::FaBackendV2::Qwen3`.
    QWEN3 = "qwen3"


class SpeakerBackendV2(str, Enum):
    """Speaker backend selected by Rust."""

    PYANNOTE_AI = "pyannote_ai"
    PYANNOTE = "pyannote"
    NEMO = "nemo"


class SpeakerEmbeddingBackendV2(str, Enum):
    """Speaker-embedding backend selected by Rust.

    One variant, and still an enum: a vector's meaning depends on which
    acoustic model produced it, so the wire has to say.
    """

    PYANNOTE = "pyannote"


class WorkerAttachmentKindV2(str, Enum):
    """Small attachment vocabulary advertised in capabilities."""

    PREPARED_AUDIO = "prepared_audio"
    PREPARED_TEXT = "prepared_text"
    INLINE_JSON = "inline_json"
    PROVIDER_MEDIA = "provider_media"
    SUBMITTED_JOB = "submitted_job"


class PreparedAudioEncodingV2(str, Enum):
    """PCM encoding used for prepared audio artifacts."""

    PCM_F32LE = "pcm_f32le"


class PreparedTextEncodingV2(str, Enum):
    """Encoding used for prepared text artifacts."""

    UTF8_JSON = "utf8_json"


class ProtocolErrorCodeV2(str, Enum):
    """Error category for protocol-level failures."""

    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    INVALID_PAYLOAD = "invalid_payload"
    MISSING_ATTACHMENT = "missing_attachment"
    ATTACHMENT_UNREADABLE = "attachment_unreadable"
    MODEL_UNAVAILABLE = "model_unavailable"
    RUNTIME_FAILURE = "runtime_failure"
    # A pinned Hugging Face Hub artifact refused this machine's request: a
    # gated repository requiring accepted terms, a missing/invalid token, or
    # no cached copy while offline. Distinct from RUNTIME_FAILURE so the
    # server can categorize it as a configuration/credential condition on the
    # operator's machine rather than a batchalign defect. Set by the Rust
    # worker boundary when a Python runner call raises
    # `batchalign.inference._model_access_errors.ModelAccessDeniedError`.
    MODEL_ACCESS_DENIED = "model_access_denied"


class FaTextModeV2(str, Enum):
    """Text-joining mode for forced-alignment payloads."""

    SPACE_JOINED = "space_joined"
    CHAR_JOINED = "char_joined"
    #: Space-joined, then every character separated by a space. Selected by
    #: `--pauses`; Rust applies it, so the host receives shaped text.
    CHAR_SPACED = "char_spaced"


class WorkerRuntimeInfoV2(BaseModel):
    """Runtime information returned during the V2 handshake."""

    python_version: str
    free_threaded: bool


class HelloRequestV2(BaseModel):
    """Initial handshake request sent by Rust."""

    protocol_version: ProtocolVersionV2
    worker_kind: WorkerKindV2


class HelloResponseV2(BaseModel):
    """Initial handshake response sent by Python."""

    protocol_version: ProtocolVersionV2
    worker_pid: int
    runtime: WorkerRuntimeInfoV2


class PreparedAudioRefV2(BaseModel):
    """File-backed prepared audio artifact."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    id: WorkerArtifactIdV2
    path: WorkerArtifactPathV2
    encoding: PreparedAudioEncodingV2
    channels: int = Field(ge=1)
    sample_rate_hz: int = Field(ge=1)
    frame_count: int = Field(ge=0)
    byte_offset: int = Field(ge=0)
    byte_len: int = Field(ge=0)


class PreparedTextRefV2(BaseModel):
    """File-backed prepared text artifact."""

    kind: Literal["prepared_text"] = "prepared_text"
    id: WorkerArtifactIdV2
    path: WorkerArtifactPathV2
    encoding: PreparedTextEncodingV2
    byte_offset: int = Field(ge=0)
    byte_len: int = Field(ge=0)


class InlineJsonRefV2(BaseModel):
    """Small inline JSON attachment."""

    kind: Literal["inline_json"] = "inline_json"
    id: WorkerArtifactIdV2
    value: WorkerJSONValue


ArtifactRefV2: TypeAlias = Annotated[
    PreparedAudioRefV2 | PreparedTextRefV2 | InlineJsonRefV2,
    Field(discriminator="kind"),
]
"""Prepared artifact descriptor carried alongside an execute request."""


class PreparedAudioInputV2(BaseModel):
    """Reference to a prepared audio attachment (internally tagged)."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    audio_ref_id: WorkerArtifactIdV2


class NotRequestedDiarizationV2(BaseModel):
    """One track expected; the provider must not separate speakers."""

    kind: Literal["not_requested"] = "not_requested"


class IntegratedDiarizationV2(BaseModel):
    """The provider separates speakers itself, into this many."""

    kind: Literal["integrated"] = "integrated"
    speakers: NumSpeakers


ProviderDiarizationV2: TypeAlias = Annotated[
    NotRequestedDiarizationV2 | IntegratedDiarizationV2,
    Field(discriminator="kind"),
]
"""Whether a provider is asked to separate speakers, and into how many.

Replaces a bare ``num_speakers``, which could not say the one thing the bridge
needs when a provider returns a monologue with no speaker: whether separation
was ASKED FOR. Without that, an absent label had to be given a number, and the
number given was zero.
"""


class ProviderMediaInputV2(BaseModel):
    """Temporary cloud-provider media input (internally tagged)."""

    kind: Literal["provider_media"] = "provider_media"
    media_path: WorkerArtifactPathV2
    diarization: ProviderDiarizationV2


class SubmittedJobInputV2(BaseModel):
    """Previously submitted provider job id (internally tagged)."""

    kind: Literal["submitted_job"] = "submitted_job"
    provider_job_id: WorkerArtifactIdV2


# Backward-compatible aliases for ASR input wrappers.
PreparedAudioAsrInputV2 = PreparedAudioInputV2
ProviderMediaAsrInputV2 = ProviderMediaInputV2
SubmittedJobAsrInputV2 = SubmittedJobInputV2

AsrInputV2: TypeAlias = Annotated[
    PreparedAudioInputV2 | ProviderMediaInputV2 | SubmittedJobInputV2,
    Field(discriminator="kind"),
]
"""Backend-specific ASR input transport (internally tagged on ``kind``)."""


class AsrRequestV2(BaseModel):
    """V2 ASR request payload (internally tagged as ``"asr"``)."""

    kind: Literal["asr"] = "asr"
    lang: LanguageCode
    backend: AsrBackendV2
    input: AsrInputV2
    # The models this request pins, chosen by the Rust control plane. The
    # worker loads exactly these and reports back what it observed; the bridge
    # refuses a disagreement. Annotated as a forward reference because the
    # identity mirror is defined further down this module, next to the loaded
    # half it pairs with.
    models: AsrRequestedModelsV2
    # Per-engine configuration extras (qwen_model, funaudio_model, ...),
    # carried verbatim from --engine-overrides. Rust omits the field from
    # the wire when empty (serde skip_serializing_if), so None here means
    # "absent", matching the schema-generated model. Python engines
    # currently read these knobs from the worker spawn argv; the field
    # exists for wire conformance, not dispatch.
    extras: dict[str, str] | None = None
    # The request's own wall-clock decode budget in seconds, derived by
    # Rust from the audio's duration (see `DecodeBudgetSeconds` in
    # `crates/batchalign-types/src/worker_v2/requests.rs`). ``None`` means
    # Rust could not derive one (a provider-media request whose duration
    # could not be probed); the receiving engine then derives its own from
    # the file it is given, exactly the pre-existing fallback.
    decode_budget_seconds: float | None = None


class ForcedAlignmentRequestV2(BaseModel):
    """V2 forced-alignment request payload (internally tagged as ``"forced_alignment"``)."""

    kind: Literal["forced_alignment"] = "forced_alignment"
    backend: FaBackendV2
    payload_ref_id: WorkerArtifactIdV2
    audio_ref_id: WorkerArtifactIdV2
    text_mode: FaTextModeV2


class MorphosyntaxRequestV2(BaseModel):
    """V2 morphosyntax request payload (internally tagged as ``"morphosyntax"``)."""

    kind: Literal["morphosyntax"] = "morphosyntax"
    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)
    retokenize: bool = False


class UtsegRequestV2(BaseModel):
    """V2 utterance-segmentation request payload (internally tagged as ``"utseg"``)."""

    kind: Literal["utseg"] = "utseg"
    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)
    # Operator opt-in to the legacy Stanza constituency-parser
    # fallback for unsupported languages. Surfaced as
    # `--utseg-fallback-stanza` on the CLI. Defaults to `False` so a
    # request from an older client that omits the field deserializes
    # with the safe "refuse" behavior; replaces the previous
    # `BA3_UTSEG_FALLBACK_STANZA` env-var control flow.
    allow_stanza_fallback: bool = False


class TranslateRequestV2(BaseModel):
    """V2 translation request payload (internally tagged as ``"translate"``)."""

    kind: Literal["translate"] = "translate"
    source_lang: LanguageCode
    target_lang: LanguageCode
    # The job's engine; keys the worker that serves the request. Mirrors
    # `crates/batchalign-types/src/worker_v2/requests.rs::TranslateBackendV2`.
    engine: TranslationBackend
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class CorefRequestV2(BaseModel):
    """V2 coreference request payload (internally tagged as ``"coref"``)."""

    kind: Literal["coref"] = "coref"
    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class SpeakerRequestV2(BaseModel):
    """V2 speaker diarization request payload (internally tagged as ``"speaker"``)."""

    kind: Literal["speaker"] = "speaker"
    backend: SpeakerBackendV2
    input: SpeakerInputV2
    expected_speakers: NumSpeakers | None = None


class SpeakerEmbeddingSpanV2(BaseModel):
    """One span to embed, in frames of the prepared recording."""

    span_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> SpeakerEmbeddingSpanV2:
        if self.end_frame < self.start_frame:
            raise ValueError("Speaker embedding span end_frame must be >= start_frame")
        return self


class SpeakerEmbeddingRequestV2(BaseModel):
    """V2 speaker-embedding request payload (tagged ``"speaker_embedding"``)."""

    kind: Literal["speaker_embedding"] = "speaker_embedding"
    backend: SpeakerEmbeddingBackendV2
    audio_ref_id: WorkerArtifactIdV2
    spans: list[SpeakerEmbeddingSpanV2]


class OpenSmileRequestV2(BaseModel):
    """V2 openSMILE request payload (internally tagged as ``"opensmile"``)."""

    kind: Literal["opensmile"] = "opensmile"
    audio_ref_id: WorkerArtifactIdV2
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "functionals"


class AvqiRequestV2(BaseModel):
    """V2 AVQI request payload (internally tagged as ``"avqi"``)."""

    kind: Literal["avqi"] = "avqi"
    cs_audio_ref_id: WorkerArtifactIdV2
    sv_audio_ref_id: WorkerArtifactIdV2


class SpeakerPreparedAudioInputV2(BaseModel):
    """Prepared-audio speaker input owned by Rust (internally tagged)."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    audio_ref_id: WorkerArtifactIdV2


# Backward-compatible alias.
SpeakerPreparedAudioRefInputV2 = SpeakerPreparedAudioInputV2

SpeakerInputV2: TypeAlias = Annotated[
    SpeakerPreparedAudioInputV2,
    Field(discriminator="kind"),
]
"""Speaker input transport (internally tagged on ``kind``)."""


TaskRequestV2: TypeAlias = Annotated[
    AsrRequestV2
    | ForcedAlignmentRequestV2
    | MorphosyntaxRequestV2
    | UtsegRequestV2
    | TranslateRequestV2
    | CorefRequestV2
    | SpeakerRequestV2
    | SpeakerEmbeddingRequestV2
    | OpenSmileRequestV2
    | AvqiRequestV2,
    Field(discriminator="kind"),
]
"""Typed execute request payload (internally tagged on ``kind``)."""


class ExecuteRequestV2(BaseModel):
    """One top-level V2 execution request."""

    request_id: WorkerRequestIdV2
    task: InferenceTaskV2
    payload: TaskRequestV2
    attachments: list[ArtifactRefV2]


# ---------------------------------------------------------------------------
# Pinned ASR model identity
#
# The Python mirror of `crates/batchalign-types/src/worker_v2/asr_model.rs`.
# Rust resolves which models a worker must load and sends them with the spawn;
# the worker loads exactly those revisions and reports back what it loaded, and
# the bridge refuses a disagreement. Two hand-written parsers only agree while
# something pins the bytes: the Rust side pins them in
# `model_manifest::tests::the_pinned_composition_serializes_in_the_shape_python_parses`.
# ---------------------------------------------------------------------------

HubCommitV2: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
"""An exact hub commit: 40 lowercase hexadecimal characters."""

ContentDigestV2: TypeAlias = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{64}$")
]
"""A SHA-256 content digest: 64 lowercase hexadecimal characters."""


class RequestedCommitV2(BaseModel):
    """Load exactly this hub commit."""

    kind: Literal["commit"] = "commit"
    commit: HubCommitV2


class RequestedTagV2(BaseModel):
    """Load exactly this published tag. Never recorded as a commit."""

    kind: Literal["tag"] = "tag"
    tag: str


class RequestedContentDigestV2(BaseModel):
    """Load the file with exactly these contents."""

    kind: Literal["content_digest"] = "content_digest"
    digest: ContentDigestV2


class RequestedProviderParameterV2(BaseModel):
    """Select the provider's model with this parameter."""

    kind: Literal["provider_parameter"] = "provider_parameter"
    parameter: str


class RequestedUnpinnedV2(BaseModel):
    """The plan could not pin this model.

    The worker loads the hub default and MUST report the commit it observed;
    it may never substitute a name for a revision.
    """

    kind: Literal["unpinned"] = "unpinned"


RequestedRevisionV2: TypeAlias = Annotated[
    RequestedCommitV2
    | RequestedTagV2
    | RequestedContentDigestV2
    | RequestedProviderParameterV2
    | RequestedUnpinnedV2,
    Field(discriminator="kind"),
]
"""What the plan asked a worker to load."""


class ObservedCommitV2(BaseModel):
    """The runtime reported the hub commit it loaded."""

    kind: Literal["commit"] = "commit"
    commit: HubCommitV2


class ObservedContentDigestV2(BaseModel):
    """The runtime loaded a file with these contents."""

    kind: Literal["content_digest"] = "content_digest"
    digest: ContentDigestV2


class ObservedNotExposedV2(BaseModel):
    """The runtime exposes no revision at all.

    True of every ModelScope loader and every cloud provider. Provider-side
    drift is invisible, and this variant says so rather than inventing a value.
    """

    kind: Literal["not_exposed"] = "not_exposed"


ObservedRevisionV2: TypeAlias = Annotated[
    ObservedCommitV2 | ObservedContentDigestV2 | ObservedNotExposedV2,
    Field(discriminator="kind"),
]
"""What the runtime actually loaded, as the worker observed it."""


class RequestedModelV2(BaseModel):
    """One model the plan pinned, before anything loaded it."""

    id: str
    revision: RequestedRevisionV2


class LoadedModelV2(BaseModel):
    """One model a worker loaded: what was asked for, and what was seen."""

    id: str
    requested: RequestedRevisionV2
    observed: ObservedRevisionV2


class WhisperModelsV2(BaseModel):
    """Hugging Face Whisper, stock or fine-tune."""

    engine: Literal["whisper"] = "whisper"
    asr: RequestedModelV2


class QwenModelsV2(BaseModel):
    """Qwen3-ASR and the forced aligner it needs for word timings."""

    engine: Literal["qwen"] = "qwen"
    asr: RequestedModelV2
    aligner: RequestedModelV2


class ParaformerModelsV2(BaseModel):
    """FunASR Paraformer, with its voice-activity and punctuation models."""

    engine: Literal["paraformer"] = "paraformer"
    asr: RequestedModelV2
    vad: RequestedModelV2
    punc: RequestedModelV2


class SenseVoiceModelsV2(BaseModel):
    """FunASR SenseVoice, with its voice-activity model."""

    engine: Literal["sense_voice"] = "sense_voice"
    asr: RequestedModelV2
    vad: RequestedModelV2


# No requested composition for in-process whisper.cpp. It is not a worker
# backend, so no request can select it and nothing ever built one; it appears
# on the LOADED side alone, as ``NativeWhisperModelIdentityV2``, which the
# in-process path reports for the ggml file it read.


class TencentModelsV2(BaseModel):
    """Tencent cloud ASR."""

    engine: Literal["tencent"] = "tencent"
    engine_model_type: RequestedModelV2


class AliyunModelsV2(BaseModel):
    """Aliyun cloud ASR."""

    engine: Literal["aliyun"] = "aliyun"
    service: RequestedModelV2


class RevModelsV2(BaseModel):
    """Rev.AI cloud ASR."""

    engine: Literal["rev"] = "rev"
    provider: RequestedModelV2


AsrRequestedModelsV2: TypeAlias = Annotated[
    WhisperModelsV2
    | QwenModelsV2
    | ParaformerModelsV2
    | SenseVoiceModelsV2
    | TencentModelsV2
    | AliyunModelsV2
    | RevModelsV2,
    Field(discriminator="engine"),
]
"""Every model of one engine's composition, as the plan requested them."""


class WhisperModelIdentityV2(BaseModel):
    """What a Whisper worker loaded."""

    engine: Literal["whisper"] = "whisper"
    asr: LoadedModelV2


class QwenModelIdentityV2(BaseModel):
    """What a Qwen3-ASR worker loaded, aligner included."""

    engine: Literal["qwen"] = "qwen"
    asr: LoadedModelV2
    aligner: LoadedModelV2


class ParaformerModelIdentityV2(BaseModel):
    """What a Paraformer worker loaded."""

    engine: Literal["paraformer"] = "paraformer"
    asr: LoadedModelV2
    vad: LoadedModelV2
    punc: LoadedModelV2


class SenseVoiceModelIdentityV2(BaseModel):
    """What a SenseVoice worker loaded."""

    engine: Literal["sense_voice"] = "sense_voice"
    asr: LoadedModelV2
    vad: LoadedModelV2


class NativeWhisperModelIdentityV2(BaseModel):
    """What the in-process whisper.cpp path loaded."""

    engine: Literal["native_whisper"] = "native_whisper"
    ggml: LoadedModelV2


class TencentModelIdentityV2(BaseModel):
    """The Tencent engine-model type the request named."""

    engine: Literal["tencent"] = "tencent"
    engine_model_type: LoadedModelV2


class AliyunModelIdentityV2(BaseModel):
    """The Aliyun service identity."""

    engine: Literal["aliyun"] = "aliyun"
    service: LoadedModelV2


class RevModelIdentityV2(BaseModel):
    """The Rev.AI provider constant."""

    engine: Literal["rev"] = "rev"
    provider: LoadedModelV2


AsrModelIdentityV2: TypeAlias = Annotated[
    WhisperModelIdentityV2
    | QwenModelIdentityV2
    | ParaformerModelIdentityV2
    | SenseVoiceModelIdentityV2
    | NativeWhisperModelIdentityV2
    | TencentModelIdentityV2
    | AliyunModelIdentityV2
    | RevModelIdentityV2,
    Field(discriminator="engine"),
]
"""Every model a worker loaded for one ASR response."""


# `AsrRequestV2`, and `ExecuteRequestV2` which embeds it, are defined ABOVE this
# mirror, so their annotation of `AsrRequestedModelsV2` was still an unresolved
# forward reference when those classes were built. Pydantic would rebuild them
# lazily on first validation, which happens to work, but until something
# triggers it `__pydantic_complete__` stays False: schema generation on an
# incomplete model then fails for a reason that has nothing to do with the
# caller who asked. Resolving them HERE, at the first point where the name
# exists, makes completeness a property of importing this module rather than of
# whatever the program happens to do first.
AsrRequestV2.model_rebuild()
ExecuteRequestV2.model_rebuild()


class WhisperChunkSpanV2(BaseModel):
    """One raw Whisper chunk span as the producer emitted it.

    Raw on purpose: chunks may overlap at a seam or arrive inverted, and the
    Rust consumer settles every producer's spans in one place
    (`batchalign::worker::chunk_spans::MonotoneChunkSpans`). The only
    invariant the wire carries is that a bound, when present, is a finite
    non-negative duration. A bound is ``None`` when the model predicted no
    timestamp for it; the words still travel, untimed.
    """

    text: str
    start_s: FiniteNonNegativeFloat | None = None
    end_s: FiniteNonNegativeFloat | None = None


class WhisperChunkResultPayloadV2(BaseModel):
    """Raw Whisper chunk output returned by Python (internally tagged)."""

    kind: Literal["whisper_chunk_result"] = "whisper_chunk_result"
    lang: LanguageCode
    text: str
    chunks: list[WhisperChunkSpanV2]
    # Which models produced this text, as the runtime observed them. Required,
    # matching Rust: a result that cannot name its models cannot be stamped or
    # cached honestly, and an optional field would let a producer forget.
    model: AsrModelIdentityV2


class AsrElementKindV2(str, Enum):
    """Stable vocabulary for one monologue element returned by ASR."""

    TEXT = "text"
    PUNCTUATION = "punctuation"


class AsrElementV2(BaseModel):
    """One raw ASR element inside a speaker monologue."""

    value: str
    start_s: FiniteNonNegativeFloat | None = None
    end_s: FiniteNonNegativeFloat | None = None
    kind: AsrElementKindV2
    confidence: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> AsrElementV2:
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s < self.start_s
        ):
            raise ValueError("ASR element end_s must be >= start_s")
        return self


class AttributedSpeakerV2(BaseModel):
    """The provider named a speaker; this is its own label for them."""

    kind: Literal["attributed"] = "attributed"
    label: Annotated[str, StringConstraints(min_length=1)]


class UndiarizedSpeakerV2(BaseModel):
    """The provider separates no speakers and named none."""

    kind: Literal["undiarized"] = "undiarized"


SpeakerAttributionV2: TypeAlias = Annotated[
    AttributedSpeakerV2 | UndiarizedSpeakerV2,
    Field(discriminator="kind"),
]
"""Who a provider attributed one monologue to.

Two states, because there are two facts, and the wire used to have a spelling
for only one. ``speaker`` was a bare string, so an engine that separates no
speakers at all had to write something, and what it wrote was ``"0"``:
indistinguishable from a provider's real first speaker, and downstream that
became a ``PAR0`` tier either way.
"""


class AsrMonologueV2(BaseModel):
    """One speaker-attributed monologue returned by a provider backend."""

    speaker: SpeakerAttributionV2
    elements: list[AsrElementV2]


class MonologueAsrResultPayloadV2(BaseModel):
    """Provider-shaped ASR output returned as speaker monologues (internally tagged)."""

    kind: Literal["monologue_asr_result"] = "monologue_asr_result"
    lang: LanguageCode
    monologues: list[AsrMonologueV2]
    # Required for the same reason as on the Whisper payload. For a cloud
    # provider this names the service and the parameter it was called with,
    # never an account credential.
    model: AsrModelIdentityV2


class WhisperTokenTimingV2(BaseModel):
    """One raw Whisper forced-alignment token onset."""

    text: str
    time_s: FiniteNonNegativeFloat


class WhisperTokenTimingResultPayloadV2(BaseModel):
    """Raw Whisper forced-alignment token output (internally tagged)."""

    kind: Literal["whisper_token_timing_result"] = "whisper_token_timing_result"
    tokens: list[WhisperTokenTimingV2]


class IndexedWordTimingV2(BaseModel):
    """One word-level timing result."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> IndexedWordTimingV2:
        if self.end_ms < self.start_ms:
            raise ValueError("Indexed word timing end_ms must be >= start_ms")
        return self


class IndexedWordTimingResultPayloadV2(BaseModel):
    """Forced-alignment indexed timing output (internally tagged)."""

    kind: Literal["indexed_word_timing_result"] = "indexed_word_timing_result"
    indexed_timings: list[IndexedWordTimingV2 | None]


class MorphosyntaxPipelineV2(str, Enum):
    """Which morphosyntax pipeline variant analyzed an item.

    The same Stanza version and language can run three different procedures,
    and their output differs, so provenance names the variant.
    """

    STANDARD = "standard"
    """Stanza over the words Rust sent, as sent."""
    MANDARIN_RETOKENIZE = "mandarin_retokenize"
    """Stanza's neural tokenizer re-segmented a Mandarin utterance."""
    CANTONESE_PYCANTONESE_POS = "cantonese_pycantonese_pos"
    """Stanza's parse with PyCantonese part-of-speech tags for Cantonese."""


class MorphosyntaxModelIdentityV2(BaseModel):
    """The Stanza model that analyzed one morphosyntax item."""

    stanza_version: ReportedEngineName
    lang: LanguageCode
    pipeline: MorphosyntaxPipelineV2


class UdRelationRepairKindV2(str, Enum):
    """Why a relation a worker received from Stanza is not the one it applied.

    Closed on both sides of the boundary: a rewrite must be named here before
    a worker can report it, so no rewrite reaches a transcript under a name
    the reader does not know.
    """

    PAD_RELATION = "pad_relation"
    """A padding label (``<PAD>``, ``<UNK>``), which is no relation at all."""
    RELATION_CASE = "relation_case"
    """A UD relation in the wrong case (``NSUBJ``), lowercased."""
    RELATION_ALIAS = "relation_alias"
    """A known non-UD spelling of a UD relation (``iob`` for ``iobj``)."""
    UNKNOWN_RELATION = "unknown_relation"
    """No UD relation and no known equivalent, degraded to ``dep``."""


class UdRelationRepairV2(BaseModel):
    """One relation rewrite a worker made inside an analysis it returned."""

    kind: UdRelationRepairKindV2
    word: str
    from_relation: str
    to_relation: str


class MorphosyntaxAnalyzedItemV2(BaseModel):
    """An item Stanza analyzed, with the model that analyzed it.

    ``repairs`` is required, with no default, for the reason ``model`` is: a
    default would make "this worker repaired nothing" and "this worker does
    not report repairs" the same value, which is the shape that let a rewrite
    live only in a log line.
    """

    kind: Literal["analyzed"] = "analyzed"
    raw_sentences: list[WorkerJSONValue]
    model: MorphosyntaxModelIdentityV2
    repairs: list[UdRelationRepairV2]


class MorphosyntaxNoWordsItemV2(BaseModel):
    """An utterance with no words: it has no morphology, and no model ran.

    Carries no identity on purpose. The producer used to attach one, naming a
    model that never saw the item.
    """

    kind: Literal["no_words"] = "no_words"


class MorphosyntaxFailedItemV2(BaseModel):
    """An item that could not be analyzed, with the reason."""

    kind: Literal["failed"] = "failed"
    error: str


MorphosyntaxItemResultV2: TypeAlias = Annotated[
    MorphosyntaxAnalyzedItemV2 | MorphosyntaxNoWordsItemV2 | MorphosyntaxFailedItemV2,
    Field(discriminator="kind"),
]
"""One morphosyntax item outcome (internally tagged on ``kind``).

A union rather than three optional fields: an analysis without its model, or
a result that is both an analysis and an error, is no longer a value this
type can hold, so neither the bridge nor the server checks for it.
"""


class MorphosyntaxResultPayloadV2(BaseModel):
    """Batched morphosyntax response payload (internally tagged)."""

    kind: Literal["morphosyntax_result"] = "morphosyntax_result"
    items: list[MorphosyntaxItemResultV2]


class UtsegBoundaryActionV2(str, Enum):
    """Closed semantic labels emitted by the utterance-boundary model."""

    ORDINARY = "ordinary"
    CAPITALIZED_ONSET = "capitalized_onset"
    PERIOD_BOUNDARY = "period_boundary"
    QUESTION_BOUNDARY = "question_boundary"
    EXCLAMATION_BOUNDARY = "exclamation_boundary"
    COMMA = "comma"


class UtsegNormalizationRevisionV2(str, Enum):
    """Closed normalization semantics for boundary-model evidence."""

    LOWER_STRIP_ASCII_PUNCTUATION_V1 = "lower-strip-ascii-punctuation-v1"


class UtsegAdjacencyPolicyRevisionV2(str, Enum):
    """Closed raw-to-applied boundary-action policy revisions."""

    SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1 = (
        "suppress-earlier-adjacent-nonordinary-v1"
    )
    SUPPRESS_EARLIER_ADJACENT_BOUNDARIES_V1 = "suppress-earlier-adjacent-boundaries-v1"


BoundaryProbabilityMicrosV2: TypeAlias = Annotated[int, Field(ge=0, le=1_000_000)]
"""Fixed-point boundary probability in the inclusive unit interval."""


class UtsegClassifiedBoundaryEvidenceV2(BaseModel):
    """Model evidence for one input word that reached classification."""

    kind: Literal["classified"] = "classified"
    raw_action: UtsegBoundaryActionV2
    applied_action: UtsegBoundaryActionV2
    boundary_probability_micros: BoundaryProbabilityMicrosV2


class UtsegNormalizationOmissionV2(BaseModel):
    """Evidence that normalization removed one word before classification."""

    kind: Literal["normalization_omission"] = "normalization_omission"


class UtsegModelShortCircuitV2(BaseModel):
    """Evidence that the normalized input was too short for inference."""

    kind: Literal["model_short_circuit"] = "model_short_circuit"


UtsegWordBoundaryEvidenceV2: TypeAlias = Annotated[
    UtsegClassifiedBoundaryEvidenceV2
    | UtsegNormalizationOmissionV2
    | UtsegModelShortCircuitV2,
    Field(discriminator="kind"),
]
"""One typed evidence state parallel to an input word."""


class UtsegBoundaryModelEvidenceV2(BaseModel):
    """Model provenance plus per-input-word boundary evidence."""

    model_id: Annotated[str, StringConstraints(min_length=1)]
    # Required, and commit-shaped. The boundary model is loaded from a pinned
    # snapshot whose commit is read off the directory on disk, so there is no
    # boundary result without an exact revision behind it.
    model_revision: HubCommitV2
    normalization_revision: UtsegNormalizationRevisionV2
    adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2
    word_evidence: list[UtsegWordBoundaryEvidenceV2]


class UtsegItemResultV2(BaseModel):
    """One utterance-segmentation item result returned by Python."""

    assignments: list[int] | None = None
    trees: list[str] | None = None
    boundary_model_evidence: UtsegBoundaryModelEvidenceV2 | None = None
    error: str | None = None


class UtsegResultPayloadV2(BaseModel):
    """Batched utterance-segmentation response payload (internally tagged)."""

    kind: Literal["utseg_result"] = "utseg_result"
    items: list[UtsegItemResultV2]


class TranslationTranslatedItemV2(BaseModel):
    """An item the engine translated, naming that engine."""

    kind: Literal["translated"] = "translated"
    raw_translation: str
    engine: ReportedEngineName


class TranslationBlankInputItemV2(BaseModel):
    """Whitespace-only input: never sent to an engine, so it names none."""

    kind: Literal["blank_input"] = "blank_input"


class TranslationProviderStatusItemV2(BaseModel):
    """The provider answered an HTTP status instead of a translation.

    The worker reports and never retries: the Rust control plane decides
    whether the status is waited out, where the wait is visible to the job's
    deadline and cancellation.
    """

    kind: Literal["provider_status"] = "provider_status"
    status: Annotated[int, Field(ge=100, le=599)]
    retry_after_s: FiniteNonNegativeFloat | None = None
    error: str


class TranslationNoResponseItemV2(BaseModel):
    """The request reached no provider answer: a transport failure."""

    kind: Literal["no_response"] = "no_response"
    error: str


class TranslationFailedItemV2(BaseModel):
    """The engine failed the item with no provider semantics; final."""

    kind: Literal["failed"] = "failed"
    error: str


TranslationItemResultV2: TypeAlias = Annotated[
    TranslationTranslatedItemV2
    | TranslationBlankInputItemV2
    | TranslationProviderStatusItemV2
    | TranslationNoResponseItemV2
    | TranslationFailedItemV2,
    Field(discriminator="kind"),
]
"""One translation item outcome (internally tagged on ``kind``)."""


class TranslationResultPayloadV2(BaseModel):
    """Batched translation response payload (internally tagged)."""

    kind: Literal["translation_result"] = "translation_result"
    items: list[TranslationItemResultV2]


class CorefChainRefV2(BaseModel):
    """One structured coreference chain reference returned by Python."""

    chain_id: int = Field(ge=0)
    is_start: bool
    is_end: bool


class CorefAnnotationV2(BaseModel):
    """One per-sentence coreference annotation returned by Python."""

    sentence_idx: int = Field(ge=0)
    words: list[list[CorefChainRefV2]]


class CorefResolvedItemV2(BaseModel):
    """A document the engine resolved, naming that engine.

    ``annotations`` may be empty: a resolved document with no chains.
    """

    kind: Literal["resolved"] = "resolved"
    annotations: list[CorefAnnotationV2]
    engine: ReportedEngineName


class CorefNoSentencesItemV2(BaseModel):
    """A document with no sentences: never sent to an engine, so it names none."""

    kind: Literal["no_sentences"] = "no_sentences"


class CorefFailedItemV2(BaseModel):
    """A document that could not be resolved, with the reason."""

    kind: Literal["failed"] = "failed"
    error: str


CorefItemResultV2: TypeAlias = Annotated[
    CorefResolvedItemV2 | CorefNoSentencesItemV2 | CorefFailedItemV2,
    Field(discriminator="kind"),
]
"""One coreference item outcome (internally tagged on ``kind``)."""


class CorefResultPayloadV2(BaseModel):
    """Batched coreference response payload (internally tagged)."""

    kind: Literal["coref_result"] = "coref_result"
    items: list[CorefItemResultV2]


class SpeakerSegmentV2(BaseModel):
    """One raw speaker diarization segment returned by Python."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: SpeakerId

    @model_validator(mode="after")
    def _validate_range(self) -> SpeakerSegmentV2:
        if self.end_ms < self.start_ms:
            raise ValueError("Speaker segment end_ms must be >= start_ms")
        return self


class PyannoteAISpeakerEvidenceV2(BaseModel):
    """Completed paid pyannoteAI job before local normalization."""

    kind: Literal["pyannote_ai"] = "pyannote_ai"
    job_id: str = Field(min_length=1)
    output: dict[str, Any]
    warning: str | None = None


class LocalPyannoteSpeakerEvidenceV2(BaseModel):
    """Segments returned by the local pyannote runtime."""

    kind: Literal["pyannote"] = "pyannote"
    segments: list[SpeakerSegmentV2]


class NemoSpeakerEvidenceV2(BaseModel):
    """Segments returned by the local NeMo runtime."""

    kind: Literal["nemo"] = "nemo"
    segments: list[SpeakerSegmentV2]


SpeakerInferenceEvidenceV2: TypeAlias = Annotated[
    PyannoteAISpeakerEvidenceV2
    | LocalPyannoteSpeakerEvidenceV2
    | NemoSpeakerEvidenceV2,
    Field(discriminator="kind"),
]


class SpeakerResultPayloadV2(BaseModel):
    """Backend-specific speaker evidence returned by the model host."""

    kind: Literal["speaker_result"] = "speaker_result"
    evidence: SpeakerInferenceEvidenceV2


class EmbeddedSpanV2(BaseModel):
    """A span the model measured."""

    kind: Literal["embedded"] = "embedded"
    vector: list[FiniteFloat]


class SpanTooShortForEmbeddingV2(BaseModel):
    """A span below the model's own minimum input length.

    Its own variant rather than an empty vector: the pinned model returns an
    all-NaN vector for such a span, which reads as a measurement and compares
    false against every threshold.
    """

    kind: Literal["too_short"] = "too_short"
    frame_count: int = Field(ge=0)


SpeakerEmbeddingOutcomeV2: TypeAlias = Annotated[
    EmbeddedSpanV2 | SpanTooShortForEmbeddingV2,
    Field(discriminator="kind"),
]


class SpeakerEmbeddingSpanResultV2(BaseModel):
    """One requested span's outcome, echoing the id it was requested under."""

    span_id: str
    outcome: SpeakerEmbeddingOutcomeV2


class SpeakerEmbeddingResultPayloadV2(BaseModel):
    """Embeddings for every requested span, plus the model's own bounds."""

    kind: Literal["speaker_embedding_result"] = "speaker_embedding_result"
    dimension: int = Field(gt=0)
    minimum_frames: int = Field(gt=0)
    spans: list[SpeakerEmbeddingSpanResultV2]


class OpenSmileResultPayloadV2(BaseModel):
    """Raw openSMILE tabular output returned by the model host (internally tagged)."""

    kind: Literal["opensmile_result"] = "opensmile_result"
    feature_set: str
    feature_level: str
    num_features: int = Field(ge=0)
    duration_segments: int = Field(ge=0)
    audio_file: str
    rows: list[dict[str, FiniteFloat]]
    success: bool
    error: str | None = None


class AvqiResultPayloadV2(BaseModel):
    """Raw AVQI metrics returned by the model host (internally tagged)."""

    kind: Literal["avqi_result"] = "avqi_result"
    avqi: FiniteFloat
    cpps: FiniteFloat
    hnr: FiniteFloat
    shimmer_local: FiniteFloat
    shimmer_local_db: FiniteFloat
    slope: FiniteFloat
    tilt: FiniteFloat
    cs_file: str
    sv_file: str
    success: bool
    error: str | None = None


# Backward-compatible aliases: with internally tagged unions, the payload
# structs carry the ``kind`` field directly, no wrapper needed.
WhisperChunkResultV2 = WhisperChunkResultPayloadV2
MonologueAsrResultV2 = MonologueAsrResultPayloadV2
WhisperTokenTimingResultV2 = WhisperTokenTimingResultPayloadV2
IndexedWordTimingResultV2 = IndexedWordTimingResultPayloadV2
MorphosyntaxResultV2 = MorphosyntaxResultPayloadV2
UtsegResultV2 = UtsegResultPayloadV2
TranslationResultV2 = TranslationResultPayloadV2
CorefResultV2 = CorefResultPayloadV2
SpeakerResultV2 = SpeakerResultPayloadV2
SpeakerEmbeddingResultV2 = SpeakerEmbeddingResultPayloadV2
OpenSmileResultV2 = OpenSmileResultPayloadV2
AvqiResultV2 = AvqiResultPayloadV2

TaskResultV2: TypeAlias = Annotated[
    WhisperChunkResultPayloadV2
    | MonologueAsrResultPayloadV2
    | WhisperTokenTimingResultPayloadV2
    | IndexedWordTimingResultPayloadV2
    | MorphosyntaxResultPayloadV2
    | UtsegResultPayloadV2
    | TranslationResultPayloadV2
    | CorefResultPayloadV2
    | SpeakerResultPayloadV2
    | SpeakerEmbeddingResultPayloadV2
    | OpenSmileResultPayloadV2
    | AvqiResultPayloadV2,
    Field(discriminator="kind"),
]
"""Typed execute result payload (internally tagged on ``kind``)."""


class ExecuteSuccessV2(BaseModel):
    """Successful execute outcome."""

    kind: Literal["success"] = "success"


class ExecuteErrorV2(BaseModel):
    """Protocol/runtime failure outcome."""

    kind: Literal["error"] = "error"
    code: ProtocolErrorCodeV2
    message: str


ExecuteOutcomeV2: TypeAlias = Annotated[
    ExecuteSuccessV2 | ExecuteErrorV2,
    Field(discriminator="kind"),
]
"""Top-level execute outcome."""


class ExecuteResponseV2(BaseModel):
    """Top-level V2 execute response.

    The outcome/result pairing is STRUCTURAL on both sides of the FFI since
    2026-08-21: the Rust type refuses a disagreeing pair at deserialization,
    and this validator refuses it where a Python handler would BUILD one, so
    the violation fails at its producer instead of surfacing as a refused
    wire line on the Rust side.
    """

    request_id: WorkerRequestIdV2
    outcome: ExecuteOutcomeV2
    result: TaskResultV2 | None = None
    elapsed_s: FiniteNonNegativeFloat

    @model_validator(mode="after")
    def _outcome_and_result_agree(self) -> ExecuteResponseV2:
        if isinstance(self.outcome, ExecuteSuccessV2) and self.result is None:
            raise ValueError(
                "execute response claimed success but carried no result payload"
            )
        if isinstance(self.outcome, ExecuteErrorV2) and self.result is not None:
            raise ValueError(
                "execute response reported an error but also carried a result payload"
            )
        return self


class ProgressEventV2(BaseModel):
    """Progress event emitted by long-running V2 tasks."""

    request_id: WorkerRequestIdV2
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    stage: str


class ShutdownRequestV2(BaseModel):
    """Shutdown request sent to a V2 worker."""

    request_id: WorkerRequestIdV2
