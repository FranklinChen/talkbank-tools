"""The ASR load: builds the ASR host one worker's requests use."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Mapping
from functools import partial

from pydantic import BaseModel, TypeAdapter

from batchalign.inference._domain_types import (
    RevAiApiKey,
    WhisperCpuPrecision,
    parse_choice,
)
from batchalign.worker._loads import (
    AsrLoad,
    ControlPlaneAsr,
    LoadedLocalAsr,
    LoadedProviderAsr,
)
from batchalign.worker._model_loading.pinned_hub import (
    resolve_repository,
    resolve_transformers,
    unpinned,
)
from batchalign.worker._types import AsrEngine, WorkerBootstrapRuntime
from batchalign.worker._types_v2 import (
    AliyunModelIdentityV2,
    AliyunModelsV2,
    AsrBackendV2,
    AsrRequestedModelsV2,
    LoadedModelV2,
    ObservedNotExposedV2,
    ParaformerModelIdentityV2,
    ParaformerModelsV2,
    QwenModelIdentityV2,
    QwenModelsV2,
    QwenRecognitionV2,
    RequestedModelV2,
    RequestedProviderParameterV2,
    RequestedUnpinnedV2,
    SenseVoiceModelIdentityV2,
    SenseVoiceModelsV2,
    TencentModelIdentityV2,
    TencentModelsV2,
    WhisperDecodeLanguageV2,
    WhisperModelsV2,
)

L = logging.getLogger("batchalign.worker")

# The engine-override key the Rust control plane sends its pinned ASR models
# under. Must equal `model_manifest::PINNED_ASR_MODELS_KEY` on the Rust side.
PINNED_ASR_MODELS_KEY = "asr_pinned_models"

_PINNED_ASR_MODELS: TypeAdapter[AsrRequestedModelsV2] = TypeAdapter(
    AsrRequestedModelsV2
)

# The engine-override key the control plane sends a Qwen worker's recognition
# policy under. Must equal `QWEN_RECOGNITION_KEY` in the Rust wire types.
QWEN_RECOGNITION_KEY = "qwen_recognition"

# The engine-override key the control plane sends a stock Whisper worker's
# decode language under. Must equal `WHISPER_DECODE_LANGUAGE_KEY` in the Rust
# wire types.
WHISPER_DECODE_LANGUAGE_KEY = "whisper_decode_language"

_QWEN_RECOGNITION: TypeAdapter[QwenRecognitionV2] = TypeAdapter(QwenRecognitionV2)
_WHISPER_DECODE_LANGUAGE: TypeAdapter[WhisperDecodeLanguageV2] = TypeAdapter(
    WhisperDecodeLanguageV2
)


def _required_spawn_policy[T](
    engine_overrides: dict[str, str] | None,
    key: str,
    adapter: TypeAdapter[T],
    absent: str,
) -> T:
    """Read a policy the control plane decided and sent at spawn under ``key``.

    Absence is not a state the worker fills in: deciding the policy here from
    the language code would make the worker a second owner of it. ``absent``
    says what a missing value means for this policy; a malformed one means the
    two sides disagree about the wire and is refused by the adapter.
    """
    raw = (engine_overrides or {}).get(key)
    if raw is None:
        raise ValueError(f"no {key!r} was received from the control plane: {absent}")
    return adapter.validate_json(raw)


def qwen_recognition(engine_overrides: dict[str, str] | None) -> QwenRecognitionV2:
    """Return the recognition policy Rust decided for this Qwen worker.

    Qwen loads only when the control plane named it, and that path always
    sends the policy, so absence means the two sides disagree.
    """
    return _required_spawn_policy(
        engine_overrides,
        QWEN_RECOGNITION_KEY,
        _QWEN_RECOGNITION,
        "the Qwen recognizer needs its recognition policy",
    )


def whisper_decode_language(
    engine_overrides: dict[str, str] | None,
) -> WhisperDecodeLanguageV2:
    """Return the decode language Rust decided for this stock Whisper worker.

    The control plane sends it to every worker that may load stock Whisper
    and has one job language (or auto-detection) with a Whisper token.
    """
    return _required_spawn_policy(
        engine_overrides,
        WHISPER_DECODE_LANGUAGE_KEY,
        _WHISPER_DECODE_LANGUAGE,
        "stock Whisper was loaded for a worker whose language has no Whisper "
        "token, or that has no single job language (per-file or none)",
    )


def pinned_asr_models(
    engine_overrides: dict[str, str] | None,
) -> AsrRequestedModelsV2 | None:
    """Return the models Rust pinned for this worker, or ``None``.

    A load input, read once by ``load_asr_engine`` and passed to the arm that
    loads. The control plane injects the pin at spawn whenever the worker key
    names a worker-hosted ASR engine (``spawn_engine_overrides``), and an ASR
    load names its engine only then, so a named engine arrives with its pin.
    ``None`` reaches the arms only from a direct caller; each arm then loads
    its engine's default checkpoint and reports an unpinned identity.

    A malformed value is NOT tolerated: it means the control plane and this
    worker disagree about the wire, and loading some other revision because
    the pin could not be read is exactly the silent substitution this
    workstream removes.
    """
    raw = (engine_overrides or {}).get(PINNED_ASR_MODELS_KEY)
    if raw is None:
        return None
    return _PINNED_ASR_MODELS.validate_json(raw)


# The stock Whisper checkpoint, mirroring `model_manifest::WHISPER_LARGE_V3`.
# Read ONLY when nothing was pinned (a direct caller); a pinned load always
# uses the id Rust sent.
STOCK_WHISPER_ID = "openai/whisper-large-v3"

# The default Qwen3-ASR checkpoint, mirroring `model_manifest::QWEN_DEFAULT_ID`.
# Same rule: consulted only when nothing was pinned.
DEFAULT_QWEN_ASR_ID = "Qwen/Qwen3-ASR-1.7B-hf"

# The default FunAudio composition, mirroring `model_manifest::SENSEVOICE` and
# `SENSEVOICE_VAD`. The VAD is the HUGGING FACE `fsmn-vad`, which is a
# different repository from the ModelScope one Paraformer loads under the same
# alias; pinning both to one id would load the wrong weights for one of them.
_DEFAULT_SENSEVOICE_ID = "FunAudioLLM/SenseVoiceSmall"
_DEFAULT_SENSEVOICE_VAD_ID = "funasr/fsmn-vad"

# The file a FunASR model directory loads its weights from. Without it FunASR
# builds the model with untrained weights and transcribes nonsense without an
# error, so a ModelScope snapshot is admitted only with it present.
_FUNASR_CHECKPOINT = ("model.pt",)

# Aliyun's identity, mirroring `model_manifest::ALIYUN_SERVICE`. A service, never
# the appkey, which identifies an account rather than a model.
_ALIYUN_SERVICE_ID = "aliyun-nls"
_ALIYUN_SERVICE_PARAMETER = "speech-transcriber-v1"


def _tag_of(member: RequestedModelV2) -> str | None:
    """The published tag a ModelScope member is pinned to, if it is pinned.

    ModelScope exposes no commit behind a tag, so a tag is the most exact
    thing that can be said about these models. ``None`` is returned for an
    UNPINNED member, which is a real case: a user may name a Paraformer
    checkpoint this build does not pin, while its voice-activity and
    punctuation models stay pinned. Any other revision kind means the manifest
    and this loader disagree about which hub a model lives on, which is a
    refusal rather than something to paper over.
    """
    revision = member.revision
    if revision.kind == "unpinned":
        return None
    if revision.kind != "tag":
        raise ValueError(
            f"{member.id} is a ModelScope model pinned by tag, but the plan "
            f"asked for a {revision.kind!r} revision"
        )
    return revision.tag


def _pinned_composition[Composition: BaseModel](
    pinned: AsrRequestedModelsV2 | None,
    expected: type[Composition],
) -> Composition | None:
    """The pinned composition for the engine being loaded, if there is one.

    Takes the expected composition TYPE rather than its engine tag, so the
    caller gets back a narrowed value and reading ``pinned.asr`` on a Tencent
    composition is a type error rather than an attribute error at load time.

    A composition for a DIFFERENT engine is a refusal, not something to work
    around: it means the control plane and this worker disagree about what
    this process is, and loading anyway would produce a transcript whose
    recorded identity belongs to another engine.
    """
    if pinned is None:
        return None
    if not isinstance(pinned, expected):
        wanted = expected.model_fields["engine"].default
        raise ValueError(
            f"the control plane pinned a {pinned.engine!r} ASR composition but "
            f"this worker is loading {wanted!r}"
        )
    return pinned


def _declared_member(member: RequestedModelV2) -> LoadedModelV2:
    """Record a member whose runtime exposes no revision.

    ModelScope loaders and every cloud provider are told which revision or
    parameter to use and report nothing back, so the observation is an
    explicit absence rather than an invented echo of the request.
    """
    return LoadedModelV2(
        id=member.id,
        requested=member.revision,
        observed=ObservedNotExposedV2(),
    )


def load_asr_engine(bootstrap: WorkerBootstrapRuntime) -> AsrLoad:
    """Load the ASR engine the control plane named, and return its host.

    The host carries the runner over the loaded model and the identity of
    what loaded (``model_identity``), which the Rust executor reads from the
    host it is handed. With no engine named, the plan's ASR runs in the
    control plane and this load builds nothing (:class:`ControlPlaneAsr`).

    Dispatches on the resolved ``AsrEngine`` so adding a new variant later
    forces a missing-arm error rather than silently loading Whisper.
    """
    lang = bootstrap.lang
    engine_overrides = bootstrap.engine_overrides or None
    # Read once, before any loader runs, so every engine loads the revision
    # the plan chose rather than whatever the hub currently serves. A load
    # input, passed to the arm that reads it; never worker state.
    pinned = pinned_asr_models(engine_overrides)

    backend = resolve_asr_engine(engine_overrides)

    if isinstance(backend, ControlPlaneAsr):
        L.info("No worker-hosted ASR engine named: the plan's ASR runs in Rust")
        return backend
    elif backend is AsrEngine.TENCENT:
        from batchalign.inference.languages.cantonese._tencent_asr import (
            load_tencent_asr,
        )

        # Rust chooses the engine-model type, through the one ISO 639-3 to
        # 639-1 conversion and the one Han-script table. Python no longer
        # derives it: the derivation it replaces had its own five-code Chinese
        # list that omitted Mandarin, so `cmn` asked for a `16k_cmn` model that
        # Tencent does not define.
        tencent_pinned = _pinned_composition(pinned, TencentModelsV2)
        if tencent_pinned is None:
            raise ValueError(
                "the Tencent ASR engine needs the engine-model type the control "
                "plane chose, and this worker was spawned without one"
            )
        revision = tencent_pinned.engine_model_type.revision
        if revision.kind != "provider_parameter":
            raise ValueError(
                f"Tencent selects its model by provider parameter, but the plan "
                f"asked for a {revision.kind!r} revision"
            )
        tencent = load_tencent_asr(lang, engine_model_type=revision.parameter)
        return LoadedProviderAsr(
            backend=AsrBackendV2.HK_TENCENT,
            runner=tencent.transcribe,
            model_identity=TencentModelIdentityV2(
                engine_model_type=_declared_member(tencent_pinned.engine_model_type)
            ),
        )
    elif backend is AsrEngine.ALIYUN:
        from batchalign.inference.languages.cantonese._aliyun_asr import load_aliyun_asr

        # Nothing to resolve: Aliyun is a service, not a checkpoint, so the
        # identity is the service it was asked for and an explicitly unexposed
        # revision. The appkey identifies an ACCOUNT rather than a model and is
        # never recorded, which is why the manifest names the service instead.
        aliyun = load_aliyun_asr(lang, engine_overrides)
        aliyun_pinned = _pinned_composition(pinned, AliyunModelsV2)
        service = (
            aliyun_pinned.service
            if aliyun_pinned is not None
            else RequestedModelV2(
                id=_ALIYUN_SERVICE_ID,
                revision=RequestedProviderParameterV2(
                    parameter=_ALIYUN_SERVICE_PARAMETER
                ),
            )
        )
        return LoadedProviderAsr(
            backend=AsrBackendV2.HK_ALIYUN,
            runner=aliyun.transcribe,
            model_identity=AliyunModelIdentityV2(service=_declared_member(service)),
        )
    elif backend is AsrEngine.FUNAUDIO:
        from batchalign.inference.languages.cantonese._funaudio_asr import (
            load_funaudio_asr,
        )
        from batchalign.inference.languages.cantonese._funaudio_common import (
            ParaformerSnapshots,
            SenseVoiceSnapshots,
        )
        from batchalign.worker._model_loading.modelscope_hub import (
            resolve_modelscope_snapshot,
        )

        # FunAudio spans two materially different compositions, and the
        # checkpoint the request selected decides which: SenseVoice with a
        # voice-activity model on Hugging Face, or Paraformer with a
        # voice-activity AND a punctuation model on ModelScope. That is why
        # the manifest resolves a composition rather than an engine alone.
        #
        # Either way every member is materialized HERE, inside the load, and
        # the recognizer is given local paths only: a download is a load phase
        # the supervisor watches for progress, never a side effect of the
        # first transcription request.
        if isinstance(pinned, ParaformerModelsV2):
            # ModelScope honours the tags it is given and reports no commit
            # back, so the identity is the declared one.
            paraformer = load_funaudio_asr(
                lang,
                engine_overrides,
                snapshots=ParaformerSnapshots(
                    asr=resolve_modelscope_snapshot(
                        pinned.asr.id,
                        _tag_of(pinned.asr),
                        kind="ASR (Paraformer)",
                        requires=_FUNASR_CHECKPOINT,
                    ),
                    vad=resolve_modelscope_snapshot(
                        pinned.vad.id,
                        _tag_of(pinned.vad),
                        kind="voice activity detection",
                        requires=_FUNASR_CHECKPOINT,
                    ),
                    punc=resolve_modelscope_snapshot(
                        pinned.punc.id,
                        _tag_of(pinned.punc),
                        kind="punctuation",
                        requires=_FUNASR_CHECKPOINT,
                    ),
                ),
                label=pinned.asr.id,
            )
            return LoadedProviderAsr(
                backend=AsrBackendV2.HK_FUNAUDIO,
                runner=paraformer.transcribe,
                model_identity=ParaformerModelIdentityV2(
                    asr=_declared_member(pinned.asr),
                    vad=_declared_member(pinned.vad),
                    punc=_declared_member(pinned.punc),
                ),
            )
        else:
            if isinstance(pinned, SenseVoiceModelsV2):
                asr_member = pinned.asr
                vad_member = pinned.vad
            elif pinned is None:
                asr_member = RequestedModelV2(
                    id=_DEFAULT_SENSEVOICE_ID, revision=RequestedUnpinnedV2()
                )
                vad_member = RequestedModelV2(
                    id=_DEFAULT_SENSEVOICE_VAD_ID, revision=RequestedUnpinnedV2()
                )
            else:
                raise ValueError(
                    f"the control plane pinned a {pinned.engine!r} ASR "
                    f"composition but this worker is loading 'funaudio'"
                )
            # FunASR's Hugging Face path IGNORES the revision it is handed
            # (`get_or_download_model_dir_hf` drops it), so passing one would
            # pin nothing. The snapshots are resolved here instead and the
            # loader is given local paths, which is also what makes the
            # observed commit a fact about the bytes on disk.
            asr_snapshot = resolve_repository(asr_member, kind="ASR")
            vad_snapshot = resolve_repository(
                vad_member, kind="voice activity detection"
            )
            sensevoice = load_funaudio_asr(
                lang,
                engine_overrides,
                snapshots=SenseVoiceSnapshots(asr=asr_snapshot, vad=vad_snapshot),
                label=asr_member.id,
            )
            return LoadedProviderAsr(
                backend=AsrBackendV2.HK_FUNAUDIO,
                runner=sensevoice.transcribe,
                model_identity=SenseVoiceModelIdentityV2(
                    asr=asr_snapshot.loaded, vad=vad_snapshot.loaded
                ),
            )
    elif backend is AsrEngine.QWEN:
        from batchalign.inference.languages.cantonese._qwen_asr import load_qwen_asr
        from batchalign.inference.languages.cantonese._qwen_common import (
            QwenSnapshots,
            admit_qwen_checkpoint,
        )
        from batchalign.inference.qwen_forced_alignment import (
            QWEN_FORCED_ALIGNER_MODEL_ID,
        )

        # The aligner is not optional: Qwen3-ASR produces word timings only
        # with it, which is why the composition makes it a required field
        # rather than something a caller can forget.
        # Named per arm rather than reusing one `pinned`: each arm narrows to a
        # DIFFERENT composition type, and Python has no block scope, so one
        # shared name would take the type of whichever arm was written first.
        qwen_pinned = _pinned_composition(pinned, QwenModelsV2)
        if qwen_pinned is None:
            asr_member = unpinned(DEFAULT_QWEN_ASR_ID)
            aligner_member = unpinned(QWEN_FORCED_ALIGNER_MODEL_ID)
        else:
            asr_member = qwen_pinned.asr
            aligner_member = qwen_pinned.aligner
        # Before the download: a pre-migration checkpoint id is refused with
        # an instruction, not after gigabytes of the wrong repository arrive.
        admit_qwen_checkpoint(asr_member.id)
        # Names of their own: the FunASR arm's `asr_snapshot` is a
        # RepositorySnapshot, and one name across arms takes one type.
        qwen_asr = resolve_transformers(asr_member, kind="ASR")
        qwen_aligner = resolve_transformers(aligner_member, kind="forced alignment")
        qwen = load_qwen_asr(
            lang,
            engine_overrides,
            recognition=qwen_recognition(engine_overrides),
            snapshots=QwenSnapshots(asr=qwen_asr, aligner=qwen_aligner),
        )
        return LoadedProviderAsr(
            backend=AsrBackendV2.HK_QWEN,
            runner=qwen.transcribe,
            model_identity=QwenModelIdentityV2(
                asr=qwen_asr.loaded, aligner=qwen_aligner.loaded
            ),
        )
    elif backend is AsrEngine.WHISPER_HUB:
        # Community HF Whisper fine-tune. Rust owns WHICH fine-tune a language
        # gets and pins its revision; the handle is the same
        # ``WhisperASRHandle`` stock Whisper uses, so downstream V2 inference
        # needs no branching on engine identity after load time.
        from batchalign.inference.asr import infer_whisper_prepared_audio
        from batchalign.inference.whisper_hub import (
            load_whisper_hub_asr,
            resolve_whisper_hub_model_id,
        )

        hub_pinned = _pinned_composition(pinned, WhisperModelsV2)
        if hub_pinned is None:
            # Unpinned (a direct caller): resolve the id the
            # way a direct caller would, and report the commit that lands.
            member = unpinned(resolve_whisper_hub_model_id(lang, engine_overrides))
        else:
            member = hub_pinned.asr
        handle = load_whisper_hub_asr(
            resolve_transformers(member, kind="ASR"),
            device_policy=bootstrap.device_policy,
        )
        return LoadedLocalAsr(
            backend=AsrBackendV2.WHISPER_HUB,
            runner=partial(infer_whisper_prepared_audio, handle),
            model_identity=handle.model_identity,
        )
    elif backend is AsrEngine.WHISPER:
        from batchalign.inference.asr import (
            infer_whisper_prepared_audio,
            load_whisper_asr,
        )

        # Stock Whisper loads ONE checkpoint whatever the language; only the
        # decode language differs, and the control plane decided it.
        decoding = whisper_decode_language(engine_overrides)
        whisper_pinned = _pinned_composition(pinned, WhisperModelsV2)
        member = (
            whisper_pinned.asr
            if whisper_pinned is not None
            else unpinned(STOCK_WHISPER_ID)
        )
        handle = load_whisper_asr(
            resolve_transformers(member, kind="ASR"),
            decoding=decoding,
            device_policy=bootstrap.device_policy,
            cpu_precision=WhisperCpuPrecision.from_overrides(engine_overrides),
        )
        return LoadedLocalAsr(
            backend=AsrBackendV2.LOCAL_WHISPER,
            runner=partial(infer_whisper_prepared_audio, handle),
            model_identity=handle.model_identity,
        )
    else:
        # Exhaustive match. ``typing.assert_never`` makes the type
        # checker prove this branch is unreachable; if a new AsrEngine
        # variant is added without a load arm here, mypy / pyright
        # flags it at compile time. At runtime it raises ``AssertionError``
        # so a regression still fails loudly instead of silently
        # falling through.
        typing.assert_never(backend)


def resolve_asr_engine(
    engine_overrides: dict[str, str] | None,
) -> AsrEngine | ControlPlaneAsr:
    """Which ASR engine this load builds, as the control plane named it.

    The control plane names the engine (``asr``) for every worker-hosted ASR
    load. It names none when the plan's ASR runs in Rust (Rev.AI,
    whisper.cpp), which is :class:`ControlPlaneAsr`: build nothing. An
    unknown name raises ``ValueError`` rather than loading anything else: a
    typo in a per-host override would otherwise produce wrong-model output.

    There is no default and no second chooser. This used to fall back to
    Rev.AI when a key was in the environment, then to a per-language table
    (Cantonese to FunAudio), then to Whisper, so a Rev.AI job's capability
    probe could load a multi-gigabyte model the job never used.
    """
    named = (engine_overrides or {}).get("asr")
    if named is None:
        return ControlPlaneAsr()
    return parse_choice(AsrEngine, named, "asr engine")


def resolve_injected_revai_api_key(
    environ: Mapping[str, str] | None = None,
) -> RevAiApiKey | None:
    """Resolve a pre-injected Rev.AI key from an explicit environment mapping."""
    env = environ if environ is not None else os.environ
    for key_name in ("BATCHALIGN_REV_API_KEY", "REVAI_API_KEY"):
        env_value = env.get(key_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return None


__all__ = [
    "WHISPER_DECODE_LANGUAGE_KEY",
    "load_asr_engine",
    "resolve_asr_engine",
    "resolve_injected_revai_api_key",
    "whisper_decode_language",
]
