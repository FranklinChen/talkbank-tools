"""ASR-engine bootstrap helpers for worker startup."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Mapping

from pydantic import BaseModel, TypeAdapter

from batchalign.inference._domain_types import (
    LanguageCode,
    RevAiApiKey,
    WhisperCpuPrecision,
    parse_choice,
)
from batchalign.inference.asr import iso3_to_language_name
from batchalign.worker._model_loading.pinned_hub import (
    hub_commit_of,
    resolve_pinned_snapshot,
)
from batchalign.worker._types import AsrEngine, WorkerBootstrapRuntime, _state
from batchalign.worker._types_v2 import (
    AliyunModelIdentityV2,
    AliyunModelsV2,
    AsrRequestedModelsV2,
    LoadedModelV2,
    ObservedCommitV2,
    ObservedNotExposedV2,
    ParaformerModelIdentityV2,
    ParaformerModelsV2,
    QwenModelIdentityV2,
    QwenModelsV2,
    RequestedModelV2,
    RequestedProviderParameterV2,
    RequestedUnpinnedV2,
    SenseVoiceModelIdentityV2,
    SenseVoiceModelsV2,
    TencentModelIdentityV2,
    TencentModelsV2,
    WhisperModelIdentityV2,
    WhisperModelsV2,
)

L = logging.getLogger("batchalign.worker")

# The engine-override key the Rust control plane sends its pinned ASR models
# under. Must equal `model_manifest::PINNED_ASR_MODELS_KEY` on the Rust side.
PINNED_ASR_MODELS_KEY = "asr_pinned_models"

_PINNED_ASR_MODELS: TypeAdapter[AsrRequestedModelsV2] = TypeAdapter(
    AsrRequestedModelsV2
)


def pinned_asr_models(
    engine_overrides: dict[str, str] | None,
) -> AsrRequestedModelsV2 | None:
    """Return the models Rust pinned for this worker, or ``None``.

    ``None`` is a real state rather than a missing value. The control plane
    injects the pin only when it named an ASR engine for the spawn, and it
    deliberately does not name one for a worker that merely PRELOADS ASR as
    part of its profile (an align job's GPU worker is the case; see
    ``EngineSelection::for_target``, which explains why filling in an ASR
    default there would overrule a better-informed per-language choice).
    Such a worker resolves its own engine and reports an unpinned identity.

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
# Read ONLY when the control plane pinned nothing (an align job's GPU worker
# preloading ASR); a pinned load always uses the id Rust sent.
_STOCK_WHISPER_ID = "openai/whisper-large-v3"

# The default Qwen3-ASR checkpoint, mirroring `model_manifest::QWEN_DEFAULT_ID`.
# Same rule: consulted only when nothing was pinned.
_DEFAULT_QWEN_ID = "Qwen/Qwen3-ASR-1.7B-hf"

# The default FunAudio composition, mirroring `model_manifest::SENSEVOICE` and
# `SENSEVOICE_VAD`. The VAD is the HUGGING FACE `fsmn-vad`, which is a
# different repository from the ModelScope one Paraformer loads under the same
# alias; pinning both to one id would load the wrong weights for one of them.
_DEFAULT_SENSEVOICE_ID = "FunAudioLLM/SenseVoiceSmall"
_DEFAULT_SENSEVOICE_VAD_ID = "funasr/fsmn-vad"

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
    models = _state.asr_pinned_models
    if models is None:
        return None
    if not isinstance(models, expected):
        wanted = expected.model_fields["engine"].default
        raise ValueError(
            f"the control plane pinned a {models.engine!r} ASR composition but "
            f"this worker is loading {wanted!r}"
        )
    return models


def _load_hub_member(
    member: RequestedModelV2,
    *,
    kind: str,
    artifacts: tuple[str, ...] | None = None,
) -> tuple[str, LoadedModelV2]:
    """Materialize one Hugging Face member, and record what actually loaded.

    Returns the local path a loader should be pointed at. Resolving the
    snapshot here rather than passing a revision down is what makes the
    observation real: the commit comes from the directory on disk, not from a
    library internal that may be absent, and FunASR's Hugging Face path
    ignores revisions entirely.

    The download event belongs here for the same reason: this is the call that
    can block for minutes on a cold cache, and it is the only place that knows
    both the model id and the revision being fetched. Announcing it downstream,
    where the loader has been handed a local path, asks whether a DIRECTORY is
    a cached repository, which it never is.

    ``kind`` names this member's role for the operator; ``artifacts`` widens
    the cache probe for loaders that pull more than ``config.json``.
    """
    from batchalign.worker._progress import emit_hf_download_if_missing

    commit = hub_commit_of(member)
    emit_hf_download_if_missing(
        member.id, kind=kind, artifacts=artifacts, revision=commit
    )
    resolved = resolve_pinned_snapshot(member.id, commit)
    return resolved.path, LoadedModelV2(
        id=member.id,
        requested=member.revision,
        observed=ObservedCommitV2(commit=resolved.commit),
    )


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


def load_asr_engine(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the ASR engine for this worker.

    The control plane may inject a resolved Rev.AI key directly into the worker
    bootstrap runtime. When it does, that injected value is authoritative and
    the worker does not rediscover credentials from ambient process state.

    Dispatches on the resolved ``AsrEngine`` so adding a new variant later
    forces a missing-arm error rather than silently loading Whisper.
    """
    lang = bootstrap.lang
    engine_overrides = bootstrap.engine_overrides or None
    rev_api_key = bootstrap.revai_api_key
    _state.rev_api_key = None
    # Read once, before any loader runs, so every engine loads the revision
    # the plan chose rather than whatever the hub currently serves.
    _state.asr_pinned_models = pinned_asr_models(engine_overrides)

    backend = resolve_asr_engine(engine_overrides, rev_api_key, lang=lang)

    if backend is AsrEngine.REV:
        _state.rev_api_key = rev_api_key
        if rev_api_key is None:
            L.error("Rev.AI key not configured")
        _state.asr_engine = AsrEngine.REV
    elif backend is AsrEngine.TENCENT:
        from batchalign.inference.languages.cantonese._tencent_asr import (
            load_tencent_asr,
        )

        # Rust chooses the engine-model type, through the one ISO 639-3 to
        # 639-1 conversion and the one Han-script table. Python no longer
        # derives it: the derivation it replaces had its own five-code Chinese
        # list that omitted Mandarin, so `cmn` asked for a `16k_cmn` model that
        # Tencent does not define.
        tencent_pinned = _pinned_composition(TencentModelsV2)
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
        load_tencent_asr(
            lang,
            engine_model_type=revision.parameter,
        )
        _state.asr_model_identity = TencentModelIdentityV2(
            engine_model_type=_declared_member(tencent_pinned.engine_model_type)
        )
        _state.asr_engine = AsrEngine.TENCENT
    elif backend is AsrEngine.ALIYUN:
        from batchalign.inference.languages.cantonese._aliyun_asr import load_aliyun_asr

        # Nothing to resolve: Aliyun is a service, not a checkpoint, so the
        # identity is the service it was asked for and an explicitly unexposed
        # revision. The appkey identifies an ACCOUNT rather than a model and is
        # never recorded, which is why the manifest names the service instead.
        load_aliyun_asr(lang, engine_overrides)
        aliyun_pinned = _pinned_composition(AliyunModelsV2)
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
        _state.asr_model_identity = AliyunModelIdentityV2(
            service=_declared_member(service)
        )
        _state.asr_engine = AsrEngine.ALIYUN
    elif backend is AsrEngine.FUNAUDIO:
        from batchalign.inference.languages.cantonese._funaudio_asr import (
            load_funaudio_asr,
        )

        # FunAudio spans two materially different compositions, and the
        # checkpoint the request selected decides which: SenseVoice with a
        # voice-activity model on Hugging Face, or Paraformer with a
        # voice-activity AND a punctuation model on ModelScope. That is why
        # the manifest resolves a composition rather than an engine alone.
        funaudio_pinned = _state.asr_pinned_models
        if isinstance(funaudio_pinned, ParaformerModelsV2):
            # ModelScope honours the revisions it is given, so the ids and
            # tags go straight down; nothing reports a commit back.
            load_funaudio_asr(
                lang,
                engine_overrides,
                model=funaudio_pinned.asr.id,
                model_revision=_tag_of(funaudio_pinned.asr),
                vad_model=funaudio_pinned.vad.id,
                vad_revision=_tag_of(funaudio_pinned.vad),
                punc_model=funaudio_pinned.punc.id,
                punc_revision=_tag_of(funaudio_pinned.punc),
            )
            _state.asr_model_identity = ParaformerModelIdentityV2(
                asr=_declared_member(funaudio_pinned.asr),
                vad=_declared_member(funaudio_pinned.vad),
                punc=_declared_member(funaudio_pinned.punc),
            )
        else:
            if isinstance(funaudio_pinned, SenseVoiceModelsV2):
                asr_member = funaudio_pinned.asr
                vad_member = funaudio_pinned.vad
            elif funaudio_pinned is None:
                asr_member = RequestedModelV2(
                    id=_DEFAULT_SENSEVOICE_ID, revision=RequestedUnpinnedV2()
                )
                vad_member = RequestedModelV2(
                    id=_DEFAULT_SENSEVOICE_VAD_ID, revision=RequestedUnpinnedV2()
                )
            else:
                raise ValueError(
                    f"the control plane pinned a {funaudio_pinned.engine!r} ASR "
                    f"composition but this worker is loading 'funaudio'"
                )
            # FunASR's Hugging Face path IGNORES the revision it is handed
            # (`get_or_download_model_dir_hf` drops it), so passing one would
            # pin nothing. The snapshots are resolved here instead and the
            # loader is given local paths, which is also what makes the
            # observed commit a fact about the bytes on disk.
            asr_path, asr_loaded = _load_hub_member(asr_member, kind="ASR")
            vad_path, vad_loaded = _load_hub_member(
                vad_member, kind="voice activity detection"
            )
            load_funaudio_asr(
                lang,
                engine_overrides,
                model_path=asr_path,
                vad_model_path=vad_path,
            )
            _state.asr_model_identity = SenseVoiceModelIdentityV2(
                asr=asr_loaded, vad=vad_loaded
            )
        _state.asr_engine = AsrEngine.FUNAUDIO
    elif backend is AsrEngine.QWEN:
        from batchalign.inference.languages.cantonese._qwen_asr import load_qwen_asr
        from batchalign.inference.qwen_forced_alignment import (
            QWEN_FORCED_ALIGNER_MODEL_ID,
        )

        # The aligner is not optional: Qwen3-ASR produces word timings only
        # with it, which is why the composition makes it a required field
        # rather than something a caller can forget.
        # Named per arm rather than reusing one `pinned`: each arm narrows to a
        # DIFFERENT composition type, and Python has no block scope, so one
        # shared name would take the type of whichever arm was written first.
        qwen_pinned = _pinned_composition(QwenModelsV2)
        if qwen_pinned is None:
            asr_member = RequestedModelV2(
                id=_DEFAULT_QWEN_ID, revision=RequestedUnpinnedV2()
            )
            aligner_member = RequestedModelV2(
                id=QWEN_FORCED_ALIGNER_MODEL_ID, revision=RequestedUnpinnedV2()
            )
        else:
            asr_member = qwen_pinned.asr
            aligner_member = qwen_pinned.aligner
        asr_path, asr_loaded = _load_hub_member(asr_member, kind="ASR")
        aligner_path, aligner_loaded = _load_hub_member(
            aligner_member, kind="forced alignment"
        )
        load_qwen_asr(
            lang,
            engine_overrides,
            model_path=asr_path,
            aligner_path=aligner_path,
        )
        _state.asr_model_identity = QwenModelIdentityV2(
            asr=asr_loaded, aligner=aligner_loaded
        )
        _state.asr_engine = AsrEngine.QWEN
    elif backend is AsrEngine.WHISPER_HUB:
        # Community HF Whisper fine-tune. Rust owns WHICH fine-tune a language
        # gets and pins its revision; the handle is the same
        # ``WhisperASRHandle`` stock Whisper uses, so downstream V2 inference
        # needs no branching on engine identity after load time.
        from batchalign.inference.whisper_hub import (
            load_whisper_hub_asr,
            resolve_whisper_hub_model_id,
        )
        from batchalign.worker._progress import HF_ARTIFACTS_WHISPER

        hub_pinned = _pinned_composition(WhisperModelsV2)
        if hub_pinned is None:
            # Unpinned (this worker merely preloads ASR): resolve the id the
            # way a direct caller would, and report the commit that lands.
            member = RequestedModelV2(
                id=resolve_whisper_hub_model_id(lang, engine_overrides),
                revision=RequestedUnpinnedV2(),
            )
        else:
            member = hub_pinned.asr
        path, loaded = _load_hub_member(
            member, kind="ASR", artifacts=HF_ARTIFACTS_WHISPER
        )
        handle = load_whisper_hub_asr(
            lang,
            engine_overrides,
            device_policy=bootstrap.device_policy,
            model_path=path,
        )
        handle.model_identity = WhisperModelIdentityV2(asr=loaded)
        _state.whisper_asr_model = handle
        _state.asr_model_identity = handle.model_identity
        _state.asr_engine = AsrEngine.WHISPER_HUB
    elif backend is AsrEngine.WHISPER:
        from batchalign.inference.asr import load_whisper_asr
        from batchalign.worker._progress import HF_ARTIFACTS_WHISPER

        # Stock Whisper loads ONE checkpoint whatever the language; only the
        # generation hint differs, and ``language`` already carries that
        # (``"auto"`` makes Whisper detect per segment). The two arms this
        # replaces passed the same model id by two routes, one of them through
        # the loader's defaults.
        language = iso3_to_language_name(lang)
        whisper_pinned = _pinned_composition(WhisperModelsV2)
        member = (
            whisper_pinned.asr
            if whisper_pinned is not None
            else RequestedModelV2(id=_STOCK_WHISPER_ID, revision=RequestedUnpinnedV2())
        )
        path, loaded = _load_hub_member(
            member, kind="ASR", artifacts=HF_ARTIFACTS_WHISPER
        )
        handle = load_whisper_asr(
            model=path,
            base=path,
            language=language,
            device_policy=bootstrap.device_policy,
            cpu_precision=WhisperCpuPrecision.from_overrides(engine_overrides),
        )
        handle.model_identity = WhisperModelIdentityV2(asr=loaded)
        _state.whisper_asr_model = handle
        _state.asr_model_identity = handle.model_identity
        _state.asr_engine = AsrEngine.WHISPER
    else:
        # Exhaustive match. ``typing.assert_never`` makes the type
        # checker prove this branch is unreachable; if a new AsrEngine
        # variant is added without a load arm here, mypy / pyright
        # flags it at compile time. At runtime it raises ``AssertionError``
        # so a regression still fails loudly instead of silently
        # falling through.
        typing.assert_never(backend)


# Per-language default ASR engine, consulted only when no explicit
# ``--engine-overrides`` and no Rev.AI key are present.
#
# Why ``yue → FUNAUDIO``: the 2026-05-26 v2 Cantonese ASR benchmark
# measured vanilla Whisper-large-v3 at 81.9% CER on Tier 3 child
# speech (worst of every engine measured), while FunASR/SenseVoiceSmall
# came in at 42.8% (best open engine). Defaulting yue workers to
# Whisper silently shipped the worst-measured engine to operators who
# never passed an override flag.
_LANG_DEFAULTS: dict[LanguageCode, AsrEngine] = {
    "yue": AsrEngine.FUNAUDIO,
}


def resolve_asr_engine(
    engine_overrides: dict[str, str] | None,
    rev_api_key: RevAiApiKey | None,
    *,
    lang: LanguageCode,
) -> AsrEngine:
    """Resolve which ASR engine this worker should load.

    Precedence:

    1. Explicit engine override from the Rust control plane. Unknown
       wire strings raise ``ValueError`` rather than silently loading
       Whisper: a typo in a per-host override would otherwise produce
       wrong-model output.
    2. Rev.AI when a key is available.
    3. Per-language default from ``_LANG_DEFAULTS`` (currently
       ``yue → FUNAUDIO``).
    4. Local Whisper fallback for every other language.

    ``lang`` is required (keyword-only) so a future caller cannot
    accidentally trigger the global Whisper fallback by forgetting
    to pass the language: that silent mis-selection is exactly the
    bug Fix 3 closes.
    """
    if engine_overrides and "asr" in engine_overrides:
        return parse_choice(AsrEngine, engine_overrides["asr"], "asr engine")
    if rev_api_key:
        return AsrEngine.REV
    return _LANG_DEFAULTS.get(lang, AsrEngine.WHISPER)


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
    "iso3_to_language_name",
    "load_asr_engine",
    "resolve_asr_engine",
    "resolve_injected_revai_api_key",
]
