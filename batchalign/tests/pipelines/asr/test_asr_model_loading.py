"""Tests for ASR worker bootstrap configuration helpers.

These tests keep the server/worker credential boundary explicit without
reaching for monkeypatching. The helpers accept plain mappings so the policy
can be checked directly.
"""

from __future__ import annotations

import pytest

from batchalign.device import DevicePolicy
from batchalign.inference._domain_types import WhisperCpuPrecision
from batchalign.tests._qwen_recognition import (
    FAKE_QWEN_SNAPSHOTS,
    SEGMENTATION_MODEL_POLICY,
)
from batchalign.tests._snapshots import (
    fake_resolve_repository,
    fake_resolve_transformers,
)
from batchalign.worker._loads import ControlPlaneAsr, LoadedLocalAsr, LoadedProviderAsr
from batchalign.worker._model_loading import asr as asr_loading
from batchalign.worker._model_loading.asr import (
    PINNED_ASR_MODELS_KEY,
    WHISPER_DECODE_LANGUAGE_KEY,
    load_asr_engine,
    resolve_asr_engine,
    resolve_injected_revai_api_key,
)
from batchalign.worker._types import (
    AsrEngine,
    InferTask,
    WorkerBootstrapRuntime,
)
from batchalign.worker._types_v2 import (
    AsrBackendV2,
    ForceWhisperLanguageV2,
    LoadedModelV2,
    ObservedNotExposedV2,
    RequestedModelV2,
    RequestedProviderParameterV2,
    TencentModelIdentityV2,
    TencentModelsV2,
)

# A stand-in for the directory a resolved snapshot lives in, and the commit
# that directory holds. These tests are about DISPATCH: which engine a
# bootstrap loads. Resolving a real snapshot would make them depend on a warm
# Hugging Face cache, which is not a property a unit test may rely on.

# What the control plane sends an English stock Whisper worker.
_ENGLISH_DECODING = ForceWhisperLanguageV2(token="en")
_ENGLISH_SPAWN = {WHISPER_DECODE_LANGUAGE_KEY: _ENGLISH_DECODING.model_dump_json()}


class _FakeHandle:
    """A loaded-model stand-in carrying the identity a real handle is built
    with, which the load puts on the host it returns."""

    model_identity = "stand-in identity"


@pytest.fixture(autouse=True)
def _no_hub_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test in this module off the network and the model cache."""
    monkeypatch.setattr(asr_loading, "resolve_transformers", fake_resolve_transformers)
    monkeypatch.setattr(asr_loading, "resolve_repository", fake_resolve_repository)


class TestResolveInjectedRevaiApiKey:
    """Credential normalization stays at the explicit runtime boundary."""

    def test_empty_values_resolve_to_none(self) -> None:
        env = {"BATCHALIGN_REV_API_KEY": "   "}
        assert resolve_injected_revai_api_key(env) is None

    def test_injected_env_helper_normalizes_supported_keys(self) -> None:
        env = {"REVAI_API_KEY": "  from-env-helper "}
        assert resolve_injected_revai_api_key(env) == "from-env-helper"


class TestResolveAsrEngine:
    """Engine selection is the control plane's: typed, and loud on bad input.

    There is no default and no second chooser. The per-language default table
    (Cantonese to FunAudio), the Rev.AI-when-a-key-is-present rule and the
    Whisper fallback are gone, with their tests: the control plane names the
    engine for every worker-hosted ASR load, and names none only when the
    plan's ASR runs in Rust, which is ``ControlPlaneAsr``.
    """

    def test_override_wins(self) -> None:
        assert resolve_asr_engine({"asr": "whisper"}) is AsrEngine.WHISPER

    def test_whisper_hub_override_wins(self) -> None:
        assert resolve_asr_engine({"asr": "whisper_hub"}) is AsrEngine.WHISPER_HUB

    def test_resolve_returns_typed_enum_not_string(self) -> None:
        # Pin the return-type contract so a future refactor that
        # accidentally returns ``str`` (the historical shape) breaks
        # immediately rather than at the dispatch site.
        result = resolve_asr_engine({"asr": "tencent"})
        assert isinstance(result, AsrEngine)
        assert result is AsrEngine.TENCENT

    def test_qwen_override_wins(self) -> None:
        assert resolve_asr_engine({"asr": "qwen"}) is AsrEngine.QWEN

    def test_no_engine_named_is_the_control_plane_not_a_default(self) -> None:
        # No `asr` named: the plan's ASR runs in Rust. Not Whisper, not a
        # per-language engine, whatever else the overrides carry.
        assert isinstance(resolve_asr_engine(None), ControlPlaneAsr)
        assert isinstance(resolve_asr_engine(_ENGLISH_SPAWN), ControlPlaneAsr)

    def test_rev_is_not_a_worker_engine(self) -> None:
        # Rev.AI runs in the control plane, which never names it to a worker.
        with pytest.raises(ValueError, match="unknown asr engine 'rev'"):
            resolve_asr_engine({"asr": "rev"})

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown asr engine 'wisper'"):
            resolve_asr_engine({"asr": "wisper"})

    def test_unknown_engine_error_mentions_supported_options(self) -> None:
        # The supported-engines list is derived from the AsrEngine
        # enum, so adding a variant requires zero changes here.
        with pytest.raises(ValueError) as exc_info:
            resolve_asr_engine({"asr": "x"})
        msg = str(exc_info.value)
        for variant in AsrEngine:
            assert variant.value in msg, (
                f"error message {msg!r} missing variant {variant.value!r}"
            )


def _refuse_every_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every ASR loader with one that records its own name.

    For tests that assert NOTHING loads: a call is recorded rather than
    raised, so a regression reports which model it would have loaded.
    """
    called: list[str] = []

    def recorder(name: str):
        def _load(*_args, **_kwargs):
            called.append(name)
            return _FakeHandle()

        return _load

    for target in (
        "batchalign.inference.asr.load_whisper_asr",
        "batchalign.inference.whisper_hub.load_whisper_hub_asr",
        "batchalign.inference.languages.cantonese._funaudio_asr.load_funaudio_asr",
        "batchalign.inference.languages.cantonese._tencent_asr.load_tencent_asr",
        "batchalign.inference.languages.cantonese._aliyun_asr.load_aliyun_asr",
        "batchalign.inference.languages.cantonese._qwen_asr.load_qwen_asr",
    ):
        monkeypatch.setattr(target, recorder(target.rsplit(".", 1)[1]))
    return called


@pytest.mark.parametrize("revai_api_key", [None, "from-rust"])
@pytest.mark.parametrize("lang", ["eng", "yue"])
def test_an_asr_load_naming_no_engine_loads_no_model(
    monkeypatch: pytest.MonkeyPatch, revai_api_key: str | None, lang: str
) -> None:
    """The capability probe of a job whose ASR runs in Rust (Rev.AI) prepares
    the ASR task naming no engine. That load builds nothing, with or without
    a Rev.AI key in the worker's environment and whatever the language.

    It used to fall through to Rev.AI when a key was present and otherwise to
    a guessed engine (FunAudio for Cantonese, Whisper for anything else),
    loading a multi-gigabyte model the job never used.
    """
    called = _refuse_every_loader(monkeypatch)

    host = load_asr_engine(
        WorkerBootstrapRuntime(
            task=InferTask.ASR,
            lang=lang,
            num_speakers=1,
            engine_overrides=_ENGLISH_SPAWN if lang == "eng" else {},
            revai_api_key=revai_api_key,
        )
    )

    assert isinstance(host, ControlPlaneAsr)
    assert called == []


def test_whisper_override_does_not_require_legacy_config(monkeypatch) -> None:
    """Whisper bootstrap should not touch legacy config-discovery paths."""

    def fake_load_whisper_asr(snapshot, *, decoding, device_policy, cpu_precision):
        assert decoding == _ENGLISH_DECODING
        assert device_policy == DevicePolicy(force_cpu=True)
        # No `whisper_cpu_dtype` extra on this job: the default precision.
        assert cpu_precision is WhisperCpuPrecision.FLOAT32
        # The worker resolves the snapshot and hands the loader that proof,
        # never an id it could download by; unpinned, it is the stock model.
        assert snapshot.model_id == asr_loading.STOCK_WHISPER_ID
        return _FakeHandle()

    monkeypatch.setattr(
        "batchalign.inference.asr.load_whisper_asr",
        fake_load_whisper_asr,
    )

    host = load_asr_engine(
        WorkerBootstrapRuntime(
            task=InferTask.ASR,
            lang="eng",
            num_speakers=1,
            engine_overrides={"asr": "whisper", **_ENGLISH_SPAWN},
            device_policy=DevicePolicy(force_cpu=True),
        )
    )

    # The host carries the backend that loaded and the identity the handle
    # was built with: one value, never a field written beside it.
    assert isinstance(host, LoadedLocalAsr)
    assert host.backend is AsrBackendV2.LOCAL_WHISPER
    assert host.model_identity == _FakeHandle.model_identity


def test_whisper_hub_override_dispatches_to_whisper_hub_loader(monkeypatch) -> None:
    """Worker dispatch must route asr=whisper_hub to the fine-tune loader."""

    captured: dict[str, object] = {}

    def fake_load_whisper_hub_asr(snapshot, *, device_policy):
        captured["device_policy"] = device_policy
        captured["snapshot"] = snapshot
        return _FakeHandle()

    monkeypatch.setattr(
        "batchalign.inference.whisper_hub.load_whisper_hub_asr",
        fake_load_whisper_hub_asr,
    )

    host = load_asr_engine(
        WorkerBootstrapRuntime(
            task=InferTask.ASR,
            lang="mal",
            num_speakers=1,
            engine_overrides={"asr": "whisper_hub"},
            device_policy=DevicePolicy(force_cpu=True),
        )
    )

    assert captured["device_policy"] == DevicePolicy(force_cpu=True)
    # Unpinned, the language's table entry is what was resolved.
    assert captured["snapshot"].model_id == "thennal/whisper-medium-ml"
    assert isinstance(host, LoadedLocalAsr)
    assert host.backend is AsrBackendV2.WHISPER_HUB
    assert host.model_identity == _FakeHandle.model_identity


def test_a_stock_whisper_load_without_a_decode_language_is_refused() -> None:
    """A stock Whisper load whose spawn carried no decode language refuses at
    load, naming the key, rather than deciding a language itself."""
    with pytest.raises(ValueError, match=WHISPER_DECODE_LANGUAGE_KEY):
        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="nan",
                num_speakers=1,
                engine_overrides={"asr": "whisper"},
            )
        )


def test_tencent_override_uses_injected_boundary_credentials(monkeypatch) -> None:
    """HK bootstrap should delegate credential discovery to the Rust boundary."""

    captured: dict[str, object] = {}

    class _FakeTencent:
        def transcribe(self, _item):
            raise AssertionError("loading must not transcribe")

    loaded = _FakeTencent()

    # Mirrors the real loader's signature, which takes no engine overrides:
    # Tencent's one knob is the engine-model type, and the control plane
    # chooses it.
    def fake_load_tencent_asr(lang, *, engine_model_type, config=None):
        captured["lang"] = lang
        captured["engine_model_type"] = engine_model_type
        captured["config"] = config
        return loaded

    # Tencent selects its model by a parameter the control plane chose, so
    # the worker refuses to load without one. The pin travels IN
    # `engine_overrides`, exactly as the control plane sends it, and is a load
    # input: `load_asr_engine` reads it before any loader runs.
    pinned = TencentModelsV2(
        engine_model_type=RequestedModelV2(
            id="tencent-asr",
            revision=RequestedProviderParameterV2(parameter="16k_yue"),
        )
    )
    overrides = {
        "asr": "tencent",
        PINNED_ASR_MODELS_KEY: pinned.model_dump_json(),
    }

    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._tencent_asr.load_tencent_asr",
        fake_load_tencent_asr,
    )

    host = load_asr_engine(
        WorkerBootstrapRuntime(
            task=InferTask.ASR,
            lang="yue",
            num_speakers=1,
            engine_overrides=overrides,
        )
    )

    # What the loader is GIVEN: the chosen parameter alone.
    assert captured == {
        "lang": "yue",
        # Chosen by the Rust control plane and carried through the pin,
        # not derived from the language by this worker.
        "engine_model_type": "16k_yue",
        "config": None,
    }
    assert isinstance(host, LoadedProviderAsr)
    assert host.backend is AsrBackendV2.HK_TENCENT
    assert host.runner == loaded.transcribe
    # The identity is the declared member: Tencent reports no revision back.
    assert host.model_identity == TencentModelIdentityV2(
        engine_model_type=LoadedModelV2(
            id="tencent-asr",
            requested=RequestedProviderParameterV2(parameter="16k_yue"),
            observed=ObservedNotExposedV2(),
        )
    )


def test_a_pin_for_another_engine_is_refused_at_load(monkeypatch) -> None:
    """The pin is a load input checked against the engine being loaded: a
    Tencent pin on a Qwen load means the two sides disagree, and nothing
    loads."""
    called = _refuse_every_loader(monkeypatch)
    pinned = TencentModelsV2(
        engine_model_type=RequestedModelV2(
            id="tencent-asr",
            revision=RequestedProviderParameterV2(parameter="16k_yue"),
        )
    )
    with pytest.raises(ValueError, match="pinned a 'tencent' ASR composition"):
        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="yue",
                num_speakers=1,
                engine_overrides={
                    "asr": "qwen",
                    PINNED_ASR_MODELS_KEY: pinned.model_dump_json(),
                },
            )
        )
    assert called == []


def test_load_qwen_asr_times_out_on_hang(monkeypatch) -> None:
    """If ``QwenRecognizerConfig.load()`` hangs (the load stuck in
    ``from_pretrained``, then the qwen-asr package's), the loader
    must raise ``TimeoutError`` within the configured timeout, not
    sit silently at 0% CPU for hours.

    Origin: 2026-05-27 v2 benchmark Bucket A, worker sat at 0% CPU
    for 70+ min on the first fixture with no Qwen model ever
    downloaded; the qwen-asr package's first-use code path blocked
    indefinitely with no observable progress. This test pins the
    timeout-and-fail-loudly contract that prevents that wasted
    compute from recurring.
    """
    import time

    from batchalign.inference.languages.cantonese import _qwen_asr

    # Force a short timeout so the test completes in ~1 second instead
    # of waiting out the production-default 1200 s.
    monkeypatch.setattr(_qwen_asr, "_QWEN_LOAD_TIMEOUT_SECONDS", 1)

    hang_invocations: list[int] = []

    def hang_warm(self, _snapshots) -> None:
        """Simulate the production hang: ``load()`` never returns."""
        hang_invocations.append(1)
        # Sleep longer than the configured timeout. If the loader
        # honours the timeout, the test takes ~1 second; if it
        # doesn't, it would wait the full 10 s and the elapsed
        # assertion below fails.
        time.sleep(10)

    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._qwen_common.QwenRecognizerConfig.load",
        hang_warm,
    )

    start = time.monotonic()
    with pytest.raises(TimeoutError, match=r"Qwen3-ASR.*timed out after 1 second"):
        _qwen_asr.load_qwen_asr(
            "yue",
            None,
            recognition=SEGMENTATION_MODEL_POLICY,
            snapshots=FAKE_QWEN_SNAPSHOTS,
        )
    elapsed = time.monotonic() - start

    assert hang_invocations == [1], "load() should have been entered exactly once"
    # The timeout must fire fast (1 s configured + a small handler
    # delivery slack). If we took anywhere near the 10 s sleep the
    # interrupt mechanism is broken.
    assert elapsed < 5.0, (
        f"Qwen loader did not honour the {_qwen_asr._QWEN_LOAD_TIMEOUT_SECONDS}s "
        f"timeout: elapsed {elapsed:.2f}s; expected <5s. The signal-based "
        f"alarm is not interrupting ``load()`` as designed."
    )
