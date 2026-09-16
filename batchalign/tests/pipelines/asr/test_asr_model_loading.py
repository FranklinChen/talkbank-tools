"""Tests for ASR worker bootstrap configuration helpers.

These tests keep the server/worker credential boundary explicit without
reaching for monkeypatching. The helpers accept plain mappings so the policy
can be checked directly.
"""

from __future__ import annotations

import pytest

from batchalign.device import DevicePolicy
from batchalign.worker._model_loading import asr as asr_loading
from batchalign.worker._model_loading.asr import (
    PINNED_ASR_MODELS_KEY,
    load_asr_engine,
    resolve_asr_engine,
    resolve_injected_revai_api_key,
)
from batchalign.worker._model_loading.pinned_hub import ResolvedSnapshot
from batchalign.worker._types import (
    AsrEngine,
    InferTask,
    WorkerBootstrapRuntime,
    _state,
)
from batchalign.worker._types_v2 import (
    RequestedModelV2,
    RequestedProviderParameterV2,
    TencentModelsV2,
)

# A stand-in for the directory a resolved snapshot lives in, and the commit
# that directory holds. These tests are about DISPATCH: which engine a
# bootstrap loads. Resolving a real snapshot would make them depend on a warm
# Hugging Face cache, which is not a property a unit test may rely on.
_FAKE_SNAPSHOT_PATH = "/cache/models--fake/snapshots/" + "a" * 40
_FAKE_SNAPSHOT_COMMIT = "a" * 40


class _FakeHandle:
    """A loaded-model stand-in with settable attributes.

    The loader records the identity it resolved onto the handle it got back,
    so a bare string here would fail with `AttributeError` and say nothing
    about dispatch.
    """


@pytest.fixture(autouse=True)
def _no_hub_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test in this module off the network and the model cache."""
    monkeypatch.setattr(
        asr_loading,
        "resolve_pinned_snapshot",
        lambda model_id, commit: ResolvedSnapshot(
            path=_FAKE_SNAPSHOT_PATH, commit=commit or _FAKE_SNAPSHOT_COMMIT
        ),
    )


class TestResolveInjectedRevaiApiKey:
    """Credential normalization stays at the explicit runtime boundary."""

    def test_empty_values_resolve_to_none(self) -> None:
        env = {"BATCHALIGN_REV_API_KEY": "   "}
        assert resolve_injected_revai_api_key(env) is None

    def test_injected_env_helper_normalizes_supported_keys(self) -> None:
        env = {"REVAI_API_KEY": "  from-env-helper "}
        assert resolve_injected_revai_api_key(env) == "from-env-helper"


class TestResolveAsrEngine:
    """Engine selection must stay deterministic, typed, and loud on bad input."""

    def test_override_wins(self) -> None:
        assert (
            resolve_asr_engine({"asr": "whisper"}, "from-config", lang="eng")
            is AsrEngine.WHISPER
        )

    def test_rev_is_selected_when_key_present(self) -> None:
        assert resolve_asr_engine(None, "from-config", lang="eng") is AsrEngine.REV

    def test_whisper_is_default_without_key(self) -> None:
        assert resolve_asr_engine(None, None, lang="eng") is AsrEngine.WHISPER

    def test_whisper_hub_override_wins(self) -> None:
        assert (
            resolve_asr_engine({"asr": "whisper_hub"}, "from-config", lang="eng")
            is AsrEngine.WHISPER_HUB
        )

    def test_resolve_returns_typed_enum_not_string(self) -> None:
        # Pin the return-type contract so a future refactor that
        # accidentally returns ``str`` (the historical shape) breaks
        # immediately rather than at the dispatch site.
        result = resolve_asr_engine({"asr": "tencent"}, None, lang="yue")
        assert isinstance(result, AsrEngine)
        assert result is AsrEngine.TENCENT

    def test_qwen_override_wins(self) -> None:
        # Qwen3-ASR is an open-weight Cantonese-capable ASR option;
        # this test pins that an explicit override routes to it
        # regardless of the per-language default for ``yue``.
        assert resolve_asr_engine({"asr": "qwen"}, None, lang="yue") is AsrEngine.QWEN

    # Per-language defaults (yue → FunASR). Explicit overrides and
    # Rev key still win; languages absent from the table fall through
    # to Whisper.

    def test_yue_without_rev_key_or_override_defaults_to_funaudio(self) -> None:
        assert resolve_asr_engine(None, None, lang="yue") is AsrEngine.FUNAUDIO

    def test_yue_with_explicit_override_still_uses_override(self) -> None:
        assert resolve_asr_engine({"asr": "qwen"}, None, lang="yue") is AsrEngine.QWEN

    def test_yue_with_rev_key_still_uses_rev(self) -> None:
        assert resolve_asr_engine(None, "from-config", lang="yue") is AsrEngine.REV

    def test_eng_without_rev_key_still_defaults_to_whisper(self) -> None:
        assert resolve_asr_engine(None, None, lang="eng") is AsrEngine.WHISPER

    def test_unknown_lang_falls_back_to_global_default(self) -> None:
        assert resolve_asr_engine(None, None, lang="mri") is AsrEngine.WHISPER

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown asr engine 'wisper'"):
            resolve_asr_engine({"asr": "wisper"}, None, lang="eng")

    def test_unknown_engine_error_mentions_supported_options(self) -> None:
        # The supported-engines list is derived from the AsrEngine
        # enum, so adding a 7th variant requires zero changes here.
        with pytest.raises(ValueError) as exc_info:
            resolve_asr_engine({"asr": "x"}, None, lang="eng")
        msg = str(exc_info.value)
        for variant in AsrEngine:
            assert variant.value in msg, (
                f"error message {msg!r} missing variant {variant.value!r}"
            )


def test_whisper_override_does_not_require_legacy_config(monkeypatch) -> None:
    """Whisper bootstrap should not touch legacy config-discovery paths."""

    def fake_load_whisper_asr(*, model, base, language, device_policy):
        assert language == "english"
        assert device_policy == DevicePolicy(force_cpu=True)
        # The worker resolves the snapshot and hands the loader a local path,
        # so both coordinates are the resolved directory rather than an id.
        assert model == base == _FAKE_SNAPSHOT_PATH
        return _FakeHandle()

    old_model = _state.whisper_asr_model
    old_engine = _state.asr_engine
    old_key = _state.rev_api_key
    try:
        monkeypatch.setattr(
            "batchalign.inference.asr.load_whisper_asr",
            fake_load_whisper_asr,
        )

        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="eng",
                num_speakers=1,
                engine_overrides={"asr": "whisper"},
                device_policy=DevicePolicy(force_cpu=True),
            )
        )

        assert isinstance(_state.whisper_asr_model, _FakeHandle)
        assert _state.asr_engine is AsrEngine.WHISPER
        # The loaded identity travels with the handle and on the worker state.
        assert _state.asr_model_identity is not None
    finally:
        _state.whisper_asr_model = old_model
        _state.asr_engine = old_engine
        _state.rev_api_key = old_key


def test_injected_revai_key_selects_rev_without_legacy_config() -> None:
    """Rev.AI selection should rely on the Rust-injected key, not config rediscovery."""

    old_engine = _state.asr_engine
    old_key = _state.rev_api_key
    try:
        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="eng",
                num_speakers=1,
                revai_api_key=" from-rust ",
            )
        )

        assert _state.asr_engine is AsrEngine.REV
        assert _state.rev_api_key == " from-rust "
    finally:
        _state.asr_engine = old_engine
        _state.rev_api_key = old_key


def test_whisper_hub_override_dispatches_to_whisper_hub_loader(monkeypatch) -> None:
    """Worker dispatch must route asr=whisper_hub to the fine-tune loader.

    RED when ``AsrEngine.WHISPER_HUB`` does not yet exist and when
    ``_model_loading.asr.load_asr_engine`` has no whisper_hub branch.
    """

    captured: dict[str, object] = {}
    old_engine = _state.asr_engine
    old_model = _state.whisper_asr_model
    try:

        def fake_load_whisper_hub_asr(
            lang, engine_overrides, *, device_policy, model_path
        ):
            captured["lang"] = lang
            captured["engine_overrides"] = engine_overrides
            captured["device_policy"] = device_policy
            captured["model_path"] = model_path
            return _FakeHandle()

        monkeypatch.setattr(
            "batchalign.inference.whisper_hub.load_whisper_hub_asr",
            fake_load_whisper_hub_asr,
        )

        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="mal",
                num_speakers=1,
                engine_overrides={"asr": "whisper_hub"},
                device_policy=DevicePolicy(force_cpu=True),
            )
        )

        assert captured["lang"] == "mal"
        assert captured["engine_overrides"] == {"asr": "whisper_hub"}
        assert captured["device_policy"] == DevicePolicy(force_cpu=True)
        assert captured["model_path"] == _FAKE_SNAPSHOT_PATH
        assert isinstance(_state.whisper_asr_model, _FakeHandle)
        assert _state.asr_engine is AsrEngine.WHISPER_HUB
    finally:
        _state.asr_engine = old_engine
        _state.whisper_asr_model = old_model


# Top-level integration tests at the worker-bootstrap seam
# (``load_asr_engine(WorkerBootstrapRuntime)``). Resolver unit tests
# in ``TestResolveAsrEngine`` above are supplements, not substitutes.


def test_load_asr_engine_yue_with_no_override_or_rev_dispatches_to_funaudio(
    monkeypatch,
) -> None:
    """yue worker with no overrides and no Rev key must load FunASR,
    not vanilla Whisper-large-v3 (the worst measured engine on
    Cantonese in the 2026-05-26 v2 benchmark, 81.9% CER on Tier 3)."""

    funaudio_calls: list[tuple[object, object]] = []
    whisper_calls: list[dict[str, object]] = []

    def fake_load_funaudio_asr(lang, engine_overrides, **_pinned):
        funaudio_calls.append((lang, engine_overrides))

    def fake_load_whisper_asr(**kwargs):
        # If this gets called, Fix 3 is not done, the bug is alive.
        # Capture so the test failure is informative rather than crashy.
        whisper_calls.append(kwargs)
        return _FakeHandle()

    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._funaudio_asr.load_funaudio_asr",
        fake_load_funaudio_asr,
    )
    monkeypatch.setattr(
        "batchalign.inference.asr.load_whisper_asr",
        fake_load_whisper_asr,
    )

    old_engine = _state.asr_engine
    old_whisper = _state.whisper_asr_model
    old_key = _state.rev_api_key
    try:
        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="yue",
                num_speakers=1,
                # No engine_overrides, no revai_api_key; this is the
                # exact bootstrap shape that produced 81.9% CER in v2.
            )
        )

        assert whisper_calls == [], (
            "yue worker with no override + no Rev key MUST NOT load "
            "vanilla Whisper-large-v3 (Fix 3 regression). "
            f"whisper called with: {whisper_calls}"
        )
        assert len(funaudio_calls) == 1, (
            "yue worker with no override + no Rev key must load FunASR. "
            f"funaudio_calls={funaudio_calls}"
        )
        assert funaudio_calls[0] == ("yue", None)
        assert _state.asr_engine is AsrEngine.FUNAUDIO
    finally:
        _state.asr_engine = old_engine
        _state.whisper_asr_model = old_whisper
        _state.rev_api_key = old_key


def test_load_asr_engine_eng_with_no_override_or_rev_still_loads_whisper(
    monkeypatch,
) -> None:
    """Languages without a per-language default entry keep the global
    Whisper fallback."""

    funaudio_calls: list[tuple[object, object]] = []

    def fake_load_funaudio_asr(lang, engine_overrides, **_pinned):
        funaudio_calls.append((lang, engine_overrides))

    def fake_load_whisper_asr(**kwargs):
        return _FakeHandle()

    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._funaudio_asr.load_funaudio_asr",
        fake_load_funaudio_asr,
    )
    monkeypatch.setattr(
        "batchalign.inference.asr.load_whisper_asr",
        fake_load_whisper_asr,
    )

    old_engine = _state.asr_engine
    old_whisper = _state.whisper_asr_model
    old_key = _state.rev_api_key
    try:
        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="eng",
                num_speakers=1,
            )
        )

        assert funaudio_calls == [], (
            f"eng worker must not load FunASR. funaudio_calls={funaudio_calls}"
        )
        assert _state.asr_engine is AsrEngine.WHISPER
        assert isinstance(_state.whisper_asr_model, _FakeHandle)
    finally:
        _state.asr_engine = old_engine
        _state.whisper_asr_model = old_whisper
        _state.rev_api_key = old_key


def test_tencent_override_uses_injected_boundary_credentials(monkeypatch) -> None:
    """HK bootstrap should delegate credential discovery to the Rust boundary."""

    captured: dict[str, object] = {}
    old_engine = _state.asr_engine
    old_pinned = _state.asr_pinned_models
    try:
        # Mirrors the real loader's signature, which takes no engine overrides:
        # Tencent's one knob is the engine-model type, and the control plane
        # chooses it. A double that still accepted overrides would keep
        # asserting a parameter the loader no longer has.
        def fake_load_tencent_asr(lang, *, engine_model_type, config=None):
            captured["lang"] = lang
            captured["engine_model_type"] = engine_model_type
            captured["config"] = config

        # Tencent selects its model by a parameter the control plane chose, so
        # the worker refuses to load without one. The pin travels IN
        # `engine_overrides`, exactly as the control plane sends it: setting
        # worker state directly would not survive, because `load_asr_engine`
        # re-reads the pin from the overrides before any loader runs.
        pinned = TencentModelsV2(
            engine_model_type=RequestedModelV2(
                id="tencent-asr",
                revision=RequestedProviderParameterV2(parameter="16k_zh_large"),
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

        load_asr_engine(
            WorkerBootstrapRuntime(
                task=InferTask.ASR,
                lang="yue",
                num_speakers=1,
                engine_overrides=overrides,
            )
        )

        # What the loader is GIVEN, which is no longer the overrides. The pin
        # travels in `engine_overrides` to `load_asr_engine`, which reads it
        # before any loader runs and passes down the chosen parameter alone;
        # the loader had no use for the rest and was carrying it unread.
        assert captured == {
            "lang": "yue",
            # Chosen by the Rust control plane and carried through the pin,
            # not derived from the language by this worker.
            "engine_model_type": "16k_zh_large",
            "config": None,
        }
        assert _state.asr_engine is AsrEngine.TENCENT
    finally:
        _state.asr_engine = old_engine
        # Restored, or this pin leaks into every later test in the session and
        # turns unrelated failures into order-dependent ones.
        _state.asr_pinned_models = old_pinned


def test_load_qwen_asr_times_out_on_hang(monkeypatch) -> None:
    """If ``QwenRecognizer.warm()`` hangs (model download/load stuck
    in the qwen-asr package's ``from_pretrained`` call), the loader
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

    def hang_warm(self) -> None:
        """Simulate the production hang: ``warm()`` never returns."""
        hang_invocations.append(1)
        # Sleep longer than the configured timeout. If the loader
        # honours the timeout, the test takes ~1 second; if it
        # doesn't, it would wait the full 10 s and the elapsed
        # assertion below fails.
        time.sleep(10)

    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._qwen_common.QwenRecognizer.warm",
        hang_warm,
    )

    start = time.monotonic()
    with pytest.raises(TimeoutError, match=r"Qwen3-ASR.*timed out after 1 second"):
        _qwen_asr.load_qwen_asr("yue", {"qwen_model": "Qwen/Qwen3-ASR-0.6B-hf"})
    elapsed = time.monotonic() - start

    assert hang_invocations == [1], "warm() should have been entered exactly once"
    # The timeout must fire fast (1 s configured + a small handler
    # delivery slack). If we took anywhere near the 10 s sleep the
    # interrupt mechanism is broken.
    assert elapsed < 5.0, (
        f"Qwen loader did not honour the {_qwen_asr._QWEN_LOAD_TIMEOUT_SECONDS}s "
        f"timeout: elapsed {elapsed:.2f}s; expected <5s. The signal-based "
        f"alarm is not interrupting ``warm()`` as designed."
    )
