"""Unit tests for HK provider wrapper modules.

Each provider's load returns the provider object it built (``TencentAsr``,
``FunAudioAsr``, ``AliyunAsr``); a request transcribes through that object.
There is no module state and no "not loaded" path to test: a provider exists
only after its load. The legacy batch wrappers (``infer_*_asr``) are gone with
the module state they read; per-item batch handling has no route here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import batchalign.inference.languages.cantonese._aliyun_asr as aliyun_asr
import batchalign.inference.languages.cantonese._funaudio_asr as funaudio_asr
import batchalign.inference.languages.cantonese._tencent_asr as tencent_asr
from batchalign.inference.asr import (
    AsrBatchItem,
    AsrElement,
    AsrMonologue,
    AttributedSpeaker,
    MonologueAsrResponse,
)
from batchalign.worker._types_v2 import (
    NotRequestedDiarizationV2,
    ProviderDiarizationV2,
)


def _item(diarization: ProviderDiarizationV2 | None = None) -> AsrBatchItem:
    """One provider ASR item, defaulting to the no-separation request.

    The default lives HERE, in the tests' own helper, rather than on the model:
    these tests are about payload handling, and the model has to keep making
    every caller say what it asked the provider for.
    """
    return AsrBatchItem(
        audio_path="clip.wav",
        lang="yue",
        diarization=(
            diarization if diarization is not None else NotRequestedDiarizationV2()
        ),
    )


def _valid_response(lang: str = "yue") -> MonologueAsrResponse:
    return MonologueAsrResponse(
        lang=lang,
        monologues=[
            AsrMonologue(
                speaker=AttributedSpeaker(label="1"),
                elements=[
                    AsrElement(value="好", ts=0.0, end_ts=0.2, type="text"),
                    AsrElement(value="   ", ts=None, end_ts=None, type="text"),
                ],
            )
        ],
    )


@dataclass
class _FakeFunRecognizer:
    payload: tuple[dict[str, object], list[dict[str, object]]] | None = None
    error: Exception | None = None

    def transcribe(self, _audio_path: str):
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            return self.payload
        return (
            {
                "monologues": [
                    {
                        # FunASR separates no speakers, so it names none.
                        "speaker": {"kind": "undiarized"},
                        "elements": [
                            {"value": "好", "ts": 0.0, "end_ts": 0.2, "type": "text"},
                            {
                                "value": "   ",
                                "ts": None,
                                "end_ts": None,
                                "type": "text",
                            },
                        ],
                    }
                ]
            },
            [],
        )


@dataclass
class _FakeTencentRecognizer:
    error: Exception | None = None

    def transcribe(self, _audio_path: str, diarization: ProviderDiarizationV2):
        if self.error is not None:
            raise self.error
        return [{"diarization": diarization.kind}]

    def monologues(self, _details):
        return {
            "monologues": [
                {
                    # Tencent does separate speakers, so it names one.
                    "speaker": {"kind": "attributed", "label": "1"},
                    "elements": [
                        {"value": "好", "ts": 0.0, "end_ts": 0.2, "type": "text"},
                        {"value": "   ", "ts": None, "end_ts": None, "type": "text"},
                    ],
                }
            ]
        }


def test_load_funaudio_asr_loads_the_given_snapshots_on_the_override_device(
    monkeypatch,
) -> None:
    from batchalign.inference.languages.cantonese._funaudio_common import (
        SenseVoiceSnapshots,
    )
    from batchalign.tests._snapshots import repository_snapshot

    created: dict[str, object] = {}

    class FakeRecognizer:
        @classmethod
        def load(cls, snapshots, *, lang: str, device: str, label: str) -> object:
            created.update(
                {"snapshots": snapshots, "lang": lang, "device": device, "label": label}
            )
            return cls()

    monkeypatch.setattr(funaudio_asr, "FunAudioRecognizer", FakeRecognizer)
    snapshots = SenseVoiceSnapshots(
        asr=repository_snapshot("org/asr"),
        vad=repository_snapshot("org/vad"),
    )

    provider = funaudio_asr.load_funaudio_asr(
        "yue",
        {"funaudio_model": "ignored-here", "funaudio_device": "cuda"},
        snapshots=snapshots,
        label="FunAudioLLM/SenseVoiceSmall",
    )

    # The checkpoint is whatever the worker materialized; `funaudio_model` was
    # already resolved into the pinned composition by the control plane.
    assert created == {
        "snapshots": snapshots,
        "lang": "yue",
        "device": "cuda",
        "label": "FunAudioLLM/SenseVoiceSmall",
    }
    # The load hands back what it built; nothing is stored at module level.
    assert isinstance(provider.recognizer, FakeRecognizer)
    assert not hasattr(funaudio_asr, "_recognizer")


def test_funaudio_transcribe_surfaces_runtime_errors() -> None:
    provider = funaudio_asr.FunAudioAsr(_FakeFunRecognizer(error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        provider.transcribe(_item())


def test_funaudio_transcribe_filters_blank_elements() -> None:
    response = funaudio_asr.FunAudioAsr(_FakeFunRecognizer()).transcribe(_item())

    assert response.lang == "yue"
    assert [element.value for element in response.monologues[0].elements] == ["好"]


def test_load_tencent_asr_returns_the_recognizer_it_built(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeRecognizer:
        def __init__(self, *, lang: str, engine_model_type: str, config=None) -> None:
            created.update(
                {
                    "lang": lang,
                    "engine_model_type": engine_model_type,
                    "config": config,
                }
            )

    monkeypatch.setattr(tencent_asr, "TencentRecognizer", FakeRecognizer)

    provider = tencent_asr.load_tencent_asr(
        "yue", engine_model_type="16k_zh_en_2.0", config={"cfg": True}
    )

    assert created == {
        "lang": "yue",
        "engine_model_type": "16k_zh_en_2.0",
        "config": {"cfg": True},
    }
    assert isinstance(provider.recognizer, FakeRecognizer)
    assert not hasattr(tencent_asr, "_recognizer")


def test_tencent_transcribe_surfaces_runtime_errors() -> None:
    provider = tencent_asr.TencentAsr(
        _FakeTencentRecognizer(error=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        provider.transcribe(_item())


def test_tencent_transcribe_filters_blank_elements() -> None:
    response = tencent_asr.TencentAsr(_FakeTencentRecognizer()).transcribe(_item())

    assert response.lang == "yue"
    assert [element.value for element in response.monologues[0].elements] == ["好"]


def test_aliyun_transcribe_runs_the_item_audio_with_the_cached_token(
    monkeypatch,
) -> None:
    """One provider object carries the account, so a transcription reads its
    credentials and token from there, never from module state."""
    expected = _valid_response()
    seen: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *, token: str, appkey: str, wav_path: str) -> None:
            seen.update({"token": token, "appkey": appkey, "wav_path": wav_path})

        def start(self) -> list[object]:
            return []

    monkeypatch.setattr(aliyun_asr, "_request_token", lambda _id, _secret: "tok")
    monkeypatch.setattr(aliyun_asr, "_AliyunRunner", FakeRunner)
    monkeypatch.setattr(aliyun_asr, "_ensure_wav", lambda path: (path, None))
    monkeypatch.setattr(aliyun_asr, "_project_results", lambda _results: expected)
    provider = aliyun_asr.AliyunAsr(ak_id="id", ak_secret="secret", appkey="appkey")

    response = provider.transcribe(_item())

    assert response == expected
    assert seen == {"token": "tok", "appkey": "appkey", "wav_path": "clip.wav"}
