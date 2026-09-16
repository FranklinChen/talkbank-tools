"""Unit tests for HK provider wrapper modules."""

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
from batchalign.worker._types import BatchInferRequest, InferTask
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


def _valid_request() -> BatchInferRequest:
    return BatchInferRequest(
        task=InferTask.ASR,
        lang="yue",
        items=[_item().model_dump()],
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


def test_load_funaudio_asr_applies_engine_overrides(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeRecognizer:
        def __init__(
            self,
            *,
            lang: str,
            model: str,
            device: str,
            model_path: str | None = None,
            model_revision: str | None = None,
            vad_model: str | None = None,
            vad_model_path: str | None = None,
            vad_revision: str | None = None,
            punc_model: str | None = None,
            punc_revision: str | None = None,
        ) -> None:
            created.update(
                {
                    "lang": lang,
                    "model": model,
                    "device": device,
                    "model_path": model_path,
                    "vad_model_path": vad_model_path,
                }
            )

    monkeypatch.setattr(funaudio_asr, "FunAudioRecognizer", FakeRecognizer)
    monkeypatch.setattr(funaudio_asr, "_recognizer", None)

    funaudio_asr.load_funaudio_asr(
        "yue",
        {"funaudio_model": "custom-model", "funaudio_device": "cuda"},
    )

    # A direct caller passes no pin, and the loader must not invent one: the
    # engine overrides still decide the checkpoint, and the resolved-snapshot
    # arguments stay absent.
    assert created == {
        "lang": "yue",
        "model": "custom-model",
        "device": "cuda",
        "model_path": None,
        "vad_model_path": None,
    }


def test_infer_funaudio_asr_requires_loaded_recognizer(monkeypatch) -> None:
    monkeypatch.setattr(funaudio_asr, "_recognizer", None)

    response = funaudio_asr.infer_funaudio_asr(_valid_request())

    # Compared against the module's own constant, not a second copy of the
    # sentence. What this test is for is that the not-loaded path reports THAT
    # error rather than some other one; the wording is not the behaviour, and
    # pinning it here is what let a punctuation edit fail a test about model
    # loading.
    assert response.results[0].error == funaudio_asr.NOT_LOADED_ERROR


def test_infer_funaudio_asr_rejects_invalid_items(monkeypatch) -> None:
    monkeypatch.setattr(funaudio_asr, "_recognizer", _FakeFunRecognizer())
    request = BatchInferRequest(task=InferTask.ASR, lang="yue", items=[{}])

    response = funaudio_asr.infer_funaudio_asr(request)

    assert response.results[0].error == "Invalid AsrBatchItem"


def test_infer_funaudio_asr_surfaces_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        funaudio_asr,
        "_recognizer",
        _FakeFunRecognizer(error=RuntimeError("boom")),
    )

    response = funaudio_asr.infer_funaudio_asr(_valid_request())

    assert response.results[0].error == "boom"


def test_infer_funaudio_asr_returns_result_payload(monkeypatch) -> None:
    monotonic_values = iter([0.0, 1.5])
    monkeypatch.setattr(funaudio_asr, "_recognizer", _FakeFunRecognizer())
    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._funaudio_asr.time.monotonic",
        lambda: next(monotonic_values),
    )

    response = funaudio_asr.infer_funaudio_asr(_valid_request())

    assert response.results[0].error is None
    assert response.results[0].result["monologues"][0]["elements"][0]["value"] == "好"
    assert response.results[0].elapsed_s == 1.5


def test_funaudio_transcribe_to_monologues_requires_loaded_recognizer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(funaudio_asr, "_recognizer", None)

    with pytest.raises(RuntimeError, match="not initialized"):
        funaudio_asr._transcribe_to_monologues(_item())


def test_infer_funaudio_asr_v2_filters_blank_elements(monkeypatch) -> None:
    monkeypatch.setattr(funaudio_asr, "_recognizer", _FakeFunRecognizer())

    response = funaudio_asr.infer_funaudio_asr_v2(_item())

    assert response.lang == "yue"
    assert [element.value for element in response.monologues[0].elements] == ["好"]


def test_infer_tencent_asr_requires_loaded_provider(monkeypatch) -> None:
    monkeypatch.setattr(tencent_asr, "_recognizer", None)

    response = tencent_asr.infer_tencent_asr(_valid_request())

    assert response.results[0].error == "Tencent ASR provider not loaded"


def test_load_tencent_asr_stores_recognizer(monkeypatch) -> None:
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

    tencent_asr.load_tencent_asr(
        "yue", engine_model_type="16k_zh_large", config={"cfg": True}
    )

    assert created == {
        "lang": "yue",
        "engine_model_type": "16k_zh_large",
        "config": {"cfg": True},
    }
    assert tencent_asr._lang == "yue"


def test_infer_tencent_asr_rejects_invalid_items(monkeypatch) -> None:
    monkeypatch.setattr(tencent_asr, "_recognizer", _FakeTencentRecognizer())
    request = BatchInferRequest(task=InferTask.ASR, lang="yue", items=[{}])

    response = tencent_asr.infer_tencent_asr(request)

    assert response.results[0].error == "Invalid AsrBatchItem"


def test_infer_tencent_asr_surfaces_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        tencent_asr,
        "_recognizer",
        _FakeTencentRecognizer(error=RuntimeError("boom")),
    )

    response = tencent_asr.infer_tencent_asr(_valid_request())

    assert response.results[0].error == "boom"


def test_infer_tencent_asr_returns_result_payload(monkeypatch) -> None:
    monotonic_values = iter([0.0, 2.0])
    monkeypatch.setattr(tencent_asr, "_recognizer", _FakeTencentRecognizer())
    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._tencent_asr.time.monotonic",
        lambda: next(monotonic_values),
    )

    response = tencent_asr.infer_tencent_asr(_valid_request())

    assert response.results[0].error is None
    assert response.results[0].result["monologues"][0]["elements"][0]["value"] == "好"
    assert response.results[0].elapsed_s == 2.0


def test_tencent_transcribe_to_monologues_requires_loaded_recognizer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tencent_asr, "_recognizer", None)

    with pytest.raises(RuntimeError, match="not initialized"):
        tencent_asr._transcribe_to_monologues(_item())


def test_infer_tencent_asr_v2_filters_blank_elements(monkeypatch) -> None:
    monkeypatch.setattr(tencent_asr, "_recognizer", _FakeTencentRecognizer())

    response = tencent_asr.infer_tencent_asr_v2(_item())

    assert response.lang == "yue"
    assert [element.value for element in response.monologues[0].elements] == ["好"]


def test_infer_aliyun_asr_v2_delegates_to_single_item_helper(monkeypatch) -> None:
    expected = _valid_response()
    monkeypatch.setattr(
        aliyun_asr,
        "_transcribe_to_monologues",
        lambda _path: expected,
    )

    response = aliyun_asr.infer_aliyun_asr_v2(_item())

    assert response == expected


def test_infer_aliyun_asr_requires_load(monkeypatch) -> None:
    monkeypatch.setattr(aliyun_asr, "_ak_id", "")

    with pytest.raises(RuntimeError, match="load_aliyun_asr"):
        aliyun_asr.infer_aliyun_asr(_valid_request())


def test_infer_aliyun_asr_rejects_invalid_items(monkeypatch) -> None:
    monkeypatch.setattr(aliyun_asr, "_ak_id", "loaded")
    request = BatchInferRequest(task=InferTask.ASR, lang="yue", items=[{}])

    response = aliyun_asr.infer_aliyun_asr(request)

    assert response.results[0].error == "Invalid AsrBatchItem"


def test_infer_aliyun_asr_surfaces_runtime_errors(monkeypatch) -> None:
    monkeypatch.setattr(aliyun_asr, "_ak_id", "loaded")
    monkeypatch.setattr(
        aliyun_asr,
        "_transcribe_to_monologues",
        lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = aliyun_asr.infer_aliyun_asr(_valid_request())

    assert response.results[0].error == "boom"


def test_infer_aliyun_asr_returns_result_payload(monkeypatch) -> None:
    monotonic_values = iter([0.0, 3.0])
    expected = _valid_response()
    monkeypatch.setattr(aliyun_asr, "_ak_id", "loaded")
    monkeypatch.setattr(
        aliyun_asr,
        "_transcribe_to_monologues",
        lambda _path: expected,
    )
    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._aliyun_asr.time.monotonic",
        lambda: next(monotonic_values),
    )

    response = aliyun_asr.infer_aliyun_asr(_valid_request())

    assert response.results[0].error is None
    assert response.results[0].result["monologues"][0]["elements"][0]["value"] == "好"
    assert response.results[0].elapsed_s == 3.0
