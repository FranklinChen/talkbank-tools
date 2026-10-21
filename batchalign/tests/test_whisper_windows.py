"""The real postprocess adapter preserves independent window ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
from transformers import AutomaticSpeechRecognitionPipeline

from batchalign.inference.asr import infer_whisper_prepared_audio
from batchalign.inference.whisper_windows import WindowOwnedWhisperPipeline
from batchalign.tests._asr_model_pins import loaded_identity
from batchalign.worker._types_v2 import AsrBackendV2, WhisperWindowsV2


def test_window_postprocess_never_passes_neighbors_to_text_merger(monkeypatch):
    calls = []

    def decode(_self, outputs, **kwargs):
        assert len(outputs) == 1
        assert "stride" not in outputs[0]
        calls.append(outputs[0]["tokens"])
        return {
            "text": " again",
            "chunks": [{"text": " again", "timestamp": (1.0, 1.5)}],
        }

    monkeypatch.setattr(AutomaticSpeechRecognitionPipeline, "postprocess", decode)
    pipe = object.__new__(WindowOwnedWhisperPipeline)
    pipe.feature_extractor = SimpleNamespace(sampling_rate=16000)
    outputs = [
        {"tokens": "first", "stride": (400000, 0, 48000)},
        {"tokens": "second", "stride": (200000, 48000, 0)},
    ]
    raw = pipe.postprocess(outputs, return_timestamps="word")
    assert calls == ["first", "second"]
    assert outputs[0]["stride"] == (400000, 0, 48000)
    owned = raw["window_observations"]
    assert [w.start_sample for w in owned.windows] == [0, 304000]
    assert [w.spans[0].start_s for w in owned.windows] == [1.0, 1.0]

    class Handle:
        sample_rate = 16000
        model_identity = loaded_identity(AsrBackendV2.LOCAL_WHISPER)

        def gen_kwargs(self):
            return {}

        def __call__(self, *_args, **_kwargs):
            return raw

    result = infer_whisper_prepared_audio(Handle(), np.zeros(504000), "eng")
    assert isinstance(result.chunks, WhisperWindowsV2)
    assert result.chunks == owned
    with pytest.raises(ValueError, match="prepared audio"):
        infer_whisper_prepared_audio(Handle(), np.zeros(1000), "eng")


def test_missing_window_ownership_is_not_inferred():
    pipe = object.__new__(WindowOwnedWhisperPipeline)
    with pytest.raises(ValueError, match="stride"):
        pipe.postprocess([{"tokens": "unknown origin"}])


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_sample", 1),
        ("sample_count", 0),
        ("right_context", 1),
    ],
)
def test_invalid_window_ownership_is_refused(field, value):
    window = dict(
        start_sample=0, sample_count=16000, left_context=0, right_context=0, spans=[]
    )
    window[field] = value
    with pytest.raises(ValueError):
        WhisperWindowsV2(sample_rate_hz=16000, windows=[window])
