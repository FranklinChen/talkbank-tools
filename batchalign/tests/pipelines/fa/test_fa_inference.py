# affects: batchalign/inference/fa/**
# affects: crates/batchalign-fa/**
"""Tests for the thin Python forced-alignment inference boundary."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from batchalign.device import DevicePolicy
from batchalign.inference.fa import (
    FaInferItem,
    infer_wave2vec_fa,
    infer_whisper_fa,
)


class _FakeAudioFile:
    """Simple audio file double returning deterministic chunks."""

    def __init__(self, audio_path: str) -> None:
        self.audio_path = audio_path
        self.chunk_calls: list[tuple[int, int]] = []

    def chunk(self, start_ms: int, end_ms: int) -> torch.Tensor:
        self.chunk_calls.append((start_ms, end_ms))
        return torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)


class _FeatureBatch(dict[str, torch.Tensor]):
    """Mapping returned by the fake Whisper processor."""

    def to(self, _device: torch.device) -> _FeatureBatch:
        return self


class _FakeWhisperProcessor:
    """Minimal Whisper processor double."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.feature_extractor = SimpleNamespace(hop_length=160)
        self.tokenizer = SimpleNamespace(prefix_tokens=[1, 2], eos_token_id=3)

    def __call__(
        self, *, audio, text: str, sampling_rate: int, return_tensors: str
    ) -> _FeatureBatch:
        self.calls.append(
            {
                "audio": audio.clone(),
                "text": text,
                "sampling_rate": sampling_rate,
                "return_tensors": return_tensors,
            }
        )
        return _FeatureBatch(
            {
                "labels": torch.tensor([[1, 2, 10, 20, 3]], dtype=torch.int64),
                "input_features": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            }
        )

    def decode(self, token_id: int | torch.Tensor) -> str:
        if int(token_id) in (1, 2, 3):
            return {
                1: "<|startoftranscript|>",
                2: "<|notimestamps|>",
                3: "<|endoftext|>",
            }[int(token_id)]
        return f"tok-{int(token_id)}"


class _FakeWhisperModel:
    """Minimal Whisper model double exposing the alignment-head seam."""

    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.generation_config = SimpleNamespace(alignment_heads=[(0, 0)])
        self.config = SimpleNamespace(median_filter_width=1)

    def parameters(self):
        yield self._parameter

    def __call__(self, **_kwargs):
        return SimpleNamespace(
            cross_attentions=[
                torch.tensor(
                    [
                        [
                            [
                                [9.0, 8.0, 7.0, 6.0],
                                [6.0, 7.0, 8.0, 9.0],
                                [1.0, 2.0, 3.0, 4.0],
                                [2.0, 3.0, 4.0, 5.0],
                                [3.0, 4.0, 5.0, 6.0],
                            ]
                        ]
                    ]
                )
            ]
        )


class _FakeWave2VecModel:
    """Minimal Wave2Vec model double."""

    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.tensor(1.0))

    def parameters(self):
        yield self._parameter

    def __call__(self, audio: torch.Tensor):
        assert audio.shape == (1, 4)
        return torch.ones((1, 4, 8), dtype=torch.float32), None


class _Span:
    """Tiny span object matching torchaudio merge-token output."""

    def __init__(self, start: int, end: int, score: float = 1.0) -> None:
        self.start = start
        self.end = end
        self.score = score


def _install_whisper_alignment_helpers(monkeypatch) -> None:
    """Install fake DTW/median-filter helpers for Whisper FA tests."""

    module = ModuleType("transformers.models.whisper.generation_whisper")
    module._dynamic_time_warping = lambda _matrix: (
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 2, 3], dtype=np.int64),
    )
    module._median_filter = lambda weights, _width: weights
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers.models.whisper.generation_whisper",
        module,
    )


def _install_torchaudio_alignment_helpers(
    monkeypatch,
    *,
    dictionary: dict[str, int] | None = None,
    spans: list[_Span] | None = None,
    capture: dict[str, Any] | None = None,
) -> None:
    """Install fake torchaudio alignment helpers for Wave2Vec FA tests."""

    if dictionary is None:
        dictionary = {"h": 1, "i": 2, "a": 3, "*": 28}
    if spans is None:
        spans = [
            _Span(10, 20),
            _Span(20, 30),
            _Span(30, 40),
        ]

    functional = ModuleType("torchaudio.functional")

    def _forced_align(emission, transcript):
        if capture is not None:
            capture["transcript"] = transcript.clone()
        return (
            torch.tensor([list(range(len(spans)))], dtype=torch.int64),
            torch.zeros((1, len(spans)), dtype=torch.float32),
        )

    functional.forced_align = _forced_align
    functional.merge_tokens = lambda alignments, scores: spans

    pipelines = ModuleType("torchaudio.pipelines")
    pipelines.MMS_FA = SimpleNamespace(get_dict=lambda: dictionary)

    torchaudio = ModuleType("torchaudio")
    torchaudio.functional = functional
    torchaudio.pipelines = pipelines

    monkeypatch.setitem(__import__("sys").modules, "torchaudio", torchaudio)
    monkeypatch.setitem(__import__("sys").modules, "torchaudio.functional", functional)
    monkeypatch.setitem(__import__("sys").modules, "torchaudio.pipelines", pipelines)


def _fa_item(
    *,
    words: list[str],
    audio_path: str = "/tmp/audio.wav",
    audio_start_ms: int = 0,
    audio_end_ms: int = 4000,
) -> dict[str, Any]:
    """Build one valid raw FA item."""

    return {
        "words": words,
        "word_ids": [f"u0:w{i}" for i in range(len(words))],
        "word_utterance_indices": [0] * len(words),
        "word_utterance_word_indices": list(range(len(words))),
        "audio_path": audio_path,
        "audio_start_ms": audio_start_ms,
        "audio_end_ms": audio_end_ms,
    }


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("word_ids", ["u0:w0"], "word_ids length mismatch"),
        ("word_utterance_indices", [0], "word_utterance_indices length mismatch"),
        (
            "word_utterance_word_indices",
            [0],
            "word_utterance_word_indices length mismatch",
        ),
    ],
)
def test_fa_infer_item_rejects_parallel_array_mismatch(
    field_name: str,
    field_value: list[Any],
    message: str,
) -> None:
    """FA payload validation should reject mismatched parallel arrays."""

    payload = _fa_item(words=["hello", "world"])
    payload[field_name] = field_value

    with pytest.raises(Exception, match=message):
        FaInferItem.model_validate(payload)


@pytest.mark.parametrize(
    (
        "force_cpu",
        "cuda_available",
        "mps_available",
        "expected_device",
        "expected_dtype",
    ),
    [
        (True, False, False, "cpu", torch.float32),
        (False, True, False, "cuda", torch.float16),
        # MPS excluded since 2026-04-05 (AGXG14X kernel deadlock), MPS
        # availability is ignored; the loader falls through to CPU.
        (False, False, True, "cpu", torch.float32),
        (False, False, False, "cpu", torch.float32),
    ],
)
def test_load_whisper_fa_selects_device_and_dtype(
    monkeypatch,
    force_cpu: bool,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
    expected_dtype,
) -> None:
    """Whisper FA loading should choose the expected device and dtype policy."""

    captured: dict[str, Any] = {}

    class _LoadedModel:
        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def eval(self) -> None:
            captured["eval"] = True

    class _ModelLoader:
        @staticmethod
        def from_pretrained(model: str, *, attn_implementation: str, torch_dtype):
            captured["model_name"] = model
            captured["attn_implementation"] = attn_implementation
            captured["torch_dtype"] = torch_dtype
            loaded = _LoadedModel()
            captured["loaded_model"] = loaded
            return loaded

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(model: str):
            captured["processor_name"] = model
            return SimpleNamespace(name=model)

    transformers = ModuleType("transformers")
    transformers.WhisperForConditionalGeneration = _ModelLoader
    transformers.WhisperProcessor = _ProcessorLoader
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_available)
    monkeypatch.setattr(
        "batchalign.inference.audio.bind_whisper_token_timestamp_extractor",
        lambda model: captured.setdefault("bound_model", model),
    )
    monkeypatch.setattr(
        "batchalign.inference.types.WhisperFAHandle",
        lambda model, processor, sample_rate: SimpleNamespace(
            model=model,
            processor=processor,
            sample_rate=sample_rate,
        ),
    )

    from batchalign.inference.fa import load_whisper_fa

    handle = load_whisper_fa(
        model="openai/fake-whisper",
        target_sample_rate=22050,
        device_policy=DevicePolicy(force_cpu=force_cpu),
    )

    assert handle.sample_rate == 22050
    assert handle.processor.name == "openai/fake-whisper"
    assert captured["model_name"] == "openai/fake-whisper"
    assert captured["processor_name"] == "openai/fake-whisper"
    assert captured["attn_implementation"] == "eager"
    assert captured["torch_dtype"] == expected_dtype
    assert captured["device"].type == expected_device
    assert captured["bound_model"] is captured["loaded_model"]
    assert captured["eval"] is True


@pytest.mark.parametrize(
    ("force_cpu", "cuda_available", "mps_available", "expected_device"),
    [
        (True, False, False, "cpu"),
        (False, True, False, "cuda"),
        # MPS excluded since 2026-04-05 (AGXG14X kernel deadlock).
        (False, False, True, "cpu"),
        (False, False, False, "cpu"),
    ],
)
def test_load_wave2vec_fa_selects_expected_device(
    monkeypatch,
    force_cpu: bool,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
) -> None:
    """Wave2Vec FA loading should choose the expected device path."""

    captured: dict[str, Any] = {}

    class _LoadedModel:
        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def float(self):
            return self

    class _Bundle:
        def get_model(self):
            captured["bundle_get_model"] = True
            return _LoadedModel()

    torchaudio = ModuleType("torchaudio")
    torchaudio.pipelines = SimpleNamespace(MMS_FA=_Bundle())
    monkeypatch.setitem(__import__("sys").modules, "torchaudio", torchaudio)
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_available)
    monkeypatch.setattr(
        "batchalign.inference.types.Wave2VecFAHandle",
        lambda model, sample_rate: SimpleNamespace(
            model=model, sample_rate=sample_rate
        ),
    )

    from batchalign.inference.fa import load_wave2vec_fa

    handle = load_wave2vec_fa(
        target_sample_rate=8000, device_policy=DevicePolicy(force_cpu=force_cpu)
    )

    assert handle.sample_rate == 8000
    assert captured["bundle_get_model"] is True
    assert captured["device"].type == expected_device


@pytest.mark.parametrize(
    (
        "force_cpu",
        "cuda_available",
        "mps_available",
        "expected_device",
        "expect_float32_cast",
    ),
    [
        (True, False, False, "cpu", False),
        (False, True, False, "cuda", False),
        # MPS excluded since 2026-04-05 (AGXG14X kernel deadlock); the loader
        # never reaches MPS, so the MPS-specific .float() cast is not exercised.
        (False, False, True, "cpu", False),
        (False, False, False, "cpu", False),
    ],
)
def test_load_wave2vec_fa_forces_float32_on_mps(
    monkeypatch,
    force_cpu: bool,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
    expect_float32_cast: bool,
) -> None:
    """Wave2Vec FA device selection and dtype verification.

    Historically, MPS required an explicit .float() cast to avoid bfloat16
    crashes. Since the MPS exclusion (2026-04-05), all rows now expect CPU
    when MPS-only, and .float() is never called.
    """

    captured: dict[str, Any] = {}

    class _LoadedModel:
        """Track .to() and .float() calls to verify dtype coercion."""

        def __init__(self) -> None:
            self.float_called = False

        def to(self, device: torch.device):
            captured["device"] = device
            return self

        def float(self):
            self.float_called = True
            captured["float_called"] = True
            return self

    class _Bundle:
        def get_model(self):
            model = _LoadedModel()
            captured["model"] = model
            return model

    torchaudio = ModuleType("torchaudio")
    torchaudio.pipelines = SimpleNamespace(MMS_FA=_Bundle())
    monkeypatch.setitem(__import__("sys").modules, "torchaudio", torchaudio)
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_available)
    monkeypatch.setattr(
        "batchalign.inference.types.Wave2VecFAHandle",
        lambda model, sample_rate: SimpleNamespace(
            model=model, sample_rate=sample_rate
        ),
    )

    from batchalign.inference.fa import load_wave2vec_fa

    load_wave2vec_fa(
        target_sample_rate=16000, device_policy=DevicePolicy(force_cpu=force_cpu)
    )

    assert captured["device"].type == expected_device
    if expect_float32_cast:
        assert captured.get("float_called") is True, (
            "Wave2Vec FA on this device must call .float() to force float32"
        )
    else:
        assert captured.get("float_called") is not True, (
            f"float() should not be called on {expected_device}"
        )


def test_infer_whisper_fa_decodes_tokens_from_alignment_heads(monkeypatch) -> None:
    """Whisper FA should use alignment heads and processor decoding."""

    _install_whisper_alignment_helpers(monkeypatch)

    handle = SimpleNamespace(
        model=_FakeWhisperModel(),
        processor=_FakeWhisperProcessor(),
        sample_rate=16000,
    )

    result = infer_whisper_fa(
        handle,
        torch.zeros(1280, dtype=torch.float32),
        "ab",
    )

    # The text reaches the processor exactly as given. This asserted "a b" from
    # a `pauses=True` argument until 2026-08-14; that reshaping is now
    # `FaTextModeV2::CharSpaced`, applied in Rust before this is called, and is
    # covered there.
    assert handle.processor.calls[0]["text"] == "ab"
    assert handle.processor.calls[0]["sampling_rate"] == 16000
    assert result == [
        ("<|startoftranscript|>", 0.0),
        ("<|notimestamps|>", 0.0),
        ("tok-10", 0.0),
        ("tok-20", 0.04),
        ("<|endoftext|>", 0.06),
    ]


def test_whisper_fa_excludes_padding_before_normalization_and_dtw(monkeypatch) -> None:
    _install_whisper_alignment_helpers(monkeypatch)
    module = __import__("sys").modules["transformers.models.whisper.generation_whisper"]
    captured: list[tuple[int, ...]] = []

    def dtw(matrix):
        captured.append(matrix.shape)
        return np.asarray([0, 1, 2]), np.asarray([0, 1, 1])

    module._dynamic_time_warping = dtw
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=_FakeWhisperProcessor(), sample_rate=16000
    )
    result = infer_whisper_fa(handle, torch.zeros(640), "ab")
    assert captured == [(3, 2)], (
        "conditioning rows and padded frames must both be absent"
    )
    assert result == [
        ("<|startoftranscript|>", 0.0),
        ("<|notimestamps|>", 0.0),
        ("tok-10", 0.0),
        ("tok-20", 0.02),
        ("<|endoftext|>", 0.02),
    ]


def test_live_whisper_host_binds_dtw_rows_to_lexical_labels(monkeypatch) -> None:
    """The actual worker callback must exclude conditioning before DTW."""
    from batchalign.worker._fa_v2 import build_default_fa_execution_host_v2

    _install_whisper_alignment_helpers(monkeypatch)
    module = __import__("sys").modules["transformers.models.whisper.generation_whisper"]
    captured = []

    def dtw(matrix):
        captured.append(matrix.clone())
        return np.asarray([0, 1, 2]), np.asarray([0, 1, 3])

    module._dynamic_time_warping = dtw
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=_FakeWhisperProcessor(), sample_rate=16000
    )
    host = build_default_fa_execution_host_v2(
        whisper_model=handle, wave2vec_model=None, canto_host=None, qwen_host=None
    )
    result = host.whisper_runner(np.zeros(1280, dtype=np.float32), "ab")
    assert captured[0].shape == (3, 4)
    assert result[2:4] == [("tok-10", 0.0), ("tok-20", 0.02)]
    # Conditioning values deliberately differ from lexical rows. Normalizing
    # them together would change these costs even if sliced before DTW but
    # after normalization.
    lexical = handle.model().cross_attentions[0][0, 0, 2:]
    std, mean = torch.std_mean(lexical, dim=-2, keepdim=True, unbiased=False)
    assert torch.equal(captured[0], -(lexical - mean) / std)


@pytest.mark.parametrize("labels", [[9, 2, 10, 20, 3], [1, 2, 10, 20, 9], [1, 2, 3]])
def test_whisper_fa_refuses_unbound_decoder_labels(monkeypatch, labels) -> None:
    _install_whisper_alignment_helpers(monkeypatch)

    class ChangedLabels(_FakeWhisperProcessor):
        def __call__(self, **kwargs):
            features = super().__call__(**kwargs)
            features["labels"] = torch.tensor([labels])
            return features

    processor = ChangedLabels()
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=processor, sample_rate=16000
    )
    with pytest.raises(ValueError, match=r"decoder labels|tokenizer prefix"):
        infer_whisper_fa(handle, torch.zeros(1280), "ab")


def test_whisper_fa_refuses_incomplete_dtw_label_coverage(monkeypatch) -> None:
    _install_whisper_alignment_helpers(monkeypatch)
    module = __import__("sys").modules["transformers.models.whisper.generation_whisper"]
    module._dynamic_time_warping = lambda matrix: (
        np.asarray([0, 1]),
        np.asarray([0, 1]),
    )
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=_FakeWhisperProcessor(), sample_rate=16000
    )
    with pytest.raises(ValueError, match="every admitted decoder label"):
        infer_whisper_fa(handle, torch.zeros(1280), "ab")


@pytest.mark.parametrize("sample_count", [0, 319, 1600])
def test_whisper_fa_refuses_audio_without_supported_attention_extent(
    monkeypatch, sample_count: int
) -> None:
    _install_whisper_alignment_helpers(monkeypatch)
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=_FakeWhisperProcessor(), sample_rate=16000
    )
    with pytest.raises(ValueError, match="encoder"):
        infer_whisper_fa(handle, torch.zeros(sample_count), "ab")


def test_whisper_fa_refuses_dtw_indices_outside_observed_audio(monkeypatch) -> None:
    _install_whisper_alignment_helpers(monkeypatch)
    handle = SimpleNamespace(
        model=_FakeWhisperModel(), processor=_FakeWhisperProcessor(), sample_rate=16000
    )
    with pytest.raises(ValueError, match="admitted acoustic window"):
        infer_whisper_fa(handle, torch.zeros(640), "ab")


def test_infer_wave2vec_fa_converts_spans_to_milliseconds(monkeypatch) -> None:
    """Wave2Vec FA should retain timings and duration-weighted model scores."""

    _install_torchaudio_alignment_helpers(
        monkeypatch,
        spans=[
            _Span(10, 20, 0.5),
            _Span(20, 30, 0.9),
            _Span(30, 40, 0.8),
        ],
    )

    handle = SimpleNamespace(model=_FakeWave2VecModel(), sample_rate=1000)

    result = infer_wave2vec_fa(
        handle,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        ["hi", "a"],
    )

    assert result == [("hi", (10, 30), 0.7), ("a", (30, 40), 0.8)]


def test_infer_wave2vec_fa_strips_blank_mapped_chars_from_targets(monkeypatch) -> None:
    """Wave2Vec FA must remove chars whose MMS dictionary maps to CTC blank."""

    captured: dict[str, Any] = {}
    _install_torchaudio_alignment_helpers(
        monkeypatch,
        dictionary={"a": 1, "b": 2, "h": 3, "i": 4, "-": 0, "*": 28},
        spans=[
            _Span(10, 20),
            _Span(20, 30),
            _Span(30, 40),
            _Span(40, 50),
        ],
        capture=captured,
    )

    handle = SimpleNamespace(model=_FakeWave2VecModel(), sample_rate=1000)

    result = infer_wave2vec_fa(
        handle,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        ["a-b", "hi"],
    )

    assert captured["transcript"].tolist() == [[1, 2, 3, 4]]
    assert result == [("a-b", (10, 30), 1.0), ("hi", (30, 50), 1.0)]


def test_infer_wave2vec_fa_uses_wildcard_when_blank_sanitization_empties_word(
    monkeypatch,
) -> None:
    """Wave2Vec FA should keep word slots even if blank-sanitization removes all chars."""

    captured: dict[str, Any] = {}
    _install_torchaudio_alignment_helpers(
        monkeypatch,
        dictionary={"a": 1, "-": 0, "*": 28},
        spans=[
            _Span(10, 20),
            _Span(20, 30),
        ],
        capture=captured,
    )

    handle = SimpleNamespace(model=_FakeWave2VecModel(), sample_rate=1000)

    result = infer_wave2vec_fa(
        handle,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        ["-", "a"],
    )

    assert captured["transcript"].tolist() == [[28, 1]]
    assert result == [("-", (10, 20), 1.0), ("a", (20, 30), 1.0)]
