"""Native reports survive the default worker log level without changing admission."""

import json
import logging
from dataclasses import dataclass
from typing import Self

import numpy as np
import pytest
import torch

from batchalign.inference.qwen_forced_alignment import (
    PairedWindowAlignment,
    QwenForcedAligner,
    QwenTimingInvalid,
    resolve_qwen_language,
)
from batchalign.inference.qwen_timing import (
    AudioWindow,
    ObservedWindowSpan,
    WindowAlignment,
)


class Inputs(dict[str, torch.Tensor]):
    def to(self, device: torch.device, dtype: torch.dtype) -> Self:
        return self


@dataclass(frozen=True)
class Config:
    timestamp_token_id: int = 7


@dataclass(frozen=True)
class Output:
    logits: torch.Tensor


@dataclass(frozen=True)
class Classifier:
    logits: torch.Tensor
    config: Config = Config()
    dtype: torch.dtype = torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def __call__(self, **inputs: torch.Tensor) -> Output:
        return Output(self.logits)


@dataclass(frozen=True)
class Processor:
    end_class: int
    timestamp_segment_time: int = 80

    def prepare_forced_aligner_inputs(
        self, *, audio: object, transcript: str, language: str
    ) -> tuple[Inputs, list[list[str]]]:
        return Inputs(input_ids=torch.tensor([[7, 7]])), [["hello"]]

    def decode_forced_alignment(
        self, **inputs: object
    ) -> list[list[dict[str, str | float]]]:
        return [
            [{"text": "hello", "start_time": 0.16, "end_time": self.end_class * 0.08}]
        ]


def native_report(end_class: int) -> PairedWindowAlignment:
    logits = torch.zeros((1, 2, 16))
    logits[0, 0, 2] = 1
    logits[0, 1, end_class] = 1
    aligner = QwenForcedAligner(Processor(end_class), Classifier(logits))
    return aligner.align(
        np.zeros(16000, dtype=np.float32), "hello", resolve_qwen_language("eng")
    )


def test_native_trace_survives_warning_level_and_rejection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        reported = native_report(14)
        assert reported.native_trace is not None
        assert reported.native_trace.timestamp_classes == (2, 14)
        with pytest.raises(QwenTimingInvalid) as rejection:
            WindowAlignment(reported, AudioWindow(1.0))
    placed = rejection.value.in_recording_window(2.0, 1.0)
    assert "recording window 2..3 s" in str(placed)
    payload = json.loads(
        str(placed).split("Qwen rejected alignment diagnostic: ", 1)[1]
    )
    assert payload["reported"]["native_trace"]["timestamp_classes"] == [2, 14]
    assert payload["reported"]["words"][0]["end_s"] == 1.12
    assert payload["window_duration_s"] == 1.0


def test_valid_native_alignment_does_not_emit_a_rejection_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        reported = native_report(11)
        admitted = WindowAlignment(reported, AudioWindow(1.0))
    timing = admitted.words[0].timing
    assert isinstance(timing, ObservedWindowSpan)
    assert timing.end_s == 0.88
    assert reported.native_trace is not None
    assert not any(
        "Qwen rejected alignment diagnostic: " in record.message
        for record in caplog.records
    )
