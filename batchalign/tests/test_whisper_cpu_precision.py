"""The stock Whisper CPU precision is a per-job selection, float32 by default.

Houjun's Batchalign loads float16 on Apple Silicon CPUs to cut memory. Ours
keeps float32 until the WER comparison is in; the knob exists so that
comparison can be run through the production wrappers
(``--engine-overrides '{"asr":"whisper","whisper_cpu_dtype":"float16"}'``).
"""

from __future__ import annotations

import pytest

from batchalign.inference._domain_types import WhisperCpuPrecision


def test_no_override_means_float32() -> None:
    assert WhisperCpuPrecision.from_overrides(None) is WhisperCpuPrecision.FLOAT32
    assert WhisperCpuPrecision.from_overrides({}) is WhisperCpuPrecision.FLOAT32
    assert (
        WhisperCpuPrecision.from_overrides({"asr": "whisper"})
        is WhisperCpuPrecision.FLOAT32
    )


def test_the_extra_selects_the_precision() -> None:
    assert (
        WhisperCpuPrecision.from_overrides({"whisper_cpu_dtype": "float16"})
        is WhisperCpuPrecision.FLOAT16
    )


def test_an_unknown_value_is_refused_not_defaulted() -> None:
    with pytest.raises(ValueError, match="whisper_cpu_dtype"):
        WhisperCpuPrecision.from_overrides({"whisper_cpu_dtype": "half"})
