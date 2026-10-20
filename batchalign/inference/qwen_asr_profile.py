"""The Qwen recognition profile: labels resolved here, policy received from Rust.

Recognition language is distinct from the timestamp companion's language.
Qwen advertises Minnan recognition under Chinese, but has no separate Minnan
forced-aligner label. Preserve ``nan`` in the request/transcript and explicitly
select the Chinese timestamp companion rather than mislabelling the session.

The decoding window and the fate of recognized punctuation are NOT decided
here. The control plane decides them once per language, from the same table
that routes transcription segmentation, and sends them to the worker as a
``QwenRecognitionV2``. This module only translates that policy into the
recognizer's terms, so the route that relies on recognizer punctuation and the
recognizer that restores it cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.languages.cantonese._qwen_chunking import (
    MAX_FORCE_ALIGN_INPUT_SECONDS,
)
from batchalign.inference.qwen_forced_alignment import (
    QwenLanguageLabel,
    resolve_qwen_recognition_language,
)
from batchalign.worker._types_v2 import (
    BoundedSecondsWindowV2,
    ChunkerDefaultWindowV2,
    QwenRecognitionV2,
    RecognitionPunctuationV2,
)


@dataclass(frozen=True, slots=True)
class QwenAsrProfile:
    recognition_language: QwenLanguageLabel
    alignment_language: QwenLanguageLabel
    prompt: str | None
    chunk_seconds: float
    punctuation: RecognitionPunctuationV2

    @classmethod
    def for_language(
        cls,
        language: LanguageCode,
        recognition: QwenRecognitionV2,
        context_override: str | None = None,
    ) -> QwenAsrProfile:
        if context_override is not None and not isinstance(context_override, str):
            raise TypeError(
                "qwen_context must be text; an empty string disables context"
            )
        # Model vocabulary only (Taiwanese is recognized under the Chinese
        # label), from the label table beside the aligner's.
        label = resolve_qwen_recognition_language(language)
        match recognition.window:
            case ChunkerDefaultWindowV2():
                chunk_seconds = MAX_FORCE_ALIGN_INPUT_SECONDS
            case BoundedSecondsWindowV2(seconds=seconds):
                chunk_seconds = float(seconds)
            case unreachable:
                assert_never(unreachable)
        return cls(
            recognition_language=label,
            alignment_language=label,
            prompt=context_override or None,
            chunk_seconds=chunk_seconds,
            punctuation=recognition.punctuation,
        )
