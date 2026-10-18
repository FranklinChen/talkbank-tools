"""Recognition language is distinct from the timestamp companion's language.

Qwen advertises Minnan recognition under Chinese, but has no separate Minnan
forced-aligner label. Preserve ``nan`` in the request/transcript and explicitly
select the Chinese timestamp companion rather than mislabelling the session.
"""

from __future__ import annotations

from dataclasses import dataclass

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.qwen_forced_alignment import (
    QwenLanguageLabel,
    resolve_qwen_language,
)


@dataclass(frozen=True, slots=True)
class QwenAsrProfile:
    recognition_language: QwenLanguageLabel
    alignment_language: QwenLanguageLabel
    prompt: str | None
    chunk_seconds: float

    @classmethod
    def for_language(
        cls, language: LanguageCode, context_override: str | None = None
    ) -> QwenAsrProfile:
        if context_override is not None and not isinstance(context_override, str):
            raise TypeError(
                "qwen_context must be text; an empty string disables context"
            )
        if language == "nan":
            chinese = resolve_qwen_language("zho")
            return cls(
                recognition_language=chinese,
                alignment_language=chinese,
                prompt=context_override or None,
                chunk_seconds=30.0,
            )
        label = resolve_qwen_language(language)
        return cls(label, label, context_override or None, 180.0)
