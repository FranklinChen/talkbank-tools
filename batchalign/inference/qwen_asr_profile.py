"""Every language-dependent Qwen recognition decision, made once per language.

Recognition language is distinct from the timestamp companion's language.
Qwen advertises Minnan recognition under Chinese, but has no separate Minnan
forced-aligner label. Preserve ``nan`` in the request/transcript and explicitly
select the Chinese timestamp companion rather than mislabelling the session.

The profile is the one owner of these decisions. The recognizer reads its
fields and never compares the language code again, so adding a profile for
another language is one entry here rather than a search for every ``if``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.languages.cantonese._qwen_chunking import (
    MAX_FORCE_ALIGN_INPUT_SECONDS,
)
from batchalign.inference.qwen_forced_alignment import (
    QwenLanguageLabel,
    resolve_qwen_language,
)

# Taiwanese decodes in shorter windows: long windows let the recognizer drift
# into repetition on spontaneous conversational speech.
TAIWANESE_WINDOW_SECONDS = 30.0


class RecognitionPunctuation(Enum):
    """What becomes of the punctuation in the recognizer's text.

    The forced aligner returns lexical units without punctuation, so the
    recognizer's punctuation survives only if it is deliberately restored.
    """

    # Attach recognized sentence/clause punctuation to the preceding aligned
    # unit, after proving exact lexical correspondence (`PunctuatedAlignment`).
    # Used where no text segmentation model exists, so these marks are the
    # utterance boundaries: Rust routes `nan` to ASR boundaries
    # (`TranscribeSegmentation::resolve`), and the two must agree.
    RESTORE = "restore"
    # Keep only the aligned units; a later segmentation stage supplies
    # utterance boundaries.
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class QwenAsrProfile:
    recognition_language: QwenLanguageLabel
    alignment_language: QwenLanguageLabel
    prompt: str | None
    chunk_seconds: float
    punctuation: RecognitionPunctuation

    @classmethod
    def for_language(
        cls, language: LanguageCode, context_override: str | None = None
    ) -> QwenAsrProfile:
        if context_override is not None and not isinstance(context_override, str):
            raise TypeError(
                "qwen_context must be text; an empty string disables context"
            )
        prompt = context_override or None
        if language == "nan":
            chinese = resolve_qwen_language("zho")
            return cls(
                recognition_language=chinese,
                alignment_language=chinese,
                prompt=prompt,
                chunk_seconds=TAIWANESE_WINDOW_SECONDS,
                punctuation=RecognitionPunctuation.RESTORE,
            )
        label = resolve_qwen_language(language)
        return cls(
            recognition_language=label,
            alignment_language=label,
            prompt=prompt,
            chunk_seconds=MAX_FORCE_ALIGN_INPUT_SECONDS,
            punctuation=RecognitionPunctuation.DISCARD,
        )
