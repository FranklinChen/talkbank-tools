"""Tencent Cloud ASR provider for built-in HK/Cantonese engines.

The ASR load builds a :class:`TencentAsr` around one ``TencentRecognizer``
(``tencent_api.py``) and records it in the worker's load registry; a request
transcribes through that object. Nothing is held at module level.
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse

from ._asr_types import admit_provider_monologues
from ._tencent_api import TencentRecognizer

L = logging.getLogger("batchalign.hk.tencent")


@dataclass(frozen=True, slots=True)
class TencentAsr:
    """The Tencent recognizer one ASR load built."""

    recognizer: TencentRecognizer

    def transcribe(self, item: AsrBatchItem) -> MonologueAsrResponse:
        """Recognize one provider media item into raw speaker monologues."""
        details = self.recognizer.transcribe(
            item.audio_path, diarization=item.diarization
        )
        return admit_provider_monologues(self.recognizer.monologues(details), item.lang)


def load_tencent_asr(
    lang: LanguageCode,
    *,
    engine_model_type: str,
    config: configparser.ConfigParser | None = None,
) -> TencentAsr:
    """Build the Tencent recognizer for one ASR load.

    Takes no ``engine_overrides``: it has none to read. Tencent's one
    configurable knob is the engine-model type, which the Rust control plane
    chooses (through the one ISO 639-3 conversion) and passes as
    ``engine_model_type``.
    """
    recognizer = TencentRecognizer(
        lang=lang, engine_model_type=engine_model_type, config=config
    )
    L.info("Tencent ASR provider loaded for lang=%s, model=%s", lang, engine_model_type)
    return TencentAsr(recognizer)


__all__ = ["TencentAsr", "load_tencent_asr"]
