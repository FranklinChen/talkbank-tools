"""FunASR ASR inference provider for built-in HK/Cantonese engines.

The ASR load builds a :class:`FunAudioAsr` around the loaded recognizer and
records it in the worker's load registry; a request transcribes through that
object. Nothing is held at module level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse

from ._asr_types import admit_provider_monologues
from ._common import EngineOverrides
from ._funaudio_common import FunAsrSnapshots, FunAudioRecognizer

L = logging.getLogger("batchalign.hk.funaudio_asr")


@dataclass(frozen=True, slots=True)
class FunAudioAsr:
    """The FunASR recognizer one ASR load built."""

    recognizer: FunAudioRecognizer

    def transcribe(self, item: AsrBatchItem) -> MonologueAsrResponse:
        """Transcribe one provider media item into raw speaker monologues."""
        payload, _timed_words = self.recognizer.transcribe(item.audio_path)
        return admit_provider_monologues(payload, item.lang)


def load_funaudio_asr(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    snapshots: FunAsrSnapshots,
    label: str,
) -> FunAudioAsr:
    """Load the FunASR recognizer for one ASR load.

    Called after the worker has materialized the composition's snapshots on
    local disk (``_model_loading.asr``), so the model is in memory when the
    load completes and no request can trigger a download. ``label`` is the
    checkpoint's model id, for logs.

    The composition (SenseVoice or Paraformer) and its checkpoint come from
    ``snapshots``, which the pinned plan decided. One ``engine_overrides`` key
    is read here: ``"funaudio_device"``, the torch device string (default
    ``"cpu"``). ``"funaudio_model"`` is resolved by the control plane into the
    pinned composition before the worker starts, so it is not read again.
    """
    device = "cpu"
    if engine_overrides and "funaudio_device" in engine_overrides:
        device = str(engine_overrides["funaudio_device"])

    recognizer = FunAudioRecognizer.load(
        snapshots, lang=lang, device=device, label=label
    )
    L.info(
        "FunAudio ASR recognizer loaded: lang=%s, model=%s, device=%s",
        lang,
        label,
        device,
    )
    return FunAudioAsr(recognizer)


__all__ = ["FunAudioAsr", "load_funaudio_asr"]
